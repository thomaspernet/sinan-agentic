"""Tests for the ``sinan_agentic_core.llm`` provider config layer."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import SecretStr, ValidationError

from sinan_agentic_core.llm import (
    AzureOpenAIProviderConfig,
    OpenAIProviderConfig,
    configure_llm_provider,
    load_llm_provider_config,
    parse_llm_provider_config,
    resolve_openai_client,
)
from sinan_agentic_core.llm import factory as factory_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_sdk_calls(monkeypatch):
    """Replace the SDK setters with no-op recorders."""
    calls: dict[str, object] = {}

    def fake_set_default_openai_client(client, use_for_tracing=True):
        calls["client"] = client
        calls["use_for_tracing"] = use_for_tracing

    def fake_set_default_openai_api(api):
        calls["api"] = api

    def fake_set_tracing_disabled(disabled):
        calls["tracing_disabled"] = disabled

    monkeypatch.setattr(factory_module, "set_default_openai_client", fake_set_default_openai_client)
    monkeypatch.setattr(factory_module, "set_default_openai_api", fake_set_default_openai_api)
    monkeypatch.setattr(factory_module, "set_tracing_disabled", fake_set_tracing_disabled)
    return calls


# ---------------------------------------------------------------------------
# parse_llm_provider_config — discriminator and defaults
# ---------------------------------------------------------------------------


class TestParseProviderConfig:
    def test_openai_defaults(self):
        cfg = parse_llm_provider_config({"provider": "openai", "api_key": "sk-test"})
        assert isinstance(cfg, OpenAIProviderConfig)
        assert cfg.api_mode == "responses"
        assert cfg.use_for_tracing is True
        assert cfg.disable_tracing is False
        assert cfg.api_key.get_secret_value() == "sk-test"

    def test_azure_defaults(self):
        cfg = parse_llm_provider_config(
            {
                "provider": "azure_openai",
                "api_key": "az-test",
                "azure_endpoint": "https://example.openai.azure.com",
                "api_version": "2024-08-01-preview",
            }
        )
        assert isinstance(cfg, AzureOpenAIProviderConfig)
        assert cfg.api_mode == "chat_completions"
        assert cfg.disable_tracing is True
        assert cfg.use_for_tracing is False

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValidationError):
            parse_llm_provider_config({"provider": "bedrock", "api_key": "x"})

    def test_missing_provider_rejected(self):
        with pytest.raises(ValidationError):
            parse_llm_provider_config({"api_key": "sk-test"})

    def test_azure_requires_endpoint(self):
        with pytest.raises(ValidationError):
            parse_llm_provider_config(
                {
                    "provider": "azure_openai",
                    "api_key": "az",
                    "api_version": "2024-08-01-preview",
                }
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            parse_llm_provider_config({"provider": "openai", "api_key": "sk", "unknown": "x"})


# ---------------------------------------------------------------------------
# configure_llm_provider — SDK wiring
# ---------------------------------------------------------------------------


class TestConfigureLLMProvider:
    def test_openai_wires_responses_api_and_tracing(self, captured_sdk_calls):
        cfg = OpenAIProviderConfig(api_key=SecretStr("sk-test"))
        client = configure_llm_provider(cfg)

        assert isinstance(client, AsyncOpenAI)
        assert not isinstance(client, AsyncAzureOpenAI)
        assert captured_sdk_calls["client"] is client
        assert captured_sdk_calls["api"] == "responses"
        assert captured_sdk_calls["use_for_tracing"] is True
        assert "tracing_disabled" not in captured_sdk_calls

    def test_openai_with_base_url_and_org(self, captured_sdk_calls):
        cfg = OpenAIProviderConfig(
            api_key=SecretStr("sk-test"),
            base_url="https://proxy.example.com/v1",
            organization="org-abc",
            project="proj-xyz",
        )
        client = configure_llm_provider(cfg)
        # AsyncOpenAI normalises base_url with a trailing slash
        assert str(client.base_url).startswith("https://proxy.example.com/v1")
        assert client.organization == "org-abc"
        assert client.project == "proj-xyz"

    def test_azure_wires_chat_completions_and_disables_tracing(self, captured_sdk_calls):
        cfg = AzureOpenAIProviderConfig(
            api_key=SecretStr("az-test"),
            azure_endpoint="https://example.openai.azure.com",
            api_version="2024-08-01-preview",
            azure_deployment="gpt-4o",
        )
        client = configure_llm_provider(cfg)

        assert isinstance(client, AsyncAzureOpenAI)
        assert captured_sdk_calls["client"] is client
        assert captured_sdk_calls["api"] == "chat_completions"
        assert captured_sdk_calls["use_for_tracing"] is False
        assert captured_sdk_calls["tracing_disabled"] is True

    def test_disable_tracing_flag_respected_for_openai(self, captured_sdk_calls):
        cfg = OpenAIProviderConfig(
            api_key=SecretStr("sk-test"),
            disable_tracing=True,
            use_for_tracing=False,
        )
        configure_llm_provider(cfg)
        assert captured_sdk_calls["tracing_disabled"] is True


# ---------------------------------------------------------------------------
# resolve_openai_client — read back the configured default
# ---------------------------------------------------------------------------


class TestResolveOpenAIClient:
    def test_returns_configured_default_client(self, monkeypatch):
        configured = object()
        monkeypatch.setattr(factory_module, "get_default_openai_client", lambda: configured)
        assert resolve_openai_client() is configured

    def test_azure_client_survives_the_round_trip(self, monkeypatch):
        """An Azure deployment resolves back as AsyncAzureOpenAI, not a plain
        AsyncOpenAI rebuilt from scratch. Regression for #35.
        """
        cfg = AzureOpenAIProviderConfig(
            api_key=SecretStr("az-test"),
            azure_endpoint="https://example.openai.azure.com",
            api_version="2024-08-01-preview",
            azure_deployment="gpt-4o",
        )
        client = factory_module._build_client(cfg)
        monkeypatch.setattr(factory_module, "get_default_openai_client", lambda: client)

        resolved = resolve_openai_client()
        assert resolved is client
        assert isinstance(resolved, AsyncAzureOpenAI)

    def test_builds_bare_async_openai_when_no_default(self, monkeypatch):
        """With no provider configured, build AsyncOpenAI() with no kwargs so the
        SDK reads OPENAI_API_KEY. Regression for #35 — must not pass api_key=None.
        """
        built: list[dict[str, object]] = []

        def fake_async_openai(**kwargs):
            built.append(kwargs)
            return "fresh-client"

        monkeypatch.setattr(factory_module, "get_default_openai_client", lambda: None)
        monkeypatch.setattr(factory_module, "AsyncOpenAI", fake_async_openai)

        assert resolve_openai_client() == "fresh-client"
        assert built == [{}]


# ---------------------------------------------------------------------------
# load_llm_provider_config — YAML + ${VAR} interpolation
# ---------------------------------------------------------------------------


class TestLoadLLMProviderConfig:
    def test_loads_openai_with_env_interpolation(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MY_OPENAI_KEY", "sk-from-env")
        path = tmp_path / "llm.yaml"
        path.write_text(textwrap.dedent("""
                llm:
                  provider: openai
                  api_key: ${MY_OPENAI_KEY}
                """).strip())
        cfg = load_llm_provider_config(path)
        assert isinstance(cfg, OpenAIProviderConfig)
        assert cfg.api_key.get_secret_value() == "sk-from-env"

    def test_loads_azure_with_mixed_string_interpolation(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("AZ_KEY", "az-secret")
        monkeypatch.setenv("AZ_RESOURCE", "my-resource")
        path = tmp_path / "llm.yaml"
        path.write_text(textwrap.dedent("""
                llm:
                  provider: azure_openai
                  api_key: ${AZ_KEY}
                  azure_endpoint: https://${AZ_RESOURCE}.openai.azure.com
                  api_version: 2024-08-01-preview
                  azure_deployment: gpt-4o
                """).strip())
        cfg = load_llm_provider_config(path)
        assert isinstance(cfg, AzureOpenAIProviderConfig)
        assert cfg.api_key.get_secret_value() == "az-secret"
        assert cfg.azure_endpoint == "https://my-resource.openai.azure.com"

    def test_missing_env_var_raises_key_error(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        path = tmp_path / "llm.yaml"
        path.write_text(textwrap.dedent("""
                llm:
                  provider: openai
                  api_key: ${MISSING_VAR}
                """).strip())
        with pytest.raises(KeyError, match="MISSING_VAR"):
            load_llm_provider_config(path)

    def test_custom_section(self, tmp_path: Path):
        path = tmp_path / "llm.yaml"
        path.write_text(textwrap.dedent("""
                openai_cfg:
                  provider: openai
                  api_key: sk-flat
                """).strip())
        cfg = load_llm_provider_config(path, section="openai_cfg")
        assert isinstance(cfg, OpenAIProviderConfig)
        assert cfg.api_key.get_secret_value() == "sk-flat"

    def test_missing_section_raises(self, tmp_path: Path):
        path = tmp_path / "llm.yaml"
        path.write_text("other:\n  foo: 1\n")
        with pytest.raises(KeyError, match="llm"):
            load_llm_provider_config(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_llm_provider_config(tmp_path / "nope.yaml")

    def test_section_must_be_mapping(self, tmp_path: Path):
        path = tmp_path / "llm.yaml"
        path.write_text("llm:\n  - not\n  - a\n  - mapping\n")
        with pytest.raises(TypeError):
            load_llm_provider_config(path)
