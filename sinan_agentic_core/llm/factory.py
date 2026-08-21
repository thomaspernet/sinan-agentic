"""Build an OpenAI client from a provider config and register it as the SDK default.

:func:`configure_llm_provider` is the single entry point that consumers call once
at process start; :func:`resolve_openai_client` reads that same client back. Both
hide the SDK's client accessors (``set_default_openai_client``,
``set_default_openai_api``, ``set_tracing_disabled``, and the default-client
getter) so callers never reach into ``agents.*`` internals.
"""

from __future__ import annotations

import logging

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

# NOTE: openai-agents==0.20.0 exports `set_default_openai_client` from the
# `agents` package root but no matching getter, so reading the configured
# client back requires the private module. This is the one place in the
# codebase that imports it — every caller goes through
# `resolve_openai_client()`. Re-check for a public getter on the next SDK bump.
from agents.models._openai_shared import get_default_openai_client
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .config import AzureOpenAIProviderConfig, OpenAIProviderConfig

logger = logging.getLogger(__name__)


def configure_llm_provider(
    config: OpenAIProviderConfig | AzureOpenAIProviderConfig,
) -> AsyncOpenAI:
    """Build the right ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` client and wire it
    into the OpenAI Agents SDK.

    Side effects:
      - Calls :func:`agents.set_default_openai_client` with ``use_for_tracing``
        from *config*.
      - Calls :func:`agents.set_default_openai_api` with ``config.api_mode``.
      - Calls :func:`agents.set_tracing_disabled` when ``config.disable_tracing``
        is true.

    Returns:
        The constructed client. Most callers ignore it; return it so callers
        that need to make raw OpenAI calls (e.g. embeddings) can reuse it.
    """
    client = _build_client(config)
    set_default_openai_client(client, use_for_tracing=config.use_for_tracing)
    set_default_openai_api(config.api_mode)
    if config.disable_tracing:
        set_tracing_disabled(True)
    logger.info(
        "Configured LLM provider %r (api_mode=%s, tracing=%s)",
        config.provider,
        config.api_mode,
        "off" if config.disable_tracing else "on",
    )
    return client


def resolve_openai_client() -> AsyncOpenAI:
    """Return the OpenAI client to use for a direct, non-SDK model call.

    Any code path that talks to the provider outside ``Runner.run(...)`` --
    a recovery branch, an embeddings call, a provider-specific escape hatch --
    resolves its client here rather than constructing one. Provider wiring
    (Azure endpoint, deployment, base URL, credentials) lives only on the
    client :func:`configure_llm_provider` installed, so rebuilding one drops it.

    Returns:
        The client installed by :func:`configure_llm_provider`, or a fresh
        :class:`~openai.AsyncOpenAI` reading ``OPENAI_API_KEY`` from the
        environment when no provider has been configured.
    """
    client = get_default_openai_client()
    if client is None:
        return AsyncOpenAI()
    return client


def _build_client(
    config: OpenAIProviderConfig | AzureOpenAIProviderConfig,
) -> AsyncOpenAI:
    if isinstance(config, AzureOpenAIProviderConfig):
        return AsyncAzureOpenAI(
            api_key=config.api_key.get_secret_value(),
            azure_endpoint=config.azure_endpoint,
            api_version=config.api_version,
            azure_deployment=config.azure_deployment,
        )
    return AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=config.base_url,
        organization=config.organization,
        project=config.project,
    )
