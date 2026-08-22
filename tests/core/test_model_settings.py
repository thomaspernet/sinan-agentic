"""Tests for the declared model-settings overlay (core/model_settings.py)."""

from __future__ import annotations

import pytest
from agents import ModelRetrySettings, ModelSettings
from pydantic import ValidationError

from sinan_agentic_core.core.model_retry import ModelRetryConfig
from sinan_agentic_core.core.model_settings import apply_declared_model_settings


class TestNothingDeclared:
    """An agent that declares neither must come out exactly as it went in."""

    def test_no_declaration_and_no_settings_stays_none(self) -> None:
        """None lets the caller omit the kwarg so the SDK default applies."""
        assert apply_declared_model_settings() is None

    def test_no_declaration_returns_the_settings_untouched(self) -> None:
        settings = ModelSettings(temperature=0.2)

        assert apply_declared_model_settings(settings) is settings


class TestRetryOverlay:
    def test_declared_policy_lands_on_fresh_settings(self) -> None:
        settings = apply_declared_model_settings(model_retry=ModelRetryConfig(max_retries=4))

        assert settings is not None
        assert settings.retry is not None
        assert settings.retry.max_retries == 4
        assert settings.retry.policy is not None

    def test_declared_policy_merges_into_existing_settings(self) -> None:
        settings = apply_declared_model_settings(
            ModelSettings(temperature=0.2), model_retry=ModelRetryConfig(max_retries=4)
        )

        assert settings is not None
        assert settings.temperature == 0.2
        assert settings.retry is not None
        assert settings.retry.max_retries == 4

    def test_settings_in_hand_win_over_the_declared_policy(self) -> None:
        settings = apply_declared_model_settings(
            ModelSettings(retry=ModelRetrySettings(max_retries=9)),
            model_retry=ModelRetryConfig(max_retries=4),
        )

        assert settings is not None
        assert settings.retry is not None
        assert settings.retry.max_retries == 9


class TestTimeoutOverlay:
    def test_declared_timeout_lands_on_fresh_settings(self) -> None:
        settings = apply_declared_model_settings(model_timeout=30.0)

        assert settings is not None
        assert settings.timeout == 30.0

    def test_declared_timeout_merges_into_existing_settings(self) -> None:
        settings = apply_declared_model_settings(ModelSettings(temperature=0.2), model_timeout=30.0)

        assert settings is not None
        assert settings.temperature == 0.2
        assert settings.timeout == 30.0

    def test_settings_in_hand_win_over_the_declared_timeout(self) -> None:
        """A caller's own override replaces the declaration field by field."""
        settings = apply_declared_model_settings(ModelSettings(timeout=5.0), model_timeout=30.0)

        assert settings is not None
        assert settings.timeout == 5.0

    def test_a_timeout_needs_no_retry_policy(self) -> None:
        """The two are separate keys: bounding an attempt must not buy a second one."""
        settings = apply_declared_model_settings(model_timeout=30.0)

        assert settings is not None
        assert settings.retry is None

    @pytest.mark.parametrize("timeout", [0, -1.0, float("inf")])
    def test_a_non_positive_or_infinite_timeout_is_rejected(self, timeout: float) -> None:
        """The SDK's own constraint on the field, reached through the overlay."""
        with pytest.raises(ValidationError):
            apply_declared_model_settings(model_timeout=timeout)


class TestBothDeclared:
    def test_retry_and_timeout_land_on_the_same_settings(self) -> None:
        """One overlay, so no agent-building path can attach half a declaration."""
        settings = apply_declared_model_settings(
            model_retry=ModelRetryConfig(max_retries=4), model_timeout=30.0
        )

        assert settings is not None
        assert settings.retry is not None
        assert settings.retry.max_retries == 4
        assert settings.timeout == 30.0
