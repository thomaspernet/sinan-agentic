"""Tests for the declarative model retry policy (core/model_retry.py)."""

from __future__ import annotations

import pytest
from agents import ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings
from agents.retry import ModelRetryAdvice, ModelRetryNormalizedError, RetryPolicyContext
from openai import APIConnectionError
from pydantic import ValidationError

from sinan_agentic_core.core.model_retry import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_TRIGGERS,
    ModelRetryConfig,
    RetryBackoffConfig,
    RetryTrigger,
    apply_model_retry,
)


def _policy_context(
    error: Exception,
    *,
    normalized: ModelRetryNormalizedError | None = None,
    advice: ModelRetryAdvice | None = None,
) -> RetryPolicyContext:
    """Build the context the SDK hands a retry policy for one failed attempt."""
    return RetryPolicyContext(
        error=error,
        attempt=1,
        max_retries=DEFAULT_MAX_RETRIES,
        stream=False,
        normalized=normalized or ModelRetryNormalizedError(),
        provider_advice=advice,
    )


class TestConfigDefaults:
    def test_defaults_cover_provider_advice_and_transport_failures(self) -> None:
        assert ModelRetryConfig().retry_on == list(DEFAULT_RETRY_TRIGGERS)
        assert ModelRetryConfig().max_retries == DEFAULT_MAX_RETRIES

    def test_no_backoff_declared_leaves_sdk_defaults(self) -> None:
        assert ModelRetryConfig().build().backoff is None


class TestConfigValidation:
    def test_rejects_empty_trigger_list(self) -> None:
        """An empty list would build a policy that never retries — refuse it."""
        with pytest.raises(ValidationError):
            ModelRetryConfig(retry_on=[])

    def test_rejects_zero_retries(self) -> None:
        with pytest.raises(ValidationError):
            ModelRetryConfig(max_retries=0)

    def test_rejects_unknown_trigger(self) -> None:
        with pytest.raises(ValidationError):
            ModelRetryConfig(retry_on=["sometimes"])

    def test_rejects_negative_delay(self) -> None:
        with pytest.raises(ValidationError):
            RetryBackoffConfig(initial_delay=-1.0)


class TestBuild:
    def test_produces_sdk_settings_with_a_policy(self) -> None:
        """Without a policy callback the SDK never schedules a retry."""
        settings = ModelRetryConfig(max_retries=3).build()

        assert isinstance(settings, ModelRetrySettings)
        assert settings.max_retries == 3
        assert settings.policy is not None

    def test_backoff_maps_onto_sdk_settings(self) -> None:
        settings = ModelRetryConfig(
            backoff=RetryBackoffConfig(
                initial_delay=0.5, max_delay=8.0, multiplier=3.0, jitter=False
            )
        ).build()

        assert settings.backoff == ModelRetryBackoffSettings(
            initial_delay=0.5, max_delay=8.0, multiplier=3.0, jitter=False
        )

    def test_partial_backoff_leaves_unset_fields_to_the_sdk(self) -> None:
        settings = ModelRetryConfig(backoff=RetryBackoffConfig(initial_delay=0.5)).build()

        assert settings.backoff.initial_delay == 0.5
        assert settings.backoff.max_delay is None
        assert settings.backoff.multiplier is None


class TestPolicyBehavior:
    """The built policy is what the SDK actually calls on a failed attempt."""

    async def test_network_error_trigger_retries_connection_failures(self) -> None:
        policy = ModelRetryConfig(retry_on=[RetryTrigger.NETWORK_ERROR]).build().policy
        context = _policy_context(
            APIConnectionError(request=None),
            normalized=ModelRetryNormalizedError(is_network_error=True),
        )

        assert (await policy(context)).retry is True

    async def test_network_error_trigger_ignores_a_rate_limit(self) -> None:
        policy = ModelRetryConfig(retry_on=[RetryTrigger.NETWORK_ERROR]).build().policy
        context = _policy_context(
            RuntimeError("429"), normalized=ModelRetryNormalizedError(status_code=429)
        )

        assert (await policy(context)).retry is False

    async def test_provider_suggested_trigger_follows_adapter_advice(self) -> None:
        policy = ModelRetryConfig(retry_on=[RetryTrigger.PROVIDER_SUGGESTED]).build().policy
        context = _policy_context(
            RuntimeError("429"),
            normalized=ModelRetryNormalizedError(status_code=429),
            advice=ModelRetryAdvice(suggested=True, retry_after=1.5),
        )

        decision = await policy(context)
        assert decision.retry is True
        assert decision.delay == 1.5

    async def test_retry_after_trigger_waits_the_advertised_delay(self) -> None:
        policy = ModelRetryConfig(retry_on=[RetryTrigger.RETRY_AFTER]).build().policy
        context = _policy_context(
            RuntimeError("429"),
            normalized=ModelRetryNormalizedError(status_code=429, retry_after=4.0),
        )

        decision = await policy(context)
        assert decision.retry is True
        assert decision.delay == 4.0

    async def test_triggers_combine_so_any_match_retries(self) -> None:
        """A network failure retries even though the provider gave no advice."""
        policy = (
            ModelRetryConfig(retry_on=[RetryTrigger.PROVIDER_SUGGESTED, RetryTrigger.NETWORK_ERROR])
            .build()
            .policy
        )
        context = _policy_context(
            APIConnectionError(request=None),
            normalized=ModelRetryNormalizedError(is_network_error=True),
        )

        decision = await policy(context)
        assert decision.retry is True

    async def test_no_trigger_matching_does_not_retry(self) -> None:
        policy = ModelRetryConfig().build().policy
        context = _policy_context(
            ValueError("bad request"), normalized=ModelRetryNormalizedError(status_code=400)
        )

        decision = await policy(context)
        assert decision.retry is False


class TestApplyModelRetry:
    """The single translation point both agent-building paths call."""

    def test_no_policy_and_no_settings_stays_none(self) -> None:
        """None lets the caller omit the kwarg so the SDK default applies."""
        assert apply_model_retry(None) is None

    def test_no_policy_returns_the_settings_untouched(self) -> None:
        settings = ModelSettings(temperature=0.2)

        assert apply_model_retry(None, settings) is settings

    def test_declared_policy_lands_on_fresh_settings(self) -> None:
        settings = apply_model_retry(ModelRetryConfig(max_retries=4))

        assert settings.retry.max_retries == 4
        assert settings.retry.policy is not None

    def test_declared_policy_merges_into_existing_settings(self) -> None:
        settings = apply_model_retry(
            ModelRetryConfig(max_retries=4), ModelSettings(temperature=0.2)
        )

        assert settings.temperature == 0.2
        assert settings.retry.max_retries == 4

    def test_settings_in_hand_win_over_the_declared_policy(self) -> None:
        settings = apply_model_retry(
            ModelRetryConfig(max_retries=4),
            ModelSettings(retry=ModelRetrySettings(max_retries=9)),
        )

        assert settings.retry.max_retries == 9
