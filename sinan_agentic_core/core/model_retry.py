"""Declarative opt-in retry policy for model calls.

The SDK only retries a model request when ``ModelSettings.retry`` carries both an
attempt budget *and* a policy callback — with either missing, the runner never
schedules a retry. Writing that callback means importing ``agents.retry`` and
wiring a ``ModelRetrySettings`` by hand at every agent.

This module carries the same choice as data: an attempt count, the error classes
to retry on, and an optional backoff schedule. ``build_model_retry_settings``
translates it into the SDK object, and ``apply_model_retry`` overlays that object
onto the ``ModelSettings`` each agent-building path assembles — so retries stay
declarative in ``agents.yaml`` and consumers never reach into ``agents.*`` to get
them. The runner's overflow-fallback branch bypasses the SDK runner and so cannot
use the overlay, but it reads the same translated settings for the one thing it
can still honor: the attempt count.

Retry is off unless declared: retrying costs latency and duplicate billed
requests, so it is opt-in per agent.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from agents import ModelRetryBackoffSettings, ModelRetrySettings, ModelSettings
from agents.retry import RetryPolicy, retry_policies
from pydantic import BaseModel, Field

# Attempts after the initial request. Two covers the transient blips a single
# retry misses without turning a hard outage into a long stall.
DEFAULT_MAX_RETRIES = 2


class RetryTrigger(str, Enum):
    """The error class a model call is retried on.

    - ``PROVIDER_SUGGESTED`` -> ``retry_policies.provider_suggested()``: follows
      the model adapter's own advice. For OpenAI that covers 408/409/429, every
      5xx, connection errors and timeouts, and honors an ``x-should-retry``
      header or a ``Retry-After`` delay when the provider sends one.
    - ``NETWORK_ERROR`` -> ``retry_policies.network_error()``: connection errors
      and timeouts only, regardless of what the provider advises.
    - ``RETRY_AFTER`` -> ``retry_policies.retry_after()``: retries only when the
      error carries an explicit ``Retry-After`` delay, and waits exactly that long.
    """

    PROVIDER_SUGGESTED = "provider_suggested"
    NETWORK_ERROR = "network_error"
    RETRY_AFTER = "retry_after"


_POLICY_BUILDERS: dict[RetryTrigger, Callable[[], RetryPolicy]] = {
    RetryTrigger.PROVIDER_SUGGESTED: retry_policies.provider_suggested,
    RetryTrigger.NETWORK_ERROR: retry_policies.network_error,
    RetryTrigger.RETRY_AFTER: retry_policies.retry_after,
}

# Provider advice already covers rate limits and 5xx; the network policy adds the
# transport failures an adapter without advice would otherwise let through.
DEFAULT_RETRY_TRIGGERS = (RetryTrigger.PROVIDER_SUGGESTED, RetryTrigger.NETWORK_ERROR)


class RetryBackoffConfig(BaseModel):
    """Delay schedule applied between attempts when no explicit delay is known.

    Every field is optional; the SDK fills an unset one with its own default. A
    provider-supplied ``Retry-After`` always wins over this schedule.
    """

    initial_delay: float | None = Field(default=None, ge=0)
    max_delay: float | None = Field(default=None, ge=0)
    multiplier: float | None = Field(default=None, ge=0)
    jitter: bool | None = None


class ModelRetryConfig(BaseModel):
    """Opt-in retry policy for one agent's model calls.

    Declared in ``agents.yaml``::

        agents:
          researcher:
            model: gpt-4o-mini
            description: Reads papers
            model_retry:
              max_retries: 3
              retry_on: [provider_suggested, network_error]
              backoff:
                initial_delay: 0.5
                max_delay: 8.0
    """

    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=1)
    retry_on: list[RetryTrigger] = Field(
        default_factory=lambda: list(DEFAULT_RETRY_TRIGGERS), min_length=1
    )
    backoff: RetryBackoffConfig | None = None

    def build(self) -> ModelRetrySettings:
        """Translate this config into the SDK's ``ModelRetrySettings``.

        Triggers are combined with ``retry_policies.any()``: the call is retried
        when any listed trigger matches, and the first delay one of them supplies
        wins over the backoff schedule.
        """
        policy = retry_policies.any(*(_POLICY_BUILDERS[trigger]() for trigger in self.retry_on))
        backoff = (
            ModelRetryBackoffSettings(**self.backoff.model_dump())
            if self.backoff is not None
            else None
        )
        return ModelRetrySettings(max_retries=self.max_retries, backoff=backoff, policy=policy)


def build_model_retry_settings(retry: ModelRetryConfig | None) -> ModelRetrySettings | None:
    """Translate a declared retry policy into the SDK settings a run honors.

    Every path that needs the translated policy calls this — :func:`apply_model_retry`,
    which overlays it onto the settings an agent carries, and
    ``BaseAgentRunner._execute_with_fallback()``, which bypasses the SDK runner and
    can only honor the attempt count — so the "off unless declared" rule lives here
    instead of once per caller.

    Args:
        retry: The agent's declared policy, or None when it opts out.

    Returns:
        The configured settings, or None when nothing is declared. Each call builds
        a fresh object, so no two agents share one.
    """
    if retry is None:
        return None

    return retry.build()


def apply_model_retry(
    retry: ModelRetryConfig | None,
    model_settings: ModelSettings | None = None,
) -> ModelSettings | None:
    """Overlay a declared retry policy onto the model settings an agent will carry.

    Both agent-building paths call this where they assemble ``ModelSettings`` —
    ``BaseAgentRunner.create_agent()`` and ``create_agent_from_registry()`` — so a
    declared policy reaches the SDK the same way whichever path built the agent.

    Args:
        retry: The agent's declared policy, or None when it opts out.
        model_settings: Settings already computed for the agent, if any.

    Returns:
        Settings carrying the policy, *model_settings* untouched when nothing is
        declared, or None when there is neither. Callers omit the
        ``model_settings=`` kwarg on None so the SDK default applies.
    """
    settings = build_model_retry_settings(retry)
    if settings is None:
        return model_settings

    # resolve() overlays the settings in hand on top of the declared retry
    # policy, so a caller that sets its own retry still wins field-by-field
    # while every other declared agent keeps the policy.
    return ModelSettings(retry=settings).resolve(model_settings)
