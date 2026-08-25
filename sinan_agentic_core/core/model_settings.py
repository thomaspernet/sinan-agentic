"""Overlay of an agent's declared model settings onto the settings it carries.

Two things an agent declares land on the SDK's ``ModelSettings`` rather than on
a run argument: ``model_retry:``, an opt-in retry policy, and ``model_timeout:``,
a bound in seconds on each model-call attempt. Riding on the agent's settings is
what makes them reach every execution path — ``execute()`` in all three modes,
``run_agent()``, handoffs, and ``as_tool()`` sub-agents — without a per-run kwarg.

They are separate declarations because they answer separate questions. A timeout
bounds how long one attempt may hang; a retry policy decides whether to pay for
another one. Declaring retries without a timeout multiplies an unbounded wait
instead of capping it, and a consumer may well want the bound without the extra
billed requests — so neither key implies the other.

They share one overlay because they land on the same object. A second
``apply_*`` function would mean every agent-building path had to remember to
call both, and a path that forgot one would still build a working agent.
"""

from __future__ import annotations

from agents import ModelSettings

from .model_retry import ModelRetryConfig, build_model_retry_settings


def apply_declared_model_settings(
    model_settings: ModelSettings | None = None,
    *,
    model_retry: ModelRetryConfig | None = None,
    model_timeout: float | None = None,
) -> ModelSettings | None:
    """Overlay an agent's declared retry policy and timeout onto its settings.

    Both agent-building paths call this where they assemble ``ModelSettings`` —
    ``BaseAgentRunner.create_agent()`` and ``create_agent_from_registry()`` — so
    a declaration reaches the SDK the same way whichever path built the agent.

    Args:
        model_settings: Settings already computed for the agent, if any.
        model_retry: The agent's declared retry policy, or None when it opts out.
        model_timeout: Seconds each model-call attempt may take, or None when the
            agent declares no bound. Must be greater than zero; the SDK rejects
            anything else.

    Returns:
        Settings carrying every declaration, *model_settings* untouched when
        there is none, or None when there is neither. Callers omit the
        ``model_settings=`` kwarg on None so the SDK default applies.
    """
    if model_retry is None and model_timeout is None:
        return model_settings

    declared = ModelSettings(
        retry=build_model_retry_settings(model_retry),
        timeout=model_timeout,
    )
    # resolve() overlays the settings in hand on top of the declared ones, so a
    # caller that sets its own retry or timeout still wins field-by-field while
    # every other declared agent keeps what it declared.
    return declared.resolve(model_settings)
