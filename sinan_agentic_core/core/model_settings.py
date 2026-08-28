"""Overlays onto the ``ModelSettings`` an agent carries into its run.

Two things an agent *declares* land on the SDK's ``ModelSettings`` rather than on
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

A prompt cache key gets its own overlay because it is not declared: a caller
names the cache shard per run, and the same agent may legitimately run under
different keys. It lands on the same object because ``extra_args`` is what the
SDK forwards to the provider.
"""

from __future__ import annotations

from dataclasses import replace

from agents import ModelSettings

from .model_retry import ModelRetryConfig, build_model_retry_settings

# The provider parameter that pins which prompt-cache shard a request routes to.
# Named here rather than imported: the SDK keeps its own copy inside
# ``agents.run_internal``, which is not public API, and the supported floor
# (``openai-agents`` 0.21.1) has no prompt-cache resolver at all.
PROMPT_CACHE_KEY_FIELD = "prompt_cache_key"


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


def apply_prompt_cache_key(
    model_settings: ModelSettings | None,
    prompt_cache_key: str | None,
) -> ModelSettings | None:
    """Overlay a caller's prompt cache key onto the settings a run will use.

    OpenAI prompt caching is prefix-based, and the provider shards its cache by
    key: sibling calls that share a leading span reuse it only when they route to
    the same shard. The SDK generates a key of its own from the conversation,
    session, or group id, but only for an official OpenAI client — an Azure
    deployment gets none — so a caller that wants sibling calls to share a shard
    names the key itself.

    A key already present in ``extra_args`` or ``extra_body`` is left untouched
    and *prompt_cache_key* is dropped: those are the two places the SDK reads a
    caller-supplied key from, and writing a second one would send two conflicting
    values for the same provider parameter.

    Args:
        model_settings: Settings already computed for the agent, if any.
        prompt_cache_key: The cache shard this run should route to, or None when
            the caller names none.

    Returns:
        Settings carrying the key, or *model_settings* untouched when there is no
        key to add or one is already set.
    """
    if prompt_cache_key is None:
        return model_settings

    settings = model_settings if model_settings is not None else ModelSettings()
    if _has_prompt_cache_key(settings.extra_args) or _has_prompt_cache_key(settings.extra_body):
        return model_settings

    extra_args = dict(settings.extra_args or {})
    extra_args[PROMPT_CACHE_KEY_FIELD] = prompt_cache_key
    return replace(settings, extra_args=extra_args)


def _has_prompt_cache_key(mapping: object) -> bool:
    """Whether *mapping* is a mapping that already names a prompt cache key."""
    return isinstance(mapping, dict) and PROMPT_CACHE_KEY_FIELD in mapping
