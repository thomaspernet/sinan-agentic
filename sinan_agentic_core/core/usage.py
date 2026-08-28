"""The one shape this package reports token usage in.

Every path that runs a model hands a consumer the same record: the runner's
three execution branches, the rescue call the overflow fallback makes outside
the SDK, and the three chat entry points. A reader of ``usage`` should not have
to know which path produced it.

The provider's cached-token count is part of that shape. It is the only evidence
a caller has that a prompt prefix was served from the provider's cache rather
than billed again, so a path reporting a hardcoded zero was indistinguishable
from a genuinely cold cache — and a number that cannot tell those apart is not
usable as evidence either way. Every path reads the count the provider returned.
"""

from __future__ import annotations

from typing import Any

from agents import Usage
from openai.types.completion_usage import CompletionUsage


def usage_record(
    *,
    requests: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
) -> dict[str, Any]:
    """Assemble the reported record from the counts one or more responses returned.

    Args:
        requests: Model calls these counts cover.
        input_tokens: Prompt tokens billed, cached ones included.
        output_tokens: Completion tokens billed.
        total_tokens: Input plus output.
        cached_tokens: Prompt tokens the provider served from its cache.
        reasoning_tokens: Output tokens the model spent thinking.

    Returns:
        A fresh dict the caller owns, nested detail mappings included.
    """
    return {
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def sdk_usage_record(usage: Usage) -> dict[str, Any]:
    """The reported record for a ``Usage`` the SDK accumulated."""
    return usage_record(
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
    )


def completion_usage_record(usage: CompletionUsage) -> dict[str, Any]:
    """The reported record for a chat-completions response.

    A branch that bypasses the SDK gets the provider's own ``CompletionUsage``
    rather than a ``Usage`` the runner accumulated. Both detail blocks are
    optional on that type and every count inside them is nullable, so a provider
    that omits one reports zero instead of failing the call it came from.
    """
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return usage_record(
        requests=1,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=(prompt_details.cached_tokens if prompt_details else None) or 0,
        reasoning_tokens=(completion_details.reasoning_tokens if completion_details else None) or 0,
    )


def aggregate_usage(result: Any) -> dict[str, Any]:
    """Sum every model response of a run into one record.

    Reads ``raw_responses``, which both ``RunResult`` and ``RunResultStreaming``
    carry once the run is over, so a streamed run reports the same accumulated
    details — the provider's cached-token count included — as a non-streamed
    one. ``Usage.add`` does the summing, so a detail the SDK learns to carry
    reaches every path without a second edit here.

    Args:
        result: A ``RunResult`` or ``RunResultStreaming``.

    Returns:
        The usage record, all zeros for a run that recorded no responses.
    """
    usage = Usage()
    for response in _responses(result):
        response_usage = getattr(response, "usage", None)
        if response_usage:
            usage.add(response_usage)
    return sdk_usage_record(usage)


def last_input_tokens(result: Any) -> int:
    """Input tokens of the run's final model call.

    Summed ``input_tokens`` counts the replayed history once per call, so it says
    nothing about how full the context window got. The last call's own input is
    the run's high-water mark.
    """
    for response in reversed(_responses(result)):
        response_usage = getattr(response, "usage", None)
        if response_usage:
            return int(response_usage.input_tokens or 0)
    return 0


def _responses(result: Any) -> list[Any]:
    """The run's model responses — empty only when the run made no model call."""
    responses: list[Any] = result.raw_responses
    return responses
