"""Typed classification of agent-run failures.

An agent loop fails in a handful of distinct ways and the framework owes each a
different reaction: an exhausted loop is worth rescuing with a condensed second
call, a refusal is a deliberate model decision that must surface untouched, and
everything else propagates. Telling them apart used to mean searching
``str(err)`` for ``"Max turns"`` and ``"context_length_exceeded"`` — a contract
owned by whoever last worded the upstream message, and one that also fired on an
unrelated error whose text happened to quote those words.

The SDK raises a distinct exception class per failure (``MaxTurnsExceeded``,
``ModelRefusalError``, ``ModelBehaviorError``), so those are decided by type.
Context overflow has no exception class in the SDK: it reaches the runner as the
provider's HTTP 400, whose machine-readable ``code`` field carries
``context_length_exceeded``. This module reads that field off the typed
``openai.APIStatusError`` rather than out of the rendered message.

:func:`classify_run_error` is the single place that decides which kind an
exception is; every branch that reacts to a failed run keys off the result.
:func:`run_error_payload` carries that decision out to callers who receive a
result dict rather than the exception itself.
"""

from __future__ import annotations

from enum import Enum

from agents import MaxTurnsExceeded, ModelBehaviorError, ModelRefusalError
from openai import APIStatusError

# The provider's machine-readable code for "this request exceeds the model's
# context window". OpenAI and Azure OpenAI both return it in the error body of a
# 400; it is part of the provider's error contract, not prose.
CONTEXT_OVERFLOW_ERROR_CODE = "context_length_exceeded"


class RunErrorKind(str, Enum):
    """Why an agent run failed, decided by exception type rather than message.

    - ``MAX_TURNS`` -> the SDK's ``MaxTurnsExceeded``: the loop hit its turn
      ceiling with work still outstanding.
    - ``CONTEXT_OVERFLOW`` -> a provider ``APIStatusError`` carrying
      ``context_length_exceeded``: the assembled input no longer fits the model.
    - ``MODEL_REFUSAL`` -> the SDK's ``ModelRefusalError``: the model declined to
      produce the requested output.
    - ``MODEL_BEHAVIOR`` -> the SDK's ``ModelBehaviorError``: the model did
      something the run cannot use — malformed structured output, a call to a
      tool that does not exist.
    - ``UNKNOWN`` -> anything else, including the framework's own ``ValueError``
      from a tool. Callers treat it as unrecoverable.
    """

    MAX_TURNS = "max_turns"
    CONTEXT_OVERFLOW = "context_overflow"
    MODEL_REFUSAL = "model_refusal"
    MODEL_BEHAVIOR = "model_behavior"
    UNKNOWN = "unknown"


# The two failures a condensed second call can rescue: the loop ran out of room,
# not out of willingness. A refusal is deliberately absent — re-asking the same
# model through a code path that bypasses the run would be routing around its
# answer, not recovering from a limit. A malformed output is absent too; that one
# is already handled in-run by the SDK's ``invalid_final_output`` handler.
FALLBACK_RECOVERABLE_KINDS = frozenset({RunErrorKind.MAX_TURNS, RunErrorKind.CONTEXT_OVERFLOW})


def classify_run_error(error: BaseException) -> RunErrorKind:
    """Classify an exception raised out of an agent run.

    Args:
        error: The exception the run raised.

    Returns:
        The matching :class:`RunErrorKind`, or ``RunErrorKind.UNKNOWN`` when the
        exception is not one the framework reacts to specifically.
    """
    if isinstance(error, MaxTurnsExceeded):
        return RunErrorKind.MAX_TURNS
    if isinstance(error, ModelRefusalError):
        return RunErrorKind.MODEL_REFUSAL
    if isinstance(error, ModelBehaviorError):
        return RunErrorKind.MODEL_BEHAVIOR
    if isinstance(error, APIStatusError) and error.code == CONTEXT_OVERFLOW_ERROR_CODE:
        return RunErrorKind.CONTEXT_OVERFLOW
    return RunErrorKind.UNKNOWN


def run_error_payload(error: BaseException) -> dict[str, str]:
    """Describe a failed run for a caller that gets a result dict, not an exception.

    An orchestration or chat call catches the failure and returns it, so the
    caller never sees the exception class. The rendered message alone leaves
    them string-matching the very text :func:`classify_run_error` exists to stop
    branching on, so the classified kind travels alongside it.

    Args:
        error: The exception the run raised.

    Returns:
        ``{"error": <rendered message>, "error_kind": <RunErrorKind value>}``.
        The kind is the enum's plain string value, so the payload stays
        JSON-serializable across an API boundary.
    """
    return {"error": str(error), "error_kind": classify_run_error(error).value}
