"""Typed classification of agent-run failures.

An agent loop fails in a handful of distinct ways and the framework owes each a
different reaction: an exhausted loop is worth rescuing with a condensed second
call, a refusal is a deliberate model decision that must surface untouched, and
everything else propagates. Telling them apart used to mean searching
``str(err)`` for ``"Max turns"`` and ``"context_length_exceeded"`` — a contract
owned by whoever last worded the upstream message, and one that also fired on an
unrelated error whose text happened to quote those words.

The SDK raises a distinct exception class per failure (``MaxTurnsExceeded``,
``ModelRefusalError``, ``ModelBehaviorError``, ``ModelTimeoutError``, the two
guardrail tripwires), so those are decided by type. Context overflow has no
exception class in the SDK: it reaches the runner as the provider's HTTP 400,
whose machine-readable ``code`` field carries ``context_length_exceeded``. This
module reads that field off the typed ``openai.APIStatusError`` rather than out
of the rendered message.

:func:`classify_run_error` is the single place that decides which kind an
exception is; every branch that reacts to a failed run keys off the result.
:func:`run_error_payload` carries that decision out to callers who receive a
result dict rather than the exception itself.

A guardrail tripwire needs more than its kind. The SDK renders it as
``Guardrail {guardrail.__class__.__name__} triggered tripwire`` — the *class*
name, identical for every input guardrail an agent declares — so the message
alone never says which check rejected the run. The payload therefore carries the
guardrail's registered name and the set of results that completed before the run
stopped, both read off the typed exception. Reporting the full set is uniform
across ``chat()``, ``chat_with_hooks()``, and ``chat_streamed()`` only from
openai-agents 0.19.2 on: before it, the non-streamed entry points discarded the
accumulator when the tripwire raised and reported an empty list where the
streamed one reported every result.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from agents import (
    InputGuardrailResult,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelTimeoutError,
    OutputGuardrailResult,
    OutputGuardrailTripwireTriggered,
)
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
      tool that does not exist, or a Responses turn that ended in a terminal
      ``failed`` / ``incomplete`` state (an output-token cap, a content filter,
      a provider-side error). That last one is the provider reporting an
      outcome rather than the model misbehaving, but openai-agents raises the
      same exception class for it — on the streamed path since 0.21.1, on the
      non-streamed one since 0.22.0 — and only the message distinguishes them,
      so they share a kind.
    - ``MODEL_TIMEOUT`` -> the SDK's ``ModelTimeoutError``: one model-call
      attempt outran the agent's declared ``model_timeout``. The bound is the
      caller's own policy, so this is a limit they chose, not a failure of the
      run — and telling it apart from ``UNKNOWN`` is what lets them raise the
      bound rather than hunt a bug.
    - ``INPUT_GUARDRAIL_TRIPWIRE`` -> the SDK's
      ``InputGuardrailTripwireTriggered``: a declared input guardrail rejected
      the request before the agent ran.
    - ``OUTPUT_GUARDRAIL_TRIPWIRE`` -> the SDK's
      ``OutputGuardrailTripwireTriggered``: the agent produced an answer a
      declared output guardrail blocked.
    - ``UNKNOWN`` -> anything else, including the framework's own ``ValueError``
      from a tool. Callers treat it as unrecoverable.

    The two tripwire kinds are separate members rather than one because the
    caller's reaction differs: an input tripwire means nothing ran and the
    request needs restating, an output tripwire means the run completed and its
    answer was withheld.
    """

    MAX_TURNS = "max_turns"
    CONTEXT_OVERFLOW = "context_overflow"
    MODEL_REFUSAL = "model_refusal"
    MODEL_BEHAVIOR = "model_behavior"
    MODEL_TIMEOUT = "model_timeout"
    INPUT_GUARDRAIL_TRIPWIRE = "input_guardrail_tripwire"
    OUTPUT_GUARDRAIL_TRIPWIRE = "output_guardrail_tripwire"
    UNKNOWN = "unknown"


# The two failures a condensed second call can rescue: the loop ran out of room,
# not out of willingness. A refusal is deliberately absent — re-asking the same
# model through a code path that bypasses the run would be routing around its
# answer, not recovering from a limit. The guardrail tripwires are absent for the
# same reason and more sharply: a tripwire is the project's own declared check
# saying no, and a retry that reaches a different outcome is the guardrail being
# defeated, not a limit being recovered from. A timeout is absent too: the rescue
# is another model call, bounded by the same declared number of seconds, so a
# provider slow enough to trip the bound trips it again.
#
# ``MODEL_BEHAVIOR`` is absent, and that takes two reasons because the kind
# covers two routes to the same exception class. A malformed final output is
# handled in-run by the SDK's ``invalid_final_output`` handler, so reaching here
# means salvage was already tried and declined. A terminal ``failed`` /
# ``incomplete`` Responses payload never meets that handler at all: openai-agents
# raises it from inside ``get_response``, before any final-output validation.
#
# The decision on that second route is to leave it out. An output-cap
# ``incomplete`` genuinely reads as "ran out of room" — the family the condensed
# call does rescue — but nothing here can act on only that subset: the two routes
# share one kind, and the sole thing separating them at the call site is the
# exception message (``status=``, ``incomplete_details=``), the contract this
# module exists to stop branching on. Admitting the kind would admit malformed
# output and unknown-tool calls with it. The rescue would also have nothing to
# work with: it condenses the tool output the failed run gathered, and the
# terminal check fires on the model call itself, so a run that dies there
# gathered none. Splitting the subset off needs its own ``RunErrorKind``, which
# needs the SDK to type the failure — not a message match added here.
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
    if isinstance(error, ModelTimeoutError):
        return RunErrorKind.MODEL_TIMEOUT
    if isinstance(error, InputGuardrailTripwireTriggered):
        return RunErrorKind.INPUT_GUARDRAIL_TRIPWIRE
    if isinstance(error, OutputGuardrailTripwireTriggered):
        return RunErrorKind.OUTPUT_GUARDRAIL_TRIPWIRE
    if isinstance(error, APIStatusError) and error.code == CONTEXT_OVERFLOW_ERROR_CODE:
        return RunErrorKind.CONTEXT_OVERFLOW
    return RunErrorKind.UNKNOWN


def _guardrail_tripwire_details(error: BaseException) -> dict[str, Any] | None:
    """Describe which guardrail stopped the run, and what else had finished.

    Args:
        error: The exception the run raised.

    Returns:
        ``{"name": <registered name of the tripping guardrail>, "results":
        [{"name": ..., "tripwire_triggered": ...}, ...]}`` when *error* is a
        guardrail tripwire, otherwise ``None``. ``results`` holds every guardrail
        that finished before the run stopped, the tripping one included, and is
        empty when the SDK attached no run data to the exception.
    """
    results: list[InputGuardrailResult] | list[OutputGuardrailResult]
    if isinstance(error, InputGuardrailTripwireTriggered):
        tripped: InputGuardrailResult | OutputGuardrailResult = error.guardrail_result
        results = error.run_data.input_guardrail_results if error.run_data else []
    elif isinstance(error, OutputGuardrailTripwireTriggered):
        tripped = error.guardrail_result
        results = error.run_data.output_guardrail_results if error.run_data else []
    else:
        return None

    return {
        "name": tripped.guardrail.get_name(),
        "results": [
            {
                "name": result.guardrail.get_name(),
                "tripwire_triggered": result.output.tripwire_triggered,
            }
            for result in results
        ],
    }


def run_error_payload(error: BaseException) -> dict[str, Any]:
    """Describe a failed run for a caller that gets a result dict, not an exception.

    An orchestration or chat call catches the failure and returns it, so the
    caller never sees the exception class. The rendered message alone leaves
    them string-matching the very text :func:`classify_run_error` exists to stop
    branching on, so the classified kind travels alongside it.

    Args:
        error: The exception the run raised.

    Returns:
        ``{"error": <rendered message>, "error_kind": <RunErrorKind value>}``,
        plus a ``"guardrail"`` entry naming the guardrail that stopped the
        run and listing every result that completed, when a tripwire is what
        stopped it. The kind is the enum's plain string value and every
        guardrail field is a string or a bool, so the payload stays
        JSON-serializable across an API boundary.
    """
    payload: dict[str, Any] = {
        "error": str(error),
        "error_kind": classify_run_error(error).value,
    }

    details = _guardrail_tripwire_details(error)
    if details is not None:
        payload["guardrail"] = details

    return payload
