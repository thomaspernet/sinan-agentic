"""Shared fixtures and helpers for sinan_agentic_core tests."""

from collections.abc import Sequence
from dataclasses import fields
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from unittest.mock import Mock

import httpx
import pytest
from agents import (
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    InputGuardrailTripwireTriggered,
    OutputGuardrail,
    OutputGuardrailResult,
    OutputGuardrailTripwireTriggered,
    RunErrorDetails,
    Usage,
    input_guardrail,
    output_guardrail,
)
from openai import BadRequestError

from sinan_agentic_core.core.run_errors import CONTEXT_OVERFLOW_ERROR_CODE
from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.registry.guardrail_registry import (
    GuardrailCategory,
    GuardrailDefinition,
)
from sinan_agentic_core.session.agent_session import AgentSession, ConversationHistory


def make_context_overflow_error(
    message: str = "This model's maximum context length is 8192 tokens.",
) -> BadRequestError:
    """Build the provider error a real context overflow raises.

    Context overflow has no SDK exception class -- it surfaces as the
    provider's HTTP 400 with ``context_length_exceeded`` in the error body's
    ``code`` field. Tests classify against that shape, not against a message.
    """
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return BadRequestError(
        message,
        response=httpx.Response(400, request=request),
        body={
            "message": message,
            "type": "invalid_request_error",
            "code": CONTEXT_OVERFLOW_ERROR_CODE,
        },
    )


def registered_input_guardrail(name: str) -> InputGuardrail[Any]:
    """An input guardrail whose only distinguishing identifier is its registration.

    Every guardrail built here wraps a function literally called ``_check`` and
    is an instance of the same ``InputGuardrail`` class, so a report that names
    it can have read that name off neither the function's ``__name__`` nor the
    class name the SDK renders into the tripwire message -- only off the
    registration.
    """

    @input_guardrail
    async def _check(ctx: Any, agent: Any, data: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    GuardrailDefinition(
        name=name,
        description="Registered for a test.",
        function=_check,
        category=GuardrailCategory.INPUT,
    )
    return _check


def registered_output_guardrail(name: str) -> OutputGuardrail[Any]:
    """The output-slot twin of :func:`registered_input_guardrail`."""

    @output_guardrail
    async def _check(ctx: Any, agent: Any, data: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    GuardrailDefinition(
        name=name,
        description="Registered for a test.",
        function=_check,
        category=GuardrailCategory.OUTPUT,
    )
    return _check


def _run_error_details(
    *,
    input_guardrail_results: list[InputGuardrailResult],
    output_guardrail_results: list[OutputGuardrailResult],
) -> RunErrorDetails:
    """The run data the SDK attaches to an ``AgentsException`` before re-raising."""
    return RunErrorDetails(
        input="Hi",
        new_items=[],
        raw_responses=[],
        last_agent=Mock(),
        context_wrapper=Mock(),
        input_guardrail_results=input_guardrail_results,
        output_guardrail_results=output_guardrail_results,
    )


def make_input_tripwire_error(
    tripped: InputGuardrail[Any],
    passed: Sequence[InputGuardrail[Any]] = (),
    *,
    with_run_data: bool = True,
) -> InputGuardrailTripwireTriggered:
    """Build the exception an input guardrail raises, as the SDK builds it.

    From openai-agents 0.19.2 the run data carries every guardrail that
    completed -- the passing ones and the tripping one -- on every entry point.
    Pass ``with_run_data=False`` for the case the SDK leaves it unset, which is
    what a redacted failure produces.
    """
    results = [
        InputGuardrailResult(
            guardrail=guardrail,
            output=GuardrailFunctionOutput(output_info=None, tripwire_triggered=False),
        )
        for guardrail in passed
    ]
    tripping = InputGuardrailResult(
        guardrail=tripped,
        output=GuardrailFunctionOutput(output_info=None, tripwire_triggered=True),
    )
    results.append(tripping)

    error = InputGuardrailTripwireTriggered(tripping)
    if with_run_data:
        error.run_data = _run_error_details(
            input_guardrail_results=results,
            output_guardrail_results=[],
        )
    return error


def make_output_tripwire_error(
    tripped: OutputGuardrail[Any],
    passed: Sequence[OutputGuardrail[Any]] = (),
) -> OutputGuardrailTripwireTriggered:
    """The output-slot twin of :func:`make_input_tripwire_error`."""

    def result(guardrail: OutputGuardrail[Any], *, triggered: bool) -> OutputGuardrailResult:
        return OutputGuardrailResult(
            guardrail=guardrail,
            agent_output="answer",
            agent=Mock(),
            output=GuardrailFunctionOutput(output_info=None, tripwire_triggered=triggered),
        )

    results = [result(guardrail, triggered=False) for guardrail in passed]
    tripping = result(tripped, triggered=True)
    results.append(tripping)

    error = OutputGuardrailTripwireTriggered(tripping)
    error.run_data = _run_error_details(
        input_guardrail_results=[],
        output_guardrail_results=results,
    )
    return error


def _declared_members(annotation: object) -> tuple[object, ...]:
    """The members of an optional annotation, or the annotation itself when it is not one."""
    if get_origin(annotation) in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def collection_field_names(cls: type) -> list[str]:
    """Field names declared as a ``dict[...]``/``list[...]``, optional or not.

    Selecting on the declared type excludes a non-collection field
    (``database_connector: Any``, ``model: str``) without naming it, and — unlike
    ``default_factory is list`` — still sees a collection field that defaults to
    ``None`` rather than to an empty container. A drift guard built on it
    therefore does not need editing when either kind of field is added.
    """
    hints = get_type_hints(cls)
    return [
        f.name
        for f in fields(cls)
        if any(get_origin(m) in (dict, list) for m in _declared_members(hints[f.name]))
    ]


def edit_every_level(seeded: dict[str, Any] | list[Any]) -> None:
    """Edit the container and the values nested in it — a shallow copy detaches only the outer."""
    for value in seeded.values() if isinstance(seeded, dict) else seeded:
        if isinstance(value, list):
            value.append("added_late")
        elif isinstance(value, dict):
            value["added_late"] = True

    if isinstance(seeded, dict):
        seeded["added_late"] = True
    else:
        seeded.append({"added_late": True})


@pytest.fixture
def session():
    """Create a fresh AgentSession."""
    return AgentSession(session_id="test-session")


@pytest.fixture
def context():
    """Create a fresh AgentContext with a mock connector."""
    return AgentContext(database_connector=Mock())


@pytest.fixture
def conversation_history():
    """Create a ConversationHistory with sample messages."""
    h = ConversationHistory()
    h.add_message("user", "Hello")
    h.add_message("assistant", "Hi there!")
    return h


@pytest.fixture
def sample_usage():
    """Create a real Usage object from the SDK."""
    return Usage(
        requests=1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )


@pytest.fixture
def mock_run_result(sample_usage):
    """Create a mock RunResult with raw_responses containing real Usage."""
    response = Mock()
    response.usage = sample_usage

    result = Mock()
    result.raw_responses = [response]
    result.final_output = "Test response"
    return result
