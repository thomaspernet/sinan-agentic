"""Shared fixtures and helpers for sinan_agentic_core tests."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints
from unittest.mock import Mock, patch

# openai 3.x types ``APIStatusError.response`` as an ``httpx2.Response`` and
# ships httpx2 as its own dependency, so a provider error built here is only
# the shape the installed client raises when it is built from httpx2. The
# httpx 1.x line still reaches the environment, but only as a transitive of the
# MCP SDK -- a library this suite does not route provider errors through.
import httpx2
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
    set_tracing_disabled,
)
from agents.testing import ScriptedModel
from openai import APIStatusError, BadRequestError

from sinan_agentic_core.core.run_errors import CONTEXT_OVERFLOW_ERROR_CODE
from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.registry.guardrail_registry import (
    GuardrailCategory,
    GuardrailDefinition,
)
from sinan_agentic_core.session.agent_session import AgentSession, ConversationHistory

# The endpoint a provider error is raised against. Only its presence matters --
# ``APIStatusError`` reads ``response.request`` back -- so every provider error
# the suite builds names the same one.
PROVIDER_ENDPOINT = "https://api.openai.com/v1/chat/completions"

StatusErrorT = TypeVar("StatusErrorT", bound=APIStatusError)


def make_provider_status_error(
    error_class: type[StatusErrorT],
    message: str,
    status_code: int,
    body: object | None,
) -> StatusErrorT:
    """Build a typed provider error the way the installed openai client raises one.

    Every provider error in the suite goes through here so the transport the
    error carries is decided once, against the client actually installed,
    rather than at each construction site.
    """
    request = httpx2.Request("POST", PROVIDER_ENDPOINT)
    return error_class(
        message,
        response=httpx2.Response(status_code, request=request),
        body=body,
    )


def make_context_overflow_error(
    message: str = "This model's maximum context length is 8192 tokens.",
) -> BadRequestError:
    """Build the provider error a real context overflow raises.

    Context overflow has no SDK exception class -- it surfaces as the
    provider's HTTP 400 with ``context_length_exceeded`` in the error body's
    ``code`` field. Tests classify against that shape, not against a message.
    """
    return make_provider_status_error(
        BadRequestError,
        message,
        400,
        {
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


@pytest.fixture(scope="session", autouse=True)
def _tracing_disabled():
    """Keep the SDK's trace exporter out of the suite.

    A scripted run drives the real ``Runner``, which opens a trace. The default
    exporter ships it to OpenAI, so leaving tracing on would make the suite
    depend on a key and a network round trip for runs that otherwise reach no
    provider at all.
    """
    set_tracing_disabled(True)
    yield
    set_tracing_disabled(False)


@contextmanager
def scripted_run(runner: Any, *steps: Any) -> Iterator[ScriptedModel]:
    """Run *runner*'s agents against a scripted model instead of a provider.

    ``BaseAgentRunner.create_agent`` runs for real inside the block — tools,
    guardrails, handoffs, instructions and model settings are assembled the way
    a production run assembles them, and the SDK's own ``Runner`` consumes
    them. Only the resolved model is swapped, which is the one thing a test
    cannot let reach a provider; ``Agent.model`` is declared ``str | Model``, so
    a :class:`ScriptedModel` is a value the field already accepts.

    Each step is one model call, given as the normalized output items that call
    returns (see :func:`agents.testing.assistant_message` and
    :func:`agents.testing.function_call`).

    Yields the model, so a test can read back what the SDK actually sent it —
    the resolved ``model_settings``, the advertised tools, the assembled input.
    On a clean exit every configured step must have been consumed: a run that
    made fewer model calls than the script describes raises
    ``UnconsumedModelSteps`` rather than passing silently. Drive a run that
    stops early with ``pytest.raises`` *inside* the block, and script only the
    calls it reaches.
    """
    model = ScriptedModel(steps)
    real_create_agent = runner.create_agent

    async def _create_scripted_agent(*args: Any, **kwargs: Any) -> Any:
        agent = await real_create_agent(*args, **kwargs)
        agent.model = model
        return agent

    with patch.object(runner, "create_agent", _create_scripted_agent):
        yield model

    model.assert_complete()


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
    result.new_items = []
    return result
