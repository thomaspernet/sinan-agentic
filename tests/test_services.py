"""Tests for services: events, hooks, usage helper, and chat (mocked Runner)."""

import asyncio
import copy
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelRefusalError,
    ToolGuardrailFunctionOutput,
    Usage,
    function_tool,
    tool_input_guardrail,
)
from pydantic import BaseModel

from sinan_agentic_core.core.output_recovery import recover_invalid_final_output
from sinan_agentic_core.core.run_errors import RunErrorKind
from sinan_agentic_core.services.chat import _usage_to_dict
from sinan_agentic_core.services.events import (
    AgentCompleteEvent,
    AgentStartEvent,
    AnswerEvent,
    BaseEvent,
    ErrorEvent,
    StreamingHelper,
    StreamingTextEvent,
    ThinkingEvent,
    ToolCallEvent,
)
from sinan_agentic_core.services.hooks import StreamingRunHooks
from sinan_agentic_core.session.agent_session import AgentSession
from tests.conftest import (
    collection_field_names,
    edit_every_level,
    make_context_overflow_error,
    make_input_tripwire_error,
    registered_input_guardrail,
)

# Every run failure a chat function classifies, and the kind it must report.
RUN_FAILURES = [
    (MaxTurnsExceeded("Max turns (10) exceeded"), RunErrorKind.MAX_TURNS),
    (ModelRefusalError("I can't help with that."), RunErrorKind.MODEL_REFUSAL),
    (make_context_overflow_error(), RunErrorKind.CONTEXT_OVERFLOW),
    (RuntimeError("Something else broke"), RunErrorKind.UNKNOWN),
]

# -- Event dataclasses ---------------------------------------------------------


class TestEvents:
    def test_base_event_to_dict(self):
        e = BaseEvent(event_type="custom")
        assert e.to_dict() == {"event_type": "custom"}

    def test_agent_start_event(self):
        e = AgentStartEvent(agent_name="analyzer", iteration=2)
        d = e.to_dict()
        assert d["event_type"] == "agent_start"
        assert d["agent_name"] == "analyzer"
        assert d["iteration"] == 2

    def test_agent_complete_event(self):
        e = AgentCompleteEvent(agent_name="analyzer")
        assert e.to_dict()["event_type"] == "agent_complete"

    def test_thinking_event(self):
        e = ThinkingEvent(message="Processing...", agent_name="bot")
        d = e.to_dict()
        assert d["message"] == "Processing..."
        assert d["agent_name"] == "bot"

    def test_tool_call_event(self):
        e = ToolCallEvent(tool_name="search", arguments={"q": "hello"})
        d = e.to_dict()
        assert d["tool_name"] == "search"
        assert d["arguments"] == {"q": "hello"}

    def test_streaming_text_event(self):
        e = StreamingTextEvent(text="chunk")
        assert e.to_dict()["text"] == "chunk"

    def test_answer_event(self):
        e = AnswerEvent(answer="42", sources=["a"], confidence=0.9)
        d = e.to_dict()
        assert d["answer"] == "42"
        assert d["confidence"] == 0.9

    def test_error_event(self):
        e = ErrorEvent(error="boom")
        assert e.to_dict()["error"] == "boom"


class TestAnswerEventOwnsItsSources:
    """The event is a fixed record, at both ends: nobody else holds its sources list."""

    def test_the_supplied_sources_are_readable(self):
        e = AnswerEvent(answer="42", sources=["data.csv"])

        assert e.to_dict()["sources"] == ["data.csv"]

    def test_a_source_added_by_the_caller_later_is_not_visible(self):
        declared = ["data.csv"]
        e = AnswerEvent(answer="42", sources=declared)

        declared.append("added_late.csv")

        assert e.sources == ["data.csv"]

    def test_a_consumer_editing_the_event_does_not_reach_the_callers_list(self):
        declared = ["data.csv"]
        e = AnswerEvent(answer="42", sources=declared)

        e.sources.append("added_by_consumer.csv")

        assert declared == ["data.csv"]

    def test_two_events_built_from_one_list_do_not_share_sources(self):
        declared = ["data.csv"]
        first = AnswerEvent(answer="42", sources=declared)
        second = AnswerEvent(answer="43", sources=declared)

        first.sources.append("added_late.csv")

        assert second.sources == ["data.csv"]

    def test_the_sources_themselves_stay_shared_with_the_caller(self):
        """Only the container is copied — a consumer matches sources by identity."""
        source = {"file": "data.csv"}

        e = AnswerEvent(answer="42", sources=[source])

        assert e.sources[0] is source

    def test_a_consumer_editing_the_payload_does_not_reach_the_event(self):
        e = AnswerEvent(answer="42", sources=["data.csv"])

        e.to_dict()["sources"].append("added_by_consumer.csv")

        assert e.sources == ["data.csv"]

    def test_two_payloads_from_one_event_do_not_share_sources(self):
        e = AnswerEvent(answer="42", sources=["data.csv"])

        first = e.to_dict()
        second = e.to_dict()
        first["sources"].append("added_by_consumer.csv")

        assert second["sources"] == ["data.csv"]

    def test_the_sources_in_the_payload_stay_shared_with_the_event(self):
        """Only the container is copied — a consumer matches sources by identity."""
        source = {"file": "data.csv"}
        e = AnswerEvent(answer="42", sources=[source])

        assert e.to_dict()["sources"][0] is e.sources[0]

    def test_an_emitted_event_is_detached_from_the_callers_list(self):
        events = []
        helper = StreamingHelper(event_callback=events.append)
        declared = ["data.csv"]

        helper.emit_answer("result", sources=declared)
        declared.append("added_late.csv")

        assert events[0].sources == ["data.csv"]

    def test_emitting_without_sources_yields_an_empty_list(self):
        events = []
        helper = StreamingHelper(event_callback=events.append)

        helper.emit_answer("result")

        assert events[0].sources == []

    def test_every_collection_field_is_detached_from_the_caller(self):
        """A collection field added later without a matching copy fails here, rather than drifting.

        The seeded sources are opaque scalars because only the container is copied
        here — an edit inside a source is meant to reach the event.
        """
        seeds: dict[str, Any] = {"sources": ["data.csv"]}
        assert set(collection_field_names(AnswerEvent)) == set(seeds), (
            "AnswerEvent gained or lost a collection field — seed it here "
            "and copy it in __post_init__"
        )
        seeded_as_supplied = copy.deepcopy(seeds)

        e = AnswerEvent(answer="42", **seeds)
        for seeded in seeds.values():
            edit_every_level(seeded)

        for name, as_supplied in seeded_as_supplied.items():
            assert getattr(e, name) == as_supplied, f"{name} is aliased to the caller's collection"


class TestToolCallEventOwnsItsArguments:
    """The event is a fixed record of one call, so nobody else holds its arguments."""

    def test_the_supplied_arguments_are_readable(self):
        e = ToolCallEvent(tool_name="search", arguments={"q": "hello", "tags": ["a"]})

        assert e.to_dict()["arguments"] == {"q": "hello", "tags": ["a"]}

    def test_an_argument_added_by_the_caller_later_is_not_visible(self):
        declared = {"q": "hello"}
        e = ToolCallEvent(tool_name="search", arguments=declared)

        declared["limit"] = 10

        assert e.arguments == {"q": "hello"}

    def test_an_edit_inside_a_nested_argument_by_the_caller_is_not_visible(self):
        declared = {"filters": {"tags": ["python"]}}
        e = ToolCallEvent(tool_name="search", arguments=declared)

        declared["filters"]["tags"].append("added_late")

        assert e.arguments == {"filters": {"tags": ["python"]}}

    def test_a_consumer_editing_the_event_does_not_reach_the_callers_dict(self):
        declared = {"filters": {"tags": ["python"]}}
        e = ToolCallEvent(tool_name="search", arguments=declared)

        e.arguments["filters"]["tags"].append("added_by_consumer")

        assert declared == {"filters": {"tags": ["python"]}}

    def test_two_events_built_from_one_dict_do_not_share_arguments(self):
        declared = {"filters": {"tags": ["python"]}}
        first = ToolCallEvent(tool_name="search", arguments=declared)
        second = ToolCallEvent(tool_name="search", arguments=declared)

        first.arguments["filters"]["tags"].append("added_late")

        assert second.arguments == {"filters": {"tags": ["python"]}}

    def test_a_consumer_editing_the_payload_does_not_reach_the_event(self):
        e = ToolCallEvent(tool_name="search", arguments={"q": "hello"})

        e.to_dict()["arguments"]["limit"] = 10

        assert e.arguments == {"q": "hello"}

    def test_a_consumer_editing_inside_the_payload_does_not_reach_the_event(self):
        e = ToolCallEvent(tool_name="search", arguments={"filters": {"tags": ["python"]}})

        e.to_dict()["arguments"]["filters"]["tags"].append("added_by_consumer")

        assert e.arguments == {"filters": {"tags": ["python"]}}

    def test_two_payloads_from_one_event_do_not_share_arguments(self):
        e = ToolCallEvent(tool_name="search", arguments={"filters": {"tags": ["python"]}})

        first = e.to_dict()
        second = e.to_dict()
        first["arguments"]["filters"]["tags"].append("added_by_consumer")

        assert second["arguments"] == {"filters": {"tags": ["python"]}}

    def test_every_collection_field_is_detached_from_the_caller(self):
        """A collection field added later without a matching copy fails here, rather than drifting."""
        seeds: dict[str, Any] = {"arguments": {"filters": {"tags": ["python"]}}}
        assert set(collection_field_names(ToolCallEvent)) == set(seeds), (
            "ToolCallEvent gained or lost a collection field — seed it here "
            "and copy it in __post_init__"
        )
        seeded_as_supplied = copy.deepcopy(seeds)

        e = ToolCallEvent(tool_name="search", **seeds)
        for seeded in seeds.values():
            edit_every_level(seeded)

        for name, as_supplied in seeded_as_supplied.items():
            assert getattr(e, name) == as_supplied, f"{name} is aliased to the caller's collection"


# -- StreamingHelper -----------------------------------------------------------


class TestStreamingHelper:
    def test_emit_agent_start(self):
        events = []
        helper = StreamingHelper(event_callback=events.append)
        helper.emit_agent_start("bot", iteration=3)
        assert len(events) == 1
        assert events[0].agent_name == "bot"

    def test_emit_answer(self):
        events = []
        helper = StreamingHelper(event_callback=events.append)
        helper.emit_answer("result", sources=["src"], confidence=0.8)
        assert events[0].answer == "result"
        assert events[0].confidence == 0.8

    def test_emit_error(self):
        events = []
        helper = StreamingHelper(event_callback=events.append)
        helper.emit_error("fail")
        assert events[0].error == "fail"

    def test_emit_agent_complete(self):
        events = []
        helper = StreamingHelper(event_callback=events.append)
        helper.emit_agent_complete("bot", iteration=2)
        assert len(events) == 1
        assert events[0].agent_name == "bot"
        assert events[0].iteration == 2

    def test_no_callback_does_not_raise(self):
        helper = StreamingHelper(event_callback=None)
        helper.emit_agent_start("bot")  # should not raise
        helper.emit_error("fail")


# -- StreamingRunHooks ---------------------------------------------------------


class TestStreamingRunHooks:
    async def test_on_tool_start(self):
        queue = asyncio.Queue()
        hooks = StreamingRunHooks(queue, {"my_tool": "My Tool"})
        mock_tool = Mock()
        mock_tool.name = "my_tool"

        await hooks.on_tool_start(None, None, mock_tool)

        assert "my_tool" in hooks.tools_called
        event = await queue.get()
        assert event["event"] == "tool_start"
        assert event["data"]["tool"] == "my_tool"
        assert event["data"]["friendly_name"] == "My Tool"

    async def test_on_tool_end(self):
        queue = asyncio.Queue()
        hooks = StreamingRunHooks(queue)
        mock_tool = Mock()
        mock_tool.name = "search"

        await hooks.on_tool_end(None, None, mock_tool, "result")

        event = await queue.get()
        assert event["event"] == "tool_end"
        assert event["data"]["tool"] == "search"

    async def test_on_agent_start(self):
        queue = asyncio.Queue()
        hooks = StreamingRunHooks(queue)

        await hooks.on_agent_start(None, Mock(name="bot"))

        event = await queue.get()
        assert event["event"] == "thinking"

    def test_friendly_name_fallback(self):
        hooks = StreamingRunHooks(asyncio.Queue())
        assert hooks._friendly_name("get_weather") == "get weather"

    def test_friendly_name_from_map(self):
        hooks = StreamingRunHooks(asyncio.Queue(), {"get_weather": "Weather Lookup"})
        assert hooks._friendly_name("get_weather") == "Weather Lookup"

    def test_the_callers_names_dict_is_not_aliased(self):
        """The hooks own their names, so a later edit to the caller's dict does not reach them."""
        declared = {"get_weather": "Weather Lookup"}

        hooks = StreamingRunHooks(asyncio.Queue(), declared)
        declared["search"] = "Late Name"

        assert "search" not in hooks.tool_friendly_names
        assert hooks._friendly_name("search") == "search"

    def test_two_hooks_built_from_one_dict_do_not_share_a_mapping(self):
        declared = {"get_weather": "Weather Lookup"}

        first = StreamingRunHooks(asyncio.Queue(), declared)
        second = StreamingRunHooks(asyncio.Queue(), declared)
        first.tool_friendly_names["search"] = "Search"

        assert "search" not in second.tool_friendly_names
        assert "search" not in declared


# -- _usage_to_dict ------------------------------------------------------------


class TestUsageToDict:
    def test_single_response(self, mock_run_result):
        usage = _usage_to_dict(mock_run_result)
        assert usage["requests"] == 1
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["input_tokens_details"]["cached_tokens"] == 0
        assert usage["output_tokens_details"]["reasoning_tokens"] == 0

    def test_multiple_responses(self):
        """Usage.add() aggregates across multiple responses."""
        from agents import Usage

        u1 = Usage(requests=1, input_tokens=100, output_tokens=40, total_tokens=140)
        u2 = Usage(requests=1, input_tokens=200, output_tokens=60, total_tokens=260)

        r1 = Mock()
        r1.usage = u1
        r2 = Mock()
        r2.usage = u2

        result = Mock()
        result.raw_responses = [r1, r2]

        usage = _usage_to_dict(result)
        assert usage["requests"] == 2
        assert usage["input_tokens"] == 300
        assert usage["output_tokens"] == 100
        assert usage["total_tokens"] == 400

    def test_empty_responses(self):
        result = Mock()
        result.raw_responses = []
        usage = _usage_to_dict(result)
        assert usage["requests"] == 0
        assert usage["total_tokens"] == 0


# -- chat() with mocked Runner ------------------------------------------------


class _Extraction(BaseModel):
    """Output type that gives an agent a schema to validate against."""

    answer: str


def _agent_double():
    """An agent stub carrying the ``tools`` and ``output_type`` defaults of a real ``Agent``."""
    return Mock(tools=[], output_type=None)


def _structured_agent():
    """A real agent whose declared output type needs schema validation."""
    return Agent(name="extractor", output_type=_Extraction)


def _guarded_agent():
    """A real agent whose one function tool carries a tool-input guardrail."""

    @function_tool
    def echo(value: str) -> str:
        """Echo a value.

        Args:
            value: Text to echo back.
        """
        return value

    @tool_input_guardrail
    def block_nothing(data):
        return ToolGuardrailFunctionOutput()

    guarded = copy.copy(echo)
    guarded.tool_input_guardrails = [block_nothing]
    return Agent(name="guarded", tools=[guarded])


def _trimming_agent(registry):
    """A real agent whose registered definition declares a trim policy.

    Trimming is a run-level setting with no slot on ``Agent``, so the chat
    service can only reach it through the definition behind the agent's name.
    """
    from sinan_agentic_core.core.tool_output_trim import ToolOutputTrimConfig
    from sinan_agentic_core.registry.agent_registry import AgentDefinition

    registry.register(
        AgentDefinition(
            name="_chat_trimming_agent",
            description="trims",
            instructions="You answer",
            tool_output_trim=ToolOutputTrimConfig(max_output_chars=4000),
        )
    )
    return Agent(name="_chat_trimming_agent")


@pytest.fixture
def trim_registry():
    """An isolated registry, patched inside the shared by-name definition resolver."""
    from sinan_agentic_core.registry.agent_registry import AgentRegistry

    registry = AgentRegistry()
    with patch(
        "sinan_agentic_core.registry.agent_registry.get_agent_registry", return_value=registry
    ):
        yield registry


def _strict_agent(registry):
    """A structured agent whose registered definition turns recovery off.

    The flag is a run-level setting with no slot on ``Agent``, so the chat
    service can only reach it through the definition behind the agent's name.
    """
    from sinan_agentic_core.registry.agent_registry import AgentDefinition

    registry.register(
        AgentDefinition(
            name="_chat_strict_agent",
            description="fails loudly",
            instructions="You extract",
            invalid_output_recovery=False,
        )
    )
    return Agent(name="_chat_strict_agent", output_type=_Extraction)


@pytest.fixture
def recovery_registry():
    """An isolated registry, patched inside the shared by-name definition resolver."""
    from sinan_agentic_core.registry.agent_registry import AgentRegistry

    registry = AgentRegistry()
    with patch(
        "sinan_agentic_core.registry.agent_registry.get_agent_registry", return_value=registry
    ):
        yield registry


def _run_result(response="ok"):
    """A run result carrying one response — enough for the usage aggregation."""
    raw = Mock()
    raw.usage = Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2)
    result = Mock()
    result.final_output = response
    result.new_items = []
    result.raw_responses = [raw]
    return result


def _streamed_result(response="ok"):
    """A streaming run result that yields no events before its final output."""
    result = _run_result(response)

    async def no_events():
        return
        yield  # make it an async generator

    result.stream_events = no_events
    return result


class TestChat:
    @staticmethod
    def _get_chat_module():
        import sys

        # Import to ensure the module is loaded, then get from sys.modules
        # to avoid the __init__.py shadowing the module with the function
        return sys.modules["sinan_agentic_core.services.chat"]

    async def test_chat_returns_usage(self):
        from agents import Usage

        chat_mod = self._get_chat_module()

        mock_usage = Usage(requests=1, input_tokens=50, output_tokens=25, total_tokens=75)
        mock_response = Mock()
        mock_response.usage = mock_usage

        mock_result = Mock()
        mock_result.final_output = "Hello!"
        mock_result.new_items = []
        mock_result.raw_responses = [mock_response]

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=mock_result)

                result = await chat_mod.chat("Hi", agent_name="test_agent", session=session)

        assert result["success"] is True
        assert result["response"] == "Hello!"
        assert result["usage"]["input_tokens"] == 50
        assert result["usage"]["output_tokens"] == 25
        assert result["usage"]["total_tokens"] == 75

    async def test_chat_error_handling(self):
        chat_mod = self._get_chat_module()

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.side_effect = ValueError("Agent not found")

            result = await chat_mod.chat("Hi", agent_name="missing", session=session)

        assert result["success"] is False
        assert "Agent not found" in result["error"]
        assert result["error_kind"] == RunErrorKind.UNKNOWN.value

    @pytest.mark.parametrize(("error", "expected"), RUN_FAILURES)
    async def test_run_failure_reports_its_kind(self, error, expected):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(side_effect=error)

                result = await chat_mod.chat("Hi", agent_name="a", session=session)

        assert result["success"] is False
        assert result["error"] == str(error)
        assert result["error_kind"] == expected.value

    async def test_chat_with_context(self):
        from agents import Usage

        chat_mod = self._get_chat_module()

        mock_usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response = Mock()
        mock_response.usage = mock_usage

        mock_result = Mock()
        mock_result.final_output = "ctx reply"
        mock_result.new_items = []
        mock_result.raw_responses = [mock_response]

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=mock_result)

                result = await chat_mod.chat("Hi", agent_name="a", session=session, context=Mock())

        assert result["success"] is True
        # Verify context was forwarded to Runner.run
        call_kwargs = mock_runner.run.call_args
        assert "context" in call_kwargs.kwargs

    async def test_tool_input_guardrails_enable_pre_approval(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _guarded_agent()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                await chat_mod.chat("Hi", agent_name="a", session=session)

        run_config = mock_runner.run.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_agent_without_guardrails_gets_no_run_config(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                await chat_mod.chat("Hi", agent_name="a", session=session)

        assert "run_config" not in mock_runner.run.call_args.kwargs

    async def test_prebuilt_agent_gets_pre_approval(self):
        """The setting is read off the agent, so it reaches the pre-built path too."""
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=_run_result())

            await chat_mod.chat("Hi", agent=_guarded_agent(), session=session)

        run_config = mock_runner.run.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_declared_trim_policy_reaches_the_run(self, trim_registry):
        """A resolved agent is trimmed here the way it is under the runner."""
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _trimming_agent(trim_registry)
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                await chat_mod.chat("Hi", agent_name="_chat_trimming_agent", session=session)

        run_config = mock_runner.run.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_structured_output_agent_gets_recovery_handler(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=_run_result())

            await chat_mod.chat("Hi", agent=_structured_agent(), session=session)

        handlers = mock_runner.run.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    async def test_plain_text_agent_gets_no_error_handlers(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                await chat_mod.chat("Hi", agent_name="a", session=session)

        assert "error_handlers" not in mock_runner.run.call_args.kwargs

    async def test_declared_recovery_opt_out_reaches_the_run(self, recovery_registry):
        """An agent that asked to fail loudly does so here the way it does under
        the runner."""
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=_run_result())

            await chat_mod.chat("Hi", agent=_strict_agent(recovery_registry), session=session)

        assert "error_handlers" not in mock_runner.run.call_args.kwargs


# -- chat_with_hooks() --------------------------------------------------------


class TestChatWithHooks:
    @staticmethod
    def _get_chat_module():
        import sys

        return sys.modules["sinan_agentic_core.services.chat"]

    async def test_yields_thinking_and_answer(self):
        from agents import Usage

        chat_mod = self._get_chat_module()

        mock_usage = Usage(requests=1, input_tokens=50, output_tokens=25, total_tokens=75)
        mock_response = Mock()
        mock_response.usage = mock_usage

        mock_result = Mock()
        mock_result.final_output = "Hooked answer"
        mock_result.new_items = []
        mock_result.raw_responses = [mock_response]

        session = AgentSession(session_id="test")

        async def run_with_hooks(**kwargs):
            hooks = kwargs.get("hooks")
            if hooks:
                mock_tool = Mock()
                mock_tool.name = "search"
                await hooks.on_tool_start(None, None, mock_tool)
                await hooks.on_tool_end(None, None, mock_tool, "ok")
            return mock_result

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(side_effect=run_with_hooks)

                events = []
                async for event in chat_mod.chat_with_hooks(
                    "Hi", agent_name="test_agent", session=session
                ):
                    events.append(event)

        event_types = [e["event"] for e in events]
        assert "thinking" in event_types
        assert "answer" in event_types
        assert "finalizing" in event_types

        answer = next(e for e in events if e["event"] == "answer")
        assert answer["data"]["response"] == "Hooked answer"
        assert "usage" in answer["data"]

    async def test_with_context(self):
        from agents import Usage

        chat_mod = self._get_chat_module()

        mock_usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response = Mock()
        mock_response.usage = mock_usage

        mock_result = Mock()
        mock_result.final_output = "ctx"
        mock_result.new_items = []
        mock_result.raw_responses = [mock_response]

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=mock_result)

                events = []
                async for event in chat_mod.chat_with_hooks(
                    "Hi", agent_name="a", session=session, context=Mock()
                ):
                    events.append(event)

        assert any(e["event"] == "answer" for e in events)

    async def test_tool_input_guardrails_enable_pre_approval(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _guarded_agent()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                async for _ in chat_mod.chat_with_hooks("Hi", agent_name="a", session=session):
                    pass

        run_config = mock_runner.run.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_agent_without_guardrails_gets_no_run_config(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                async for _ in chat_mod.chat_with_hooks("Hi", agent_name="a", session=session):
                    pass

        assert "run_config" not in mock_runner.run.call_args.kwargs

    async def test_declared_trim_policy_reaches_the_run(self, trim_registry):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _trimming_agent(trim_registry)
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                async for _ in chat_mod.chat_with_hooks(
                    "Hi", agent_name="_chat_trimming_agent", session=session
                ):
                    pass

        run_config = mock_runner.run.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_structured_output_agent_gets_recovery_handler(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=_run_result())

            async for _ in chat_mod.chat_with_hooks(
                "Hi", agent=_structured_agent(), session=session
            ):
                pass

        handlers = mock_runner.run.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    async def test_plain_text_agent_gets_no_error_handlers(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=_run_result())

                async for _ in chat_mod.chat_with_hooks("Hi", agent_name="a", session=session):
                    pass

        assert "error_handlers" not in mock_runner.run.call_args.kwargs

    async def test_declared_recovery_opt_out_reaches_the_run(self, recovery_registry):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=_run_result())

            async for _ in chat_mod.chat_with_hooks(
                "Hi", agent=_strict_agent(recovery_registry), session=session
            ):
                pass

        assert "error_handlers" not in mock_runner.run.call_args.kwargs

    async def test_error_yields_error_event(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.side_effect = ValueError("Agent missing")

            events = []
            async for event in chat_mod.chat_with_hooks(
                "Hi", agent_name="missing", session=session
            ):
                events.append(event)

        assert any(e["event"] == "error" for e in events)

    @pytest.mark.parametrize(("error", "expected"), RUN_FAILURES)
    async def test_run_failure_reports_its_kind(self, error, expected):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(side_effect=error)

                events = [
                    event
                    async for event in chat_mod.chat_with_hooks(
                        "Hi", agent_name="a", session=session
                    )
                ]

        errors = [e for e in events if e["event"] == "error"]
        assert errors == [
            {"event": "error", "data": {"error": str(error), "error_kind": expected.value}}
        ]


class TestChatWithHooksAnswerOwnsItsToolList:
    """The answer payload is a fixed record, so it and the hooks never share a list."""

    @staticmethod
    async def _answer_and_hooks(tool_name="search"):
        """Run one turn that calls a tool; return its answer payload and the run's hooks."""
        import sys

        chat_mod = sys.modules["sinan_agentic_core.services.chat"]
        captured = []

        async def run_with_hooks(**kwargs):
            hooks = kwargs["hooks"]
            captured.append(hooks)
            tool = Mock()
            tool.name = tool_name
            await hooks.on_tool_start(None, None, tool)
            return _run_result("done")

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(side_effect=run_with_hooks)

                events = [
                    event
                    async for event in chat_mod.chat_with_hooks(
                        "Hi", agent_name="a", session=session
                    )
                ]

        answer = next(e for e in events if e["event"] == "answer")
        return answer["data"], captured[0]

    async def test_the_payload_reports_the_tools_the_hooks_recorded(self):
        data, hooks = await self._answer_and_hooks()

        assert data["tools_called"] == ["search"]
        assert hooks.tools_called == ["search"]

    async def test_the_payload_does_not_share_the_hooks_accumulator(self):
        data, hooks = await self._answer_and_hooks()

        assert data["tools_called"] is not hooks.tools_called

    async def test_a_consumer_editing_the_payload_does_not_reach_the_hooks(self):
        data, hooks = await self._answer_and_hooks()

        data["tools_called"].append("added_by_consumer")

        assert hooks.tools_called == ["search"]

    async def test_a_later_tool_call_does_not_change_a_delivered_answer(self):
        """Reusing the hooks after the answer cannot rewrite what it already reported."""
        data, hooks = await self._answer_and_hooks()

        late_tool = Mock()
        late_tool.name = "added_late"
        await hooks.on_tool_start(None, None, late_tool)

        assert data["tools_called"] == ["search"]
        assert hooks.tools_called == ["search", "added_late"]


# -- chat_streamed() ----------------------------------------------------------


class TestChatStreamed:
    @staticmethod
    def _get_chat_module():
        import sys

        return sys.modules["sinan_agentic_core.services.chat"]

    async def test_yields_stream_events(self):
        from agents import Usage
        from openai.types.responses import ResponseTextDeltaEvent

        chat_mod = self._get_chat_module()

        mock_usage = Usage(requests=1, input_tokens=60, output_tokens=30, total_tokens=90)
        mock_response = Mock()
        mock_response.usage = mock_usage

        # Text delta event
        mock_text_data = Mock(spec=ResponseTextDeltaEvent)
        mock_text_data.delta = "Hello"
        mock_text_event = Mock()
        mock_text_event.type = "raw_response_event"
        mock_text_event.data = mock_text_data

        # Tool call event
        mock_tool_event = Mock()
        mock_tool_event.type = "run_item_stream_event"
        mock_tool_event.item = Mock()
        mock_tool_event.item.type = "tool_call_item"
        mock_tool_event.item.name = "search"

        # Tool output event
        mock_tool_output = Mock()
        mock_tool_output.type = "run_item_stream_event"
        mock_tool_output.item = Mock()
        mock_tool_output.item.type = "tool_call_output_item"
        mock_tool_output.item.output = "tool result data"

        # Message output event
        mock_msg_event = Mock()
        mock_msg_event.type = "run_item_stream_event"
        mock_msg_event.item = Mock()
        mock_msg_event.item.type = "message_output_item"

        # Agent updated event
        mock_agent_event = Mock()
        mock_agent_event.type = "agent_updated_stream_event"
        mock_agent_event.new_agent = Mock()
        mock_agent_event.new_agent.name = "sub_agent"

        # Build mock streaming result
        mock_result = Mock()
        mock_result.final_output = "Streamed answer"
        mock_result.new_items = []
        mock_result.raw_responses = [mock_response]

        async def mock_stream_events():
            yield mock_text_event
            yield mock_tool_event
            yield mock_tool_output
            yield mock_msg_event
            yield mock_agent_event

        mock_result.stream_events = mock_stream_events

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = mock_result
                with patch.object(chat_mod, "ItemHelpers") as mock_helpers:
                    mock_helpers.text_message_output.return_value = "full message"

                    events = []
                    async for event in chat_mod.chat_streamed(
                        "Hi", agent_name="test_agent", session=session
                    ):
                        events.append(event)

        event_types = [e["event"] for e in events]
        assert "text_delta" in event_types
        assert "tool_call" in event_types
        assert "tool_output" in event_types
        assert "message_output" in event_types
        assert "agent_updated" in event_types
        assert "answer" in event_types

        answer = next(e for e in events if e["event"] == "answer")
        assert answer["data"]["response"] == "Streamed answer"
        assert "search" in answer["data"]["tools_called"]
        assert "usage" in answer["data"]

    async def test_with_context(self):
        from agents import Usage

        chat_mod = self._get_chat_module()

        mock_usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response = Mock()
        mock_response.usage = mock_usage

        mock_result = Mock()
        mock_result.final_output = "ctx"
        mock_result.new_items = []
        mock_result.raw_responses = [mock_response]

        async def empty_stream():
            return
            yield  # make it an async generator

        mock_result.stream_events = empty_stream

        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = mock_result

                events = []
                async for event in chat_mod.chat_streamed(
                    "Hi", agent_name="a", session=session, context=Mock()
                ):
                    events.append(event)

        assert any(e["event"] == "answer" for e in events)

    async def test_tool_input_guardrails_enable_pre_approval(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _guarded_agent()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = _streamed_result()

                async for _ in chat_mod.chat_streamed("Hi", agent_name="a", session=session):
                    pass

        run_config = mock_runner.run_streamed.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_agent_without_guardrails_gets_no_run_config(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = _streamed_result()

                async for _ in chat_mod.chat_streamed("Hi", agent_name="a", session=session):
                    pass

        assert "run_config" not in mock_runner.run_streamed.call_args.kwargs

    async def test_declared_trim_policy_reaches_the_run(self, trim_registry):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _trimming_agent(trim_registry)
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = _streamed_result()

                async for _ in chat_mod.chat_streamed(
                    "Hi", agent_name="_chat_trimming_agent", session=session
                ):
                    pass

        run_config = mock_runner.run_streamed.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_structured_output_agent_gets_recovery_handler(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run_streamed.return_value = _streamed_result()

            async for _ in chat_mod.chat_streamed("Hi", agent=_structured_agent(), session=session):
                pass

        handlers = mock_runner.run_streamed.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    async def test_plain_text_agent_gets_no_error_handlers(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = _streamed_result()

                async for _ in chat_mod.chat_streamed("Hi", agent_name="a", session=session):
                    pass

        assert "error_handlers" not in mock_runner.run_streamed.call_args.kwargs

    async def test_declared_recovery_opt_out_reaches_the_run(self, recovery_registry):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "Runner") as mock_runner:
            mock_runner.run_streamed.return_value = _streamed_result()

            async for _ in chat_mod.chat_streamed(
                "Hi", agent=_strict_agent(recovery_registry), session=session
            ):
                pass

        assert "error_handlers" not in mock_runner.run_streamed.call_args.kwargs

    async def test_error_yields_error_event(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.side_effect = RuntimeError("Stream failed")

            events = []
            async for event in chat_mod.chat_streamed("Hi", agent_name="missing", session=session):
                events.append(event)

        assert any(e["event"] == "error" for e in events)

    @pytest.mark.parametrize(("error", "expected"), RUN_FAILURES)
    async def test_run_failure_reports_its_kind(self, error, expected):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.side_effect = error

                events = [
                    event
                    async for event in chat_mod.chat_streamed("Hi", agent_name="a", session=session)
                ]

        errors = [e for e in events if e["event"] == "error"]
        assert errors == [
            {"event": "error", "data": {"error": str(error), "error_kind": expected.value}}
        ]


def _reasoning_result(summary_texts, final_output="ok"):
    """A finished run whose new_items carry one reasoning item."""
    from agents.items import ReasoningItem
    from openai.types.responses import ResponseReasoningItem

    item = ReasoningItem(
        agent=Mock(),
        raw_item=ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[{"text": text, "type": "summary_text"} for text in summary_texts],
        ),
    )
    result = Mock()
    result.final_output = final_output
    result.raw_responses = []
    result.new_items = [item]
    return result


class TestChatReasoning:
    """The non-streaming path returns only final_output, so it reports the rest."""

    @staticmethod
    def _get_chat_module():
        import sys

        return sys.modules["sinan_agentic_core.services.chat"]

    async def _chat(self, result):
        chat_mod = self._get_chat_module()

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=result)
                return await chat_mod.chat(
                    "Hi", agent_name="test_agent", session=AgentSession(session_id="test")
                )

    async def test_the_result_reports_what_the_model_reasoned(self):
        result = await self._chat(_reasoning_result(["first thought", "then this"]))

        assert result["reasoning"] == ["first thought", "then this"]

    async def test_a_run_without_reasoning_reports_an_empty_list(self):
        """Absent, not missing — a caller reads the key unconditionally."""
        result = await self._chat(_reasoning_result([]))

        assert result["reasoning"] == []


class TestChatWithHooksReasoning:
    """This path runs to completion, so the whole summary arrives before the answer."""

    @staticmethod
    def _get_chat_module():
        import sys

        return sys.modules["sinan_agentic_core.services.chat"]

    async def _stream(self, result):
        chat_mod = self._get_chat_module()

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=result)
                return [
                    event
                    async for event in chat_mod.chat_with_hooks(
                        "Hi", agent_name="test_agent", session=AgentSession(session_id="test")
                    )
                ]

    async def test_the_whole_summary_arrives_as_one_event(self):
        events = await self._stream(_reasoning_result(["first thought", "then this"]))

        reasoning = next(e for e in events if e["event"] == "reasoning")
        assert reasoning["data"]["summary"] == ["first thought", "then this"]

    async def test_it_arrives_before_the_answer(self):
        """A step log renders the thinking that led to the answer, not after it."""
        events = await self._stream(_reasoning_result(["first thought"]))

        names = [e["event"] for e in events]
        assert names.index("reasoning") < names.index("answer")

    async def test_a_run_without_reasoning_says_nothing(self):
        events = await self._stream(_reasoning_result([]))

        assert not [e for e in events if e["event"] == "reasoning"]


class TestChatStreamedReasoning:
    """OPUS reads this path, so the model's thinking has to reach it here too."""

    @staticmethod
    def _get_chat_module():
        import sys

        return sys.modules["sinan_agentic_core.services.chat"]

    @staticmethod
    def _raw(data):
        event = Mock()
        event.type = "raw_response_event"
        event.data = data
        return event

    @staticmethod
    def _reasoning_item(summary_texts):
        from agents.items import ReasoningItem
        from openai.types.responses import ResponseReasoningItem

        event = Mock()
        event.type = "run_item_stream_event"
        event.item = ReasoningItem(
            agent=Mock(),
            raw_item=ResponseReasoningItem(
                id="rs_1",
                type="reasoning",
                summary=[{"text": text, "type": "summary_text"} for text in summary_texts],
            ),
        )
        return event

    async def _stream(self, raw_events):
        chat_mod = self._get_chat_module()

        mock_result = Mock()
        mock_result.final_output = "Streamed answer"
        mock_result.new_items = []
        mock_result.raw_responses = []
        mock_result.new_items = []

        async def stream_events():
            for raw in raw_events:
                yield raw

        mock_result.stream_events = stream_events

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = mock_result
                return [
                    event
                    async for event in chat_mod.chat_streamed(
                        "Hi", agent_name="test_agent", session=AgentSession(session_id="test")
                    )
                ]

    async def test_the_summary_text_is_forwarded_chunk_by_chunk(self):
        from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent

        def delta(text, index=0):
            return ResponseReasoningSummaryTextDeltaEvent(
                delta=text,
                item_id="rs_1",
                output_index=0,
                sequence_number=1,
                summary_index=index,
                type="response.reasoning_summary_text.delta",
            )

        events = await self._stream([self._raw(delta("Look")), self._raw(delta("ing", 1))])

        forwarded = [e["data"] for e in events if e["event"] == "reasoning_delta"]
        assert forwarded == [{"delta": "Look", "index": 0}, {"delta": "ing", "index": 1}]

    async def test_the_part_boundaries_bracket_each_thought(self):
        from openai.types.responses import (
            ResponseReasoningSummaryPartAddedEvent,
            ResponseReasoningSummaryPartDoneEvent,
        )

        added = ResponseReasoningSummaryPartAddedEvent(
            item_id="rs_1",
            output_index=0,
            part={"text": "", "type": "summary_text"},
            sequence_number=1,
            summary_index=0,
            type="response.reasoning_summary_part.added",
        )
        done = ResponseReasoningSummaryPartDoneEvent(
            item_id="rs_1",
            output_index=0,
            part={"text": "Reading the schema", "type": "summary_text"},
            sequence_number=2,
            summary_index=0,
            type="response.reasoning_summary_part.done",
        )

        events = await self._stream([self._raw(added), self._raw(done)])

        assert [e for e in events if e["event"].startswith("reasoning_part")] == [
            {"event": "reasoning_part_added", "data": {"index": 0}},
            {"event": "reasoning_part_done", "data": {"index": 0, "text": "Reading the schema"}},
        ]

    async def test_the_terminal_event_repeats_every_thought(self):
        events = await self._stream([self._reasoning_item(["first thought", "then this"])])

        reasoning = next(e for e in events if e["event"] == "reasoning")
        assert reasoning["data"]["summary"] == ["first thought", "then this"]

    async def test_a_reasoning_item_without_a_summary_says_nothing(self):
        events = await self._stream([self._reasoning_item([])])

        assert not [e for e in events if e["event"] == "reasoning"]

    async def test_an_unrelated_raw_event_is_still_ignored(self):
        """The new branch is a fallthrough, so it must not turn noise into events."""
        events = await self._stream([self._raw(Mock())])

        assert [e["event"] for e in events] == ["answer"]


class TestChatStreamedAnswerOwnsItsToolList:
    """The answer payload is a fixed record, so it and the stream never share a list."""

    @staticmethod
    def _tool_call_event(tool_name="search"):
        event = Mock()
        event.type = "run_item_stream_event"
        event.item = Mock()
        event.item.type = "tool_call_item"
        event.item.name = tool_name
        return event

    @classmethod
    async def _answer_and_accumulator(cls):
        """Stream one turn that calls a tool.

        Returns the answer payload and the list the stream accumulated tool
        names into.  That accumulator is a local of the generator, so the frame
        it is suspended in is the only handle on the object the payload must
        not be — the counterpart of ``hooks.tools_called`` for the hooks path.
        """
        import sys

        chat_mod = sys.modules["sinan_agentic_core.services.chat"]

        result = _run_result("Streamed answer")

        async def stream_events():
            yield cls._tool_call_event()

        result.stream_events = stream_events
        session = AgentSession(session_id="test")
        payload, accumulator = None, None

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = result

                stream = chat_mod.chat_streamed("Hi", agent_name="a", session=session)
                async for event in stream:
                    if event["event"] == "answer":
                        payload = event["data"]
                        accumulator = stream.ag_frame.f_locals["tools_called"]
                        break
                await stream.aclose()

        return payload, accumulator

    async def test_the_payload_reports_the_tools_the_stream_recorded(self):
        payload, accumulator = await self._answer_and_accumulator()

        assert payload["tools_called"] == ["search"]
        assert accumulator == ["search"]

    async def test_the_payload_does_not_share_the_streams_accumulator(self):
        payload, accumulator = await self._answer_and_accumulator()

        assert payload["tools_called"] is not accumulator

    async def test_a_consumer_editing_the_payload_does_not_reach_the_stream(self):
        payload, accumulator = await self._answer_and_accumulator()

        payload["tools_called"].append("added_by_consumer")

        assert accumulator == ["search"]


# -- guardrail tripwires across all three chat functions -----------------------


class TestGuardrailTripwireReachesEveryChatFunction:
    """One tripwire, three entry points, one report.

    ``chat()`` and ``chat_with_hooks()`` run through ``Runner.run()`` while
    ``chat_streamed()`` runs through ``Runner.run_streamed()``. Before
    openai-agents 0.19.2 the non-streamed pair discarded the guardrail
    accumulator when the tripwire raised, so the same handler saw an empty
    result list there and a full one under streaming. All three now report the
    guardrail that rejected the run and everything that finished beside it.
    """

    @staticmethod
    def _get_chat_module():
        import sys

        return sys.modules["sinan_agentic_core.services.chat"]

    @staticmethod
    def _tripwire():
        return make_input_tripwire_error(
            registered_input_guardrail("blocks_pii"),
            passed=[registered_input_guardrail("off_topic")],
        )

    EXPECTED_GUARDRAIL = {
        "name": "blocks_pii",
        "results": [
            {"name": "off_topic", "tripwire_triggered": False},
            {"name": "blocks_pii", "tripwire_triggered": True},
        ],
    }

    async def _chat_failure(self):
        chat_mod = self._get_chat_module()
        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(side_effect=self._tripwire())
                result = await chat_mod.chat(
                    "Hi", agent_name="a", session=AgentSession(session_id="test")
                )
        return {key: result[key] for key in ("error", "error_kind", "guardrail")}

    async def _hooks_failure(self):
        chat_mod = self._get_chat_module()
        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(side_effect=self._tripwire())
                events = [
                    event
                    async for event in chat_mod.chat_with_hooks(
                        "Hi", agent_name="a", session=AgentSession(session_id="test")
                    )
                ]
        return next(e["data"] for e in events if e["event"] == "error")

    async def _streamed_failure(self):
        chat_mod = self._get_chat_module()
        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.return_value = _agent_double()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.side_effect = self._tripwire()
                events = [
                    event
                    async for event in chat_mod.chat_streamed(
                        "Hi", agent_name="a", session=AgentSession(session_id="test")
                    )
                ]
        return next(e["data"] for e in events if e["event"] == "error")

    async def test_chat_reports_the_guardrail(self):
        failure = await self._chat_failure()
        assert failure["error_kind"] == RunErrorKind.INPUT_GUARDRAIL_TRIPWIRE.value
        assert failure["guardrail"] == self.EXPECTED_GUARDRAIL

    async def test_all_three_report_the_same_thing(self):
        reports = [
            await self._chat_failure(),
            await self._hooks_failure(),
            await self._streamed_failure(),
        ]
        assert reports.count(reports[0]) == len(reports)
