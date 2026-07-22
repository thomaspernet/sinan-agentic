"""Tests for services: events, hooks, usage helper, and chat (mocked Runner)."""

import asyncio
import copy
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
from tests.conftest import make_context_overflow_error

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
    """An isolated registry, patched where ``build_run_config`` looks the definition up."""
    from sinan_agentic_core.registry.agent_registry import AgentRegistry

    registry = AgentRegistry()
    with patch("sinan_agentic_core.core.run_config.get_agent_registry", return_value=registry):
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
    """An isolated registry, patched where ``build_error_handlers`` looks the definition up."""
    from sinan_agentic_core.registry.agent_registry import AgentRegistry

    registry = AgentRegistry()
    with patch("sinan_agentic_core.core.output_recovery.get_agent_registry", return_value=registry):
        yield registry


def _run_result(response="ok"):
    """A run result carrying one response — enough for the usage aggregation."""
    raw = Mock()
    raw.usage = Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2)
    result = Mock()
    result.final_output = response
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
