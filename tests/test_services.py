"""Tests for services: events, hooks, usage helper, and chat (mocked Runner)."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

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
from sinan_agentic_core.services.chat import _usage_to_dict
from sinan_agentic_core.session.agent_session import AgentSession


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


class TestChat:
    @staticmethod
    def _get_chat_module():
        import sys
        # Import to ensure the module is loaded, then get from sys.modules
        # to avoid the __init__.py shadowing the module with the function
        import sinan_agentic_core.services.chat  # noqa: F811
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
            mock_factory.return_value = Mock()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=mock_result)

                result = await chat_mod.chat(
                    "Hi", agent_name="test_agent", session=session
                )

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

            result = await chat_mod.chat(
                "Hi", agent_name="missing", session=session
            )

        assert result["success"] is False
        assert "Agent not found" in result["error"]

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
            mock_factory.return_value = Mock()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=mock_result)

                result = await chat_mod.chat(
                    "Hi", agent_name="a", session=session, context=Mock()
                )

        assert result["success"] is True
        # Verify context was forwarded to Runner.run
        call_kwargs = mock_runner.run.call_args
        assert "context" in call_kwargs.kwargs


# -- chat_with_hooks() --------------------------------------------------------


class TestChatWithHooks:
    @staticmethod
    def _get_chat_module():
        import sys
        import sinan_agentic_core.services.chat  # noqa: F811
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
            mock_factory.return_value = Mock()
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
            mock_factory.return_value = Mock()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run = AsyncMock(return_value=mock_result)

                events = []
                async for event in chat_mod.chat_with_hooks(
                    "Hi", agent_name="a", session=session, context=Mock()
                ):
                    events.append(event)

        assert any(e["event"] == "answer" for e in events)

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


# -- chat_streamed() ----------------------------------------------------------


class TestChatStreamed:
    @staticmethod
    def _get_chat_module():
        import sys
        import sinan_agentic_core.services.chat  # noqa: F811
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
            mock_factory.return_value = Mock()
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
            mock_factory.return_value = Mock()
            with patch.object(chat_mod, "Runner") as mock_runner:
                mock_runner.run_streamed.return_value = mock_result

                events = []
                async for event in chat_mod.chat_streamed(
                    "Hi", agent_name="a", session=session, context=Mock()
                ):
                    events.append(event)

        assert any(e["event"] == "answer" for e in events)

    async def test_error_yields_error_event(self):
        chat_mod = self._get_chat_module()
        session = AgentSession(session_id="test")

        with patch.object(chat_mod, "create_agent_from_registry") as mock_factory:
            mock_factory.side_effect = RuntimeError("Stream failed")

            events = []
            async for event in chat_mod.chat_streamed(
                "Hi", agent_name="missing", session=session
            ):
                events.append(event)

        assert any(e["event"] == "error" for e in events)
