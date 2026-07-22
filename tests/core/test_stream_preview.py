"""Tests for streamed tool-output previews (core/stream_preview.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from sinan_agentic_core.core.base_runner import BaseAgentRunner
from sinan_agentic_core.core.stream_preview import (
    TOOL_OUTPUT_PREVIEW_CHARS,
    tool_output_preview,
)
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from sinan_agentic_core.registry.guardrail_registry import GuardrailRegistry
from sinan_agentic_core.registry.tool_registry import ToolRegistry
from sinan_agentic_core.session.agent_session import AgentSession

AGENT_NAME = "streaming_agent"

# Three times the preview, so a path that kept its own length would have to match
# this one exactly to pass rather than merely being close.
BULKY_OUTPUT = "x" * (TOOL_OUTPUT_PREVIEW_CHARS * 3)


class TestToolOutputPreview:
    def test_short_output_survives_whole(self) -> None:
        assert tool_output_preview("done") == "done"

    def test_output_at_the_limit_survives_whole(self) -> None:
        exact = "y" * TOOL_OUTPUT_PREVIEW_CHARS
        assert tool_output_preview(exact) == exact

    def test_longer_output_is_cut_to_the_limit(self) -> None:
        preview = tool_output_preview(BULKY_OUTPUT)

        assert len(preview) == TOOL_OUTPUT_PREVIEW_CHARS
        assert BULKY_OUTPUT.startswith(preview)

    def test_non_string_output_is_rendered_before_being_cut(self) -> None:
        """A tool returns whatever its signature declares, not only strings."""
        assert tool_output_preview({"rows": 3}) == "{'rows': 3}"

    def test_a_rendered_non_string_is_cut_too(self) -> None:
        preview = tool_output_preview(["item"] * 1000)

        assert len(preview) == TOOL_OUTPUT_PREVIEW_CHARS


def _streamed_result(output: object) -> Mock:
    """A streamed run whose only event is one tool returning *output*."""
    item = Mock()
    item.type = "tool_call_output_item"
    item.output = output

    event = Mock()
    event.type = "run_item_stream_event"
    event.item = item

    async def stream_events() -> Any:
        yield event

    result = Mock()
    result.final_output = "answer"
    result.raw_responses = []
    result.stream_events = stream_events
    return result


async def _runner_events(output: object) -> list[dict[str, Any]]:
    """Events ``BaseAgentRunner._execute_streamed`` emits for one tool output."""
    agent_reg = AgentRegistry()
    agent_reg.register(
        AgentDefinition(name=AGENT_NAME, description="streams", instructions="stream")
    )

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=ToolRegistry()),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry",
            return_value=GuardrailRegistry(),
        ),
    ):
        runner = BaseAgentRunner()

    events: list[dict[str, Any]] = []
    with (
        patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
        patch("sinan_agentic_core.core.base_runner.Runner") as sdk_runner,
    ):
        sdk_runner.run_streamed = Mock(return_value=_streamed_result(output))
        await runner._execute_streamed(
            AGENT_NAME,
            Mock(),
            AgentSession(session_id="runner"),
            events.append,
            10,
            "hello",
        )

    return events


async def _chat_events(output: object) -> list[dict[str, Any]]:
    """Events ``chat_streamed`` yields for one tool output."""
    import sys

    chat_mod = sys.modules["sinan_agentic_core.services.chat"]

    with (
        patch.object(chat_mod, "create_agent_from_registry", return_value=Mock(tools=[])),
        patch.object(chat_mod, "Runner") as sdk_runner,
    ):
        sdk_runner.run_streamed = Mock(return_value=_streamed_result(output))
        return [
            event
            async for event in chat_mod.chat_streamed(
                "hello", agent_name=AGENT_NAME, session=AgentSession(session_id="chat")
            )
        ]


def _tool_output_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(event for event in events if event["event"] == "tool_output")


class TestStreamingPathsAgree:
    """Both streaming paths preview one tool's output the same way."""

    async def test_runner_previews_at_the_shared_length(self) -> None:
        event = _tool_output_event(await _runner_events(BULKY_OUTPUT))

        assert event["data"]["output"] == tool_output_preview(BULKY_OUTPUT)

    async def test_chat_service_previews_at_the_shared_length(self) -> None:
        event = _tool_output_event(await _chat_events(BULKY_OUTPUT))

        assert event["data"]["output"] == tool_output_preview(BULKY_OUTPUT)

    async def test_both_paths_emit_the_same_event(self) -> None:
        """One declaration, so neither path can drift to its own length."""
        runner_event = _tool_output_event(await _runner_events(BULKY_OUTPUT))
        chat_event = _tool_output_event(await _chat_events(BULKY_OUTPUT))

        assert runner_event == chat_event
