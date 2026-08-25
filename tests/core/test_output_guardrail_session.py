"""Tests that a tripped output guardrail leaves no blocked output in the session.

An output guardrail exists to keep an answer away from the caller. Persisting
that answer defeats it a different way: the caller never sees it, but the model
reads it back as history on the next turn. openai-agents 0.19.3/0.19.4 moved the
session save behind the output guardrails on both run paths, so a tripwire now
drops the assistant output while keeping the turn's input.

Both paths are live here — ``BaseAgentRunner`` wires ``output_guardrails`` onto
every agent it builds and passes a session into every run — and the session
receiving the writes is this package's own ``SessionABC`` implementation. These
tests drive the real SDK loop against a scripted model so the retention policy
is asserted where the two meet, not against a mocked ``Runner``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelResponse,
    OutputGuardrailTripwireTriggered,
    Runner,
    Usage,
    output_guardrail,
)
from agents.models.interface import Model
from openai.types.responses import Response, ResponseCompletedEvent

from sinan_agentic_core.core.base_runner import _CollectingSessionWrapper
from sinan_agentic_core.session.agent_session import AgentSession
from tests.core.conftest import ASSISTANT_ROLE, assistant_message

# The answer the scripted model produces. A guardrail either blocks it or lets
# it through, and the session is searched for this exact text either way.
BLOCKED_OUTPUT = "the answer a guardrail rejects"

USER_INPUT = "produce an answer"


class ScriptedModel(Model):
    """A model that answers with one fixed assistant message, never calling out.

    Both SDK entry points are implemented so one instance serves the streamed
    and the non-streamed run alike.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Return the scripted message as a completed non-streamed response."""
        return ModelResponse(
            output=[assistant_message(self._text, message_id="msg_scripted")],
            usage=Usage(),
            response_id="resp_scripted",
        )

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Emit the scripted message as a single ``response.completed`` event."""
        yield ResponseCompletedEvent(
            response=Response(
                id="resp_scripted",
                created_at=0.0,
                model="scripted",
                object="response",
                output=[assistant_message(self._text, message_id="msg_scripted")],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
            sequence_number=0,
            type="response.completed",
        )


@output_guardrail
async def trips(ctx: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
    """Reject every answer."""
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)


@output_guardrail
async def passes(ctx: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
    """Accept every answer."""
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)


def _agent(guardrail: Any) -> Agent[Any]:
    """An agent answering with ``BLOCKED_OUTPUT`` behind *guardrail*."""
    return Agent(
        name="guarded",
        instructions="answer",
        model=ScriptedModel(BLOCKED_OUTPUT),
        output_guardrails=[guardrail],
    )


def _assistant_contents(items: list[Any]) -> list[str]:
    """The content of every assistant message in *items*."""
    return [item["content"] for item in items if item.get("role") == ASSISTANT_ROLE]


class TestNonStreamedRun:
    """``Runner.run`` — the path ``BaseAgentRunner._execute_run`` takes."""

    async def test_tripped_guardrail_keeps_blocked_output_out_of_the_session(self) -> None:
        session = AgentSession(session_id="blocked-non-streamed")

        with pytest.raises(OutputGuardrailTripwireTriggered):
            await Runner.run(_agent(trips), USER_INPUT, session=session)

        assert _assistant_contents(await session.get_items()) == []

    async def test_tripped_guardrail_still_keeps_the_turn_input(self) -> None:
        """The rejected answer is dropped, not the turn that asked for it."""
        session = AgentSession(session_id="input-retained")

        with pytest.raises(OutputGuardrailTripwireTriggered):
            await Runner.run(_agent(trips), USER_INPUT, session=session)

        assert [item["content"] for item in await session.get_items()] == [USER_INPUT]

    async def test_passing_guardrail_persists_the_output(self) -> None:
        """Without this the blocked-output assertions would hold vacuously."""
        session = AgentSession(session_id="allowed-non-streamed")

        await Runner.run(_agent(passes), USER_INPUT, session=session)

        assert _assistant_contents(await session.get_items()) == [BLOCKED_OUTPUT]


class TestStreamedRun:
    """``Runner.run_streamed`` — the path ``BaseAgentRunner._execute_streamed`` takes."""

    async def test_tripped_guardrail_keeps_blocked_output_out_of_the_session(self) -> None:
        session = AgentSession(session_id="blocked-streamed")
        stream = Runner.run_streamed(_agent(trips), USER_INPUT, session=session)

        with pytest.raises(OutputGuardrailTripwireTriggered):
            async for _ in stream.stream_events():
                pass

        assert _assistant_contents(await session.get_items()) == []

    async def test_passing_guardrail_persists_the_output(self) -> None:
        session = AgentSession(session_id="allowed-streamed")
        stream = Runner.run_streamed(_agent(passes), USER_INPUT, session=session)

        async for _ in stream.stream_events():
            pass

        assert _assistant_contents(await session.get_items()) == [BLOCKED_OUTPUT]


class TestFallbackSessionWrapper:
    """The fallback path runs against ``_CollectingSessionWrapper``, not the session itself.

    The wrapper forwards ``add_items``, so it can only collect what the SDK
    chose to save. A blocked answer must therefore reach neither the collector
    nor the session behind it.
    """

    async def test_tripped_guardrail_reaches_neither_collector_nor_session(self) -> None:
        session = AgentSession(session_id="blocked-fallback")
        collecting = _CollectingSessionWrapper(session)

        with pytest.raises(OutputGuardrailTripwireTriggered):
            await Runner.run(_agent(trips), USER_INPUT, session=collecting)

        assert _assistant_contents(await session.get_items()) == []
        assert not [item for item in collecting.raw_items if BLOCKED_OUTPUT in str(item)]
