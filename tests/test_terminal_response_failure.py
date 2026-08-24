"""Tests that a terminal provider response fails the turn instead of emptying it.

A Responses turn can end in ``failed`` or ``incomplete`` — an output-token cap,
a content filter, a provider-side error — and the payload then carries no
output. openai-agents 0.21.1 raised ``ModelBehaviorError`` for that on the
streamed and websocket paths but not on the non-streamed one, where
``get_response()`` handed the run a ``ModelResponse(output=[])``: a successful
empty turn. ``chat()`` reported ``{"success": True}`` with an empty response and
committed an empty assistant message to the session, which the model then read
back as history. 0.22.0 raises on that path too.

Both halves of the change are pinned here, and against the real seam: the
non-streamed Responses path is the framework's default (``api_mode`` defaults to
``"responses"``, ``chat()`` runs unstreamed), so these tests drive a real
``OpenAIResponsesModel`` over a client whose ``responses.create`` returns the
terminal payload, rather than raising a hand-built exception at a mocked
``Runner``. The failure therefore originates where the provider's would.

The kind the failure classifies as is asserted here too, because it is the
decision ``FALLBACK_RECOVERABLE_KINDS`` was left unchanged on: a terminal status
raises before any final-output validation, so it never meets the
``invalid_final_output`` handler, and it is still not something a condensed
second call can rescue.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents import Agent, ModelBehaviorError, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from agents.testing import assistant_message
from openai import AsyncOpenAI
from openai.types.responses import Response
from openai.types.responses.response import IncompleteDetails

from sinan_agentic_core.core.run_errors import (
    FALLBACK_RECOVERABLE_KINDS,
    RunErrorKind,
    classify_run_error,
)
from sinan_agentic_core.services.chat import chat
from sinan_agentic_core.session.agent_session import AgentSession

# The two terminal statuses a Responses payload can carry. ``incomplete`` is the
# output-token cap; ``failed`` is a provider-side error or a content filter.
TERMINAL_STATUSES = ["failed", "incomplete"]

MODEL_NAME = "gpt-4o-mini"

USER_INPUT = "produce an answer"

ANSWERED_OUTPUT = "the answer a completed turn carries"

ASSISTANT_ROLE = "assistant"


def _response(status: str, output: list[Any]) -> Response:
    """Build the Responses payload a provider returns with *status*.

    ``incomplete_details`` is populated for an ``incomplete`` payload the way the
    provider populates it, since the reason is what the SDK renders into the
    error message.
    """
    return Response(
        id=f"resp_{status}",
        created_at=0.0,
        model=MODEL_NAME,
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status=status,
        incomplete_details=(
            IncompleteDetails(reason="max_output_tokens") if status == "incomplete" else None
        ),
    )


def _agent_over(response: Response) -> Agent[Any]:
    """An agent on the real non-streamed Responses path, answering with *response*.

    Only the transport call is replaced, so every layer the provider's payload
    normally crosses — the terminal-status check included — is the SDK's own.
    """
    client = AsyncOpenAI(api_key="test")
    client.responses.create = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return Agent(
        name="terminal_responder",
        instructions="answer",
        model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client),
    )


async def _chat_over(response: Response, session_id: str) -> tuple[dict[str, Any], list[Any]]:
    """Run one ``chat()`` turn against *response*, returning it and the session."""
    session = AgentSession(session_id=session_id)
    result = await chat(USER_INPUT, agent=_agent_over(response), session=session)
    return result, await session.get_items()


class TestTerminalResponseFailsTheTurn:
    """A terminal payload is a failed turn, not a successful empty one."""

    @pytest.mark.parametrize("status", TERMINAL_STATUSES)
    async def test_chat_reports_the_failure(self, status):
        result, _ = await _chat_over(_response(status, output=[]), f"terminal-{status}")

        assert result["success"] is False
        assert result["error_kind"] == RunErrorKind.MODEL_BEHAVIOR.value

    @pytest.mark.parametrize("status", TERMINAL_STATUSES)
    async def test_nothing_is_written_to_the_session(self, status):
        """The empty turn used to be committed as an assistant message the model
        read back as history. A raise leaves the turn's input and nothing else."""
        _, items = await _chat_over(_response(status, output=[]), f"session-{status}")

        assert [item["role"] for item in items] == ["user"]

    @pytest.mark.parametrize("status", TERMINAL_STATUSES)
    async def test_the_failure_carries_no_response(self, status):
        result, _ = await _chat_over(_response(status, output=[]), f"payload-{status}")

        assert "response" not in result

    async def test_a_completed_turn_still_answers(self):
        """The control: the status is what fails the turn, not the transport stub."""
        completed = _response("completed", output=[assistant_message(ANSWERED_OUTPUT)])
        result, items = await _chat_over(completed, "completed")

        assert result["success"] is True
        assert result["response"] == ANSWERED_OUTPUT
        assert [item["role"] for item in items] == ["user", ASSISTANT_ROLE]


class TestTerminalResponseClassification:
    """The kind the raise classifies as, read off the exception the SDK builds."""

    @pytest.mark.parametrize("status", TERMINAL_STATUSES)
    async def test_it_classifies_as_model_behavior(self, status):
        error = await _raised_by(_response(status, output=[]))

        assert classify_run_error(error) is RunErrorKind.MODEL_BEHAVIOR

    @pytest.mark.parametrize("status", TERMINAL_STATUSES)
    async def test_it_is_not_worth_a_condensed_second_call(self, status):
        """Recorded decision: the kind stays outside the recoverable set. It
        cannot be admitted for the terminal subset alone — malformed output
        shares the kind, and only the message tells the two apart."""
        error = await _raised_by(_response(status, output=[]))

        assert classify_run_error(error) not in FALLBACK_RECOVERABLE_KINDS


async def _raised_by(response: Response) -> ModelBehaviorError:
    """The exception the real non-streamed Responses path raises for *response*.

    ``chat()`` returns the failure as a dict, so the run is driven through
    ``Runner`` here to read the classified exception itself.
    """
    with pytest.raises(ModelBehaviorError) as raised:
        await Runner.run(_agent_over(response), USER_INPUT)
    return raised.value
