"""Tests that a retried model call leaves the committed session untouched.

A model retry re-issues the request that failed; it must not re-write the
conversation that request was built from. Before openai-agents 0.19.0 the
non-streamed retry rewound session items unconditionally, so a retry popped the
turn's prepared input — history already committed to a plain local session
included — and the run continued against a session missing earlier turns.

The shape the bug needs is what this framework wires: ``apply_model_retry``
overlays a declared policy onto the ``ModelSettings`` every agent-building path
assembles, and every run path passes a session into ``Runner.run``. The SDK
skips the rewind for a session without ``pop_item``, but both session objects
that reach a run here implement it — ``AgentSession`` and, on the recovery
branch, ``_CollectingSessionWrapper``. These tests drive the real SDK loop
against a model that fails once, so the retry is asserted where the two meet
rather than against a mocked ``Runner``.

The streamed path is not covered: its rewind only ever touched the server
conversation tracker, never the session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agents import Agent, ModelResponse, Runner, Usage
from agents.models.interface import Model
from openai import APIConnectionError
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from sinan_agentic_core.core.base_runner import _CollectingSessionWrapper
from sinan_agentic_core.core.model_retry import (
    ModelRetryConfig,
    RetryBackoffConfig,
    apply_model_retry,
)
from sinan_agentic_core.session.agent_session import AgentSession

# A turn committed to the session before the run under test. This is what an
# unconditional rewind pops, and it is not part of the failed request.
COMMITTED_INPUT = "the turn already in the session"

USER_INPUT = "the turn that fails once"

RECOVERED_OUTPUT = "the answer the second attempt produces"

ASSISTANT_ROLE = "assistant"

USER_ROLE = "user"


def _assistant_message(text: str) -> ResponseOutputMessage:
    """Build the single assistant message the recovered attempt answers with."""
    return ResponseOutputMessage(
        id="msg_recovered",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role=ASSISTANT_ROLE,
        status="completed",
        type="message",
    )


class FailsOnceModel(Model):
    """A model whose first request fails on the transport, and whose second answers.

    ``APIConnectionError`` is what the ``network_error`` trigger matches, so the
    declared policy schedules exactly one retry and the run completes.
    """

    def __init__(self) -> None:
        self.attempts = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Fail the first attempt, answer every later one."""
        self.attempts += 1
        if self.attempts == 1:
            raise APIConnectionError(request=None)
        return ModelResponse(
            output=[_assistant_message(RECOVERED_OUTPUT)],
            usage=Usage(),
            response_id="resp_recovered",
        )

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Unused — the streamed retry never rewound session items."""
        raise NotImplementedError


def _retrying_agent(model: Model) -> Agent[Any]:
    """An agent on the declared retry path, with the backoff wait taken out.

    The triggers and attempt budget are the declared defaults — the policy an
    ``agents.yaml`` entry gets from a bare ``model_retry:``. Only the delay
    schedule is overridden, so the retry is immediate instead of waiting out the
    SDK's default backoff.
    """
    retry = ModelRetryConfig(
        backoff=RetryBackoffConfig(initial_delay=0.0, max_delay=0.0, jitter=False)
    )
    return Agent(
        name="retrying",
        instructions="answer",
        model=model,
        model_settings=apply_model_retry(retry),
    )


def _contents(items: list[Any]) -> list[str]:
    """The content of every item in *items*, in session order."""
    return [item["content"] for item in items]


async def _session_with_committed_turn(session_id: str) -> AgentSession:
    """A session already holding one committed turn before the run starts."""
    session = AgentSession(session_id=session_id)
    await session.add_items([{"role": USER_ROLE, "content": COMMITTED_INPUT}])
    return session


class TestNonStreamedRetry:
    """``Runner.run`` — the path ``_execute_basic`` and ``run_agent`` take."""

    async def test_the_retry_happens(self) -> None:
        """Without this the retention assertions below would hold vacuously."""
        session = await _session_with_committed_turn("retry-taken")
        model = FailsOnceModel()

        result = await Runner.run(_retrying_agent(model), USER_INPUT, session=session)

        assert model.attempts == 2
        assert result.final_output == RECOVERED_OUTPUT

    async def test_a_retry_keeps_the_already_committed_turn(self) -> None:
        session = await _session_with_committed_turn("committed-retained")
        model = FailsOnceModel()

        await Runner.run(_retrying_agent(model), USER_INPUT, session=session)

        assert _contents(await session.get_items()) == [
            COMMITTED_INPUT,
            USER_INPUT,
            RECOVERED_OUTPUT,
        ]

    async def test_a_retry_does_not_duplicate_the_turn_it_re_sends(self) -> None:
        """The request is re-issued; the input it was built from is saved once."""
        session = await _session_with_committed_turn("input-not-duplicated")

        await Runner.run(_retrying_agent(FailsOnceModel()), USER_INPUT, session=session)

        assert _contents(await session.get_items()).count(USER_INPUT) == 1


class TestFallbackSessionWrapper:
    """The recovery branch runs against ``_CollectingSessionWrapper``, not the session.

    The wrapper forwards ``pop_item``, so the SDK's "no ``pop_item``, no rewind"
    guard does not spare it — the session behind it is reachable by a rewind.
    """

    async def test_a_retry_keeps_the_already_committed_turn(self) -> None:
        session = await _session_with_committed_turn("committed-retained-fallback")
        collecting = _CollectingSessionWrapper(session)
        model = FailsOnceModel()

        await Runner.run(_retrying_agent(model), USER_INPUT, session=collecting)

        assert model.attempts == 2
        assert _contents(await session.get_items()) == [
            COMMITTED_INPUT,
            USER_INPUT,
            RECOVERED_OUTPUT,
        ]
