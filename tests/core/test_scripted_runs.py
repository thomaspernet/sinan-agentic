"""End-to-end runs of ``BaseAgentRunner`` against the SDK's scripted model.

Every other runner test replaces ``agents.Runner`` with a mock and asserts
against a result object the test built itself, so the seam this framework owns
— how the runner hands the SDK its agent, hooks, run config, turn ceiling and
error handlers — never executes. These tests keep the real ``Runner`` and swap
only the model, so a kwarg the SDK stopped accepting or a ``RunConfig`` field
that moved fails here instead of passing green.

``ScriptedModel`` records each call at the provider-neutral ``Model`` boundary,
which is what makes the assertions possible: the resolved ``ModelSettings``,
the advertised tools and the assembled input are read back from what the SDK
actually sent, not from a captured kwarg.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import (
    MaxTurnsExceeded,
    ModelBehaviorError,
    function_tool,
)
from agents.testing import UnconsumedModelSteps, assistant_message, function_call
from pydantic import BaseModel

from sinan_agentic_core.core.base_runner import BaseAgentRunner
from sinan_agentic_core.core.capabilities.steering import STEERING_ITEM_ROLE
from sinan_agentic_core.core.model_retry import ModelRetryConfig
from sinan_agentic_core.core.tool_error_recovery import ToolErrorRecovery
from sinan_agentic_core.core.turn_budget import TurnBudget
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from sinan_agentic_core.registry.guardrail_registry import GuardrailRegistry
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry
from sinan_agentic_core.session.agent_session import AgentSession
from tests.conftest import scripted_run

WEATHER_CALL_ID = "call-weather"
FAILING_CALL_ID = "call-failing"
TOOL_FAILURE = json.dumps({"error": "upstream is down"})


def _steering_text(call) -> str:
    """The steering item's text from a recorded model call, or "" when absent.

    A call the capabilities had nothing to say on carries no steering item, so a
    test asserting a fragment is *absent* reads an empty string rather than an
    item that is not there.
    """
    trailing = call.input[-1] if call.input else {}
    if trailing.get("role") != STEERING_ITEM_ROLE:
        return ""
    return str(trailing["content"])


class Extraction(BaseModel):
    """Structured output for the agents that declare one."""

    answer: str


@function_tool
def weather(city: str) -> str:
    """Report the weather in a city."""
    return f"sunny in {city}"


@function_tool
def failing_lookup(city: str) -> str:
    """Return the error payload the recovery capability tracks."""
    return TOOL_FAILURE


@pytest.fixture
def runner():
    """A runner over agents covering each seam these tests drive."""
    agent_reg = AgentRegistry()
    tool_reg = ToolRegistry()
    guardrail_reg = GuardrailRegistry()

    tool_reg.register(ToolDefinition(name="weather", function=weather, description="weather"))
    tool_reg.register(
        ToolDefinition(name="failing_lookup", function=failing_lookup, description="fails")
    )

    agent_reg.register(
        AgentDefinition(
            name="assistant",
            description="answers questions",
            instructions="You answer questions.",
            tools=["weather", "failing_lookup"],
        )
    )
    agent_reg.register(
        AgentDefinition(
            name="bounded_assistant",
            description="answers under a bound",
            instructions="You answer questions.",
            model_timeout=30.0,
            model_retry=ModelRetryConfig(max_retries=4),
        )
    )
    agent_reg.register(
        AgentDefinition(
            name="extractor",
            description="extracts structured data",
            instructions="You extract data.",
            output_dataclass=Extraction,
        )
    )
    agent_reg.register(
        AgentDefinition(
            name="strict_extractor",
            description="extracts structured data without recovery",
            instructions="You extract data.",
            output_dataclass=Extraction,
            invalid_output_recovery=False,
        )
    )

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        return BaseAgentRunner()


# ------------------------------------------------------------------ #
# execute() — basic branch
# ------------------------------------------------------------------ #


class TestBasicBranchRunsForReal:
    async def test_a_tool_calling_run_returns_the_models_final_text(self, runner, context, session):
        """Two model calls, one real tool execution, one real session."""
        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ):
            result = await runner.execute("assistant", context, session, input_text="weather?")

        assert result == "It is sunny in Paris."

    async def test_the_tool_result_reaches_the_next_model_call(self, runner, context, session):
        """The runner's tools run through the SDK, so their output lands in the input."""
        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ) as model:
            await runner.execute("assistant", context, session, input_text="weather?")

        outputs = [
            item["output"]
            for item in model.last_call.input
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        assert outputs == ["sunny in Paris"]

    async def test_the_declared_tools_are_advertised_to_the_model(self, runner, context, session):
        """``_build_tools`` output reaches the model boundary, not just an agent kwarg."""
        with scripted_run(runner, [assistant_message("done")]) as model:
            await runner.execute("assistant", context, session, input_text="hello")

        assert [tool.name for tool in model.first_call.tools] == ["weather", "failing_lookup"]

    async def test_usage_is_aggregated_from_the_real_run(self, runner, context, session):
        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ):
            await runner.execute("assistant", context, session, input_text="weather?")

        assert runner.last_usage["requests"] == 2

    async def test_a_run_that_stops_early_fails_instead_of_passing(self, runner, context, session):
        """``UnconsumedModelSteps`` is what makes a missing second call visible."""
        with pytest.raises(UnconsumedModelSteps):
            with scripted_run(
                runner,
                [assistant_message("done")],
                [assistant_message("never reached")],
            ):
                await runner.execute("assistant", context, session, input_text="hello")


# ------------------------------------------------------------------ #
# Declared model settings reaching the model boundary
# ------------------------------------------------------------------ #


class TestDeclaredModelSettingsReachTheModel:
    """``model_timeout`` and ``model_retry`` are only real if the model sees them."""

    async def test_the_declared_bound_reaches_the_model_call(self, runner, context, session):
        with scripted_run(runner, [assistant_message("done")]) as model:
            await runner.execute("bounded_assistant", context, session, input_text="hello")

        assert model.last_call.model_settings.timeout == 30.0

    async def test_the_declared_retry_policy_reaches_the_model_call(self, runner, context, session):
        with scripted_run(runner, [assistant_message("done")]) as model:
            await runner.execute("bounded_assistant", context, session, input_text="hello")

        assert model.last_call.model_settings.retry.max_retries == 4

    async def test_an_agent_declaring_neither_leaves_both_unset(self, runner, context, session):
        with scripted_run(runner, [assistant_message("done")]) as model:
            await runner.execute("assistant", context, session, input_text="hello")

        assert model.last_call.model_settings.timeout is None
        assert model.last_call.model_settings.retry is None


# ------------------------------------------------------------------ #
# Turn budget
# ------------------------------------------------------------------ #


class TestTurnBudgetOnARealRun:
    async def test_the_budget_counts_the_turns_the_sdk_actually_ran(self, runner, context, session):
        budget = TurnBudget(default_turns=5, absolute_max=5)

        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ):
            await runner.execute(
                "assistant", context, session, turn_budget=budget, input_text="weather?"
            )

        assert budget.turns_used == 2

    async def test_the_budget_section_reaches_the_next_model_call(self, runner, context, session):
        """The steering item is rebuilt per call, so the model sees the count."""
        budget = TurnBudget(default_turns=5, absolute_max=5)

        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ) as model:
            await runner.execute(
                "assistant", context, session, turn_budget=budget, input_text="weather?"
            )

        assert "remaining" in _steering_text(model.last_call)

    async def test_the_absolute_max_is_the_ceiling_the_sdk_enforces(self, runner, context, session):
        """The budget's hardest ceiling becomes the SDK's ``max_turns``."""
        budget = TurnBudget(default_turns=1, absolute_max=1)

        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
        ):
            with pytest.raises(MaxTurnsExceeded):
                await runner.execute(
                    "assistant",
                    context,
                    session,
                    turn_budget=budget,
                    max_turns=10,
                    input_text="weather?",
                )


# ------------------------------------------------------------------ #
# Tool error recovery
# ------------------------------------------------------------------ #


class TestToolErrorRecoveryOnARealRun:
    async def test_the_capability_records_an_error_the_sdk_reported(self, runner, context, session):
        """``on_tool_end`` fires through the SDK's own hook chain."""
        recovery = ToolErrorRecovery()

        with scripted_run(
            runner,
            [function_call("failing_lookup", {"city": "Paris"}, call_id=FAILING_CALL_ID)],
            [assistant_message("I could not reach the service.")],
        ):
            await runner.execute(
                "assistant", context, session, error_recovery=recovery, input_text="weather?"
            )

        assert recovery.has_errors

    async def test_the_recovery_section_reaches_the_next_model_call(self, runner, context, session):
        """The point of tracking the error is telling the model about it."""
        recovery = ToolErrorRecovery()

        with scripted_run(
            runner,
            [function_call("failing_lookup", {"city": "Paris"}, call_id=FAILING_CALL_ID)],
            [assistant_message("I could not reach the service.")],
        ) as model:
            await runner.execute(
                "assistant", context, session, error_recovery=recovery, input_text="weather?"
            )

        assert "Tool Error Recovery" in _steering_text(model.last_call)


# ------------------------------------------------------------------ #
# Prompt stability and steering placement
# ------------------------------------------------------------------ #


class TestThePromptStaysStableAcrossAModelCall:
    """The reason the fragments moved: a byte-identical prompt is cacheable.

    A provider reuses only the longest leading span a later request shares with
    an earlier one, and the system prompt is serialized first. These drive the
    real SDK over a run of several tool calls and read back what it actually
    sent, which is the only place the per-call difference would show.
    """

    @staticmethod
    async def _run_with_three_tool_calls(runner, context, session, **execute_kwargs):
        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=f"{WEATHER_CALL_ID}-1")],
            [function_call("failing_lookup", {"city": "Rome"}, call_id=FAILING_CALL_ID)],
            [function_call("weather", {"city": "Oslo"}, call_id=f"{WEATHER_CALL_ID}-2")],
            [assistant_message("It is sunny in Paris.")],
        ) as model:
            await runner.execute(
                "assistant", context, session, input_text="weather?", **execute_kwargs
            )
        return model

    async def test_every_call_of_one_run_sends_the_same_instructions(
        self, runner, context, session
    ):
        budget = TurnBudget(default_turns=9, absolute_max=9)
        recovery = ToolErrorRecovery()

        model = await self._run_with_three_tool_calls(
            runner, context, session, turn_budget=budget, error_recovery=recovery
        )

        sent = [call.system_instructions for call in model.calls]
        assert len(sent) == 4
        assert sent == [sent[0]] * len(sent)

    async def test_the_volatile_fragments_ride_in_the_input_instead(self, runner, context, session):
        """Both capabilities report changing state, and neither is in the prompt."""
        budget = TurnBudget(default_turns=9, absolute_max=9)
        recovery = ToolErrorRecovery()

        model = await self._run_with_three_tool_calls(
            runner, context, session, turn_budget=budget, error_recovery=recovery
        )

        assert "Turn budget" not in model.calls[0].system_instructions
        assert "Turn budget" in _steering_text(model.calls[0])
        # The recovery fragment only exists once a tool has failed, which is
        # exactly the kind of mid-run change the prompt must not carry.
        assert "Tool Error Recovery" not in _steering_text(model.calls[1])
        assert "Tool Error Recovery" in _steering_text(model.calls[2])
        assert "Tool Error Recovery" not in model.calls[2].system_instructions

    async def test_the_budget_count_advances_between_calls(self, runner, context, session):
        """A fragment that never changed would pass the stability test for free."""
        budget = TurnBudget(default_turns=9, absolute_max=9)

        model = await self._run_with_three_tool_calls(runner, context, session, turn_budget=budget)

        steering = [_steering_text(call) for call in model.calls]
        assert len(set(steering)) == len(steering)

    async def test_the_steering_item_is_last_and_never_persisted(self, runner, context, session):
        budget = TurnBudget(default_turns=9, absolute_max=9)

        model = await self._run_with_three_tool_calls(runner, context, session, turn_budget=budget)

        assert model.last_call.input[-1]["role"] == STEERING_ITEM_ROLE
        stored = await session.get_items()
        assert not any(item.get("role") == STEERING_ITEM_ROLE for item in stored)


class TestSteeringReachesEveryExecutionBranch:
    """The steering item is installed through the run config, which every
    ``Runner``-calling branch builds — so each branch is asserted, not assumed."""

    async def test_the_streamed_branch_steers(self, runner, context, session):
        budget = TurnBudget(default_turns=5, absolute_max=5)

        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ) as model:
            await runner.execute(
                "assistant",
                context,
                session,
                streaming=True,
                turn_budget=budget,
                input_text="weather?",
            )

        assert "Turn budget" in _steering_text(model.last_call)

    async def test_the_fallback_branch_steers(self, runner, context, session):
        """The branch that adds an overflow rescue still runs the SDK for the loop."""
        budget = TurnBudget(default_turns=5, absolute_max=5)

        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ) as model:
            await runner.execute(
                "assistant",
                context,
                session,
                fallback_on_overflow=True,
                turn_budget=budget,
                input_text="weather?",
            )

        assert "Turn budget" in _steering_text(model.last_call)


# ------------------------------------------------------------------ #
# Streaming branch
# ------------------------------------------------------------------ #


class TestStreamedBranchRunsForReal:
    @staticmethod
    async def _stream(runner, context, session):
        """Stream one tool-calling run; return its output and the events it emitted."""
        events = []
        with scripted_run(
            runner,
            [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            [assistant_message("It is sunny in Paris.")],
        ):
            result = await runner.execute(
                "assistant",
                context,
                session,
                streaming=True,
                on_event=events.append,
                input_text="weather?",
            )
        return result, events

    async def test_it_returns_the_models_final_text(self, runner, context, session):
        result, _ = await self._stream(runner, context, session)

        assert result == "It is sunny in Paris."

    async def test_the_events_come_from_real_sdk_stream_items(self, runner, context, session):
        _, events = await self._stream(runner, context, session)

        emitted = {event["event"] for event in events}
        assert {"tool_call", "tool_output", "message_output", "answer"} <= emitted

    async def test_the_tool_event_names_the_tool_the_sdk_ran(self, runner, context, session):
        _, events = await self._stream(runner, context, session)

        tool_calls = [e for e in events if e["event"] == "tool_call"]
        assert [e["data"]["tool"] for e in tool_calls] == ["weather"]

    async def test_the_answer_event_reports_the_runs_own_usage(self, runner, context, session):
        _, events = await self._stream(runner, context, session)

        answer = next(e for e in events if e["event"] == "answer")
        assert answer["data"]["usage"]["requests"] == 2


# ------------------------------------------------------------------ #
# Invalid structured output (error_handlers wiring)
# ------------------------------------------------------------------ #


class TestInvalidOutputRecoveryOnARealRun:
    """``error_handlers`` is a run kwarg, so only a real run proves the SDK calls it."""

    async def test_a_fenced_payload_is_salvaged(self, runner, context, session):
        with scripted_run(runner, [assistant_message('```json\n{"answer": "yes"}\n```')]):
            result = await runner.execute("extractor", context, session, input_text="extract")

        assert result.answer == "yes"

    async def test_an_agent_that_opted_out_still_fails(self, runner, context, session):
        with scripted_run(runner, [assistant_message('```json\n{"answer": "yes"}\n```')]):
            with pytest.raises(ModelBehaviorError):
                await runner.execute("strict_extractor", context, session, input_text="extract")


# ------------------------------------------------------------------ #
# Fallback branch
# ------------------------------------------------------------------ #


class TestFallbackBranchOnARealRun:
    async def test_a_real_max_turns_failure_triggers_the_rescue_call(
        self, runner, context, session
    ):
        """The overflow the branch rescues comes from the SDK, not from a mock."""
        completion = Mock(usage=None)
        completion.choices = [Mock()]
        completion.choices[0].message.content = "Rescued answer"

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=completion)
        client.with_options = Mock(return_value=client)

        with (
            scripted_run(
                runner,
                [function_call("weather", {"city": "Paris"}, call_id=WEATHER_CALL_ID)],
            ),
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client", return_value=client),
        ):
            result = await runner.execute(
                "assistant",
                context,
                session,
                fallback_on_overflow=True,
                max_turns=1,
                input_text="weather?",
            )

        assert result == "Rescued answer"


# ------------------------------------------------------------------ #
# run_agent (backward-compatible entry point)
# ------------------------------------------------------------------ #


class TestRunAgentRunsForReal:
    async def test_it_returns_the_output_and_the_runs_usage(self, runner, context):
        with scripted_run(runner, [assistant_message("Answered.")]):
            result = await runner.run_agent(
                "assistant", AgentSession(session_id="run-agent"), context, "hello"
            )

        assert result["output"] == "Answered."
        assert result["usage"]["requests"] == 1
