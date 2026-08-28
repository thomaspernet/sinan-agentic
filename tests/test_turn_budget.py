"""Tests for the turn budget system (core/turn_budget.py + turn_budget_tool.py)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import RunContextWrapper
from pydantic import ValidationError

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.turn_budget import (
    DEFAULT_ABSOLUTE_MAX_TURNS,
    DEFAULT_EXTENSION_SIZE,
    DEFAULT_MAX_EXTENSIONS,
    DEFAULT_REMINDER_AT,
    DEFAULT_TURNS,
    TurnBudget,
    TurnBudgetConfig,
    build_turn_budget,
)
from sinan_agentic_core.utils import get_turn_budget, set_turn_budget

# ------------------------------------------------------------------ #
# TurnBudget dataclass
# ------------------------------------------------------------------ #


class TestTurnBudgetDefaults:
    def test_default_values(self):
        budget = TurnBudget()
        assert budget.default_turns == DEFAULT_TURNS
        assert budget.reminder_at == DEFAULT_REMINDER_AT
        assert budget.max_extensions == DEFAULT_MAX_EXTENSIONS
        assert budget.extension_size == DEFAULT_EXTENSION_SIZE
        assert budget.absolute_max == DEFAULT_ABSOLUTE_MAX_TURNS

    def test_initial_state(self):
        budget = TurnBudget()
        assert budget.turns_used == 0
        assert budget.extensions_used == 0
        assert budget.extension_reasons == []

    def test_custom_values(self):
        budget = TurnBudget(
            default_turns=5, reminder_at=1, max_extensions=2, extension_size=3, absolute_max=15
        )
        assert budget.default_turns == 5
        assert budget.absolute_max == 15


class TestTurnBudgetProperties:
    def test_effective_max_no_extensions(self):
        budget = TurnBudget(default_turns=10)
        assert budget.effective_max == 10

    def test_effective_max_with_extensions(self):
        budget = TurnBudget(default_turns=10, extension_size=5)
        budget.extensions_used = 2
        assert budget.effective_max == 20

    def test_remaining_full(self):
        budget = TurnBudget(default_turns=10)
        assert budget.remaining == 10

    def test_remaining_after_turns(self):
        budget = TurnBudget(default_turns=10)
        budget.turns_used = 7
        assert budget.remaining == 3

    def test_remaining_never_negative(self):
        budget = TurnBudget(default_turns=5)
        budget.turns_used = 10
        assert budget.remaining == 0

    def test_is_warning_true(self):
        budget = TurnBudget(default_turns=10, reminder_at=2)
        budget.turns_used = 8
        assert budget.is_warning is True

    def test_is_warning_false_plenty_left(self):
        budget = TurnBudget(default_turns=10, reminder_at=2)
        budget.turns_used = 5
        assert budget.is_warning is False

    def test_is_warning_false_when_exhausted(self):
        budget = TurnBudget(default_turns=10, reminder_at=2)
        budget.turns_used = 10
        assert budget.is_warning is False  # exhausted, not warning

    def test_is_exhausted(self):
        budget = TurnBudget(default_turns=5)
        budget.turns_used = 5
        assert budget.is_exhausted is True

    def test_is_exhausted_false(self):
        budget = TurnBudget(default_turns=5)
        budget.turns_used = 4
        assert budget.is_exhausted is False

    def test_can_extend_yes(self):
        budget = TurnBudget(default_turns=10, max_extensions=3, extension_size=5, absolute_max=25)
        assert budget.can_extend is True

    def test_can_extend_no_max_extensions(self):
        budget = TurnBudget(default_turns=10, max_extensions=2)
        budget.extensions_used = 2
        assert budget.can_extend is False

    def test_can_extend_no_would_exceed_absolute(self):
        budget = TurnBudget(default_turns=10, max_extensions=5, extension_size=5, absolute_max=12)
        budget.extensions_used = 0
        # effective_max=10, +5 would be 15, exceeds absolute_max=12
        assert budget.can_extend is False


class TestRequestExtension:
    def test_successful_extension(self):
        budget = TurnBudget(default_turns=10, max_extensions=3, extension_size=5, absolute_max=25)
        budget.turns_used = 10
        success, msg = budget.request_extension("Need to process more documents")
        assert success is True
        assert budget.extensions_used == 1
        assert budget.effective_max == 15
        assert budget.remaining == 5
        assert "approved" in msg.lower() or "approved" in msg

    def test_extension_denied_max_reached(self):
        budget = TurnBudget(default_turns=10, max_extensions=1, extension_size=5, absolute_max=25)
        budget.extensions_used = 1
        success, msg = budget.request_extension("Need more")
        assert success is False
        assert "maximum extensions" in msg.lower()

    def test_extension_denied_absolute_exceeded(self):
        budget = TurnBudget(default_turns=10, max_extensions=5, extension_size=5, absolute_max=12)
        success, msg = budget.request_extension("Need more")
        assert success is False
        assert "absolute maximum" in msg.lower()

    def test_extension_tracks_reason(self):
        budget = TurnBudget(default_turns=10, max_extensions=3, extension_size=5, absolute_max=25)
        budget.request_extension("Reason A")
        budget.request_extension("Reason B")
        assert budget.extension_reasons == ["Reason A", "Reason B"]

    def test_multiple_extensions(self):
        budget = TurnBudget(default_turns=5, max_extensions=3, extension_size=3, absolute_max=20)
        budget.request_extension("first")
        assert budget.effective_max == 8
        budget.request_extension("second")
        assert budget.effective_max == 11
        budget.request_extension("third")
        assert budget.effective_max == 14
        success, _ = budget.request_extension("fourth")
        assert success is False
        assert budget.effective_max == 14


class TestRecordTurn:
    def test_increments_counter(self):
        budget = TurnBudget()
        budget.record_turn()
        assert budget.turns_used == 1
        budget.record_turn()
        assert budget.turns_used == 2


class TestReset:
    def test_resets_state(self):
        budget = TurnBudget()
        budget.turns_used = 5
        budget.extensions_used = 2
        budget.extension_reasons.extend(["a", "b"])
        budget.reset()
        assert budget.turns_used == 0
        assert budget.extensions_used == 0
        assert budget.extension_reasons == []


class TestBuildInstructionSection:
    def test_initial_shows_budget(self):
        budget = TurnBudget(default_turns=10)
        section = budget.build_instruction_section()
        assert "10 turns" in section
        assert "Plan your work" in section

    def test_normal_shows_remaining(self):
        budget = TurnBudget(default_turns=10, reminder_at=2)
        budget.turns_used = 5
        section = budget.build_instruction_section()
        assert "5 of 10" in section

    def test_warning_with_extension_available(self):
        budget = TurnBudget(
            default_turns=10, reminder_at=2, max_extensions=1, extension_size=5, absolute_max=20
        )
        budget.turns_used = 9
        section = budget.build_instruction_section()
        assert "1 turn(s) remaining" in section
        assert "request_extension" in section

    def test_warning_without_extension(self):
        budget = TurnBudget(default_turns=10, reminder_at=2, max_extensions=0, absolute_max=10)
        budget.turns_used = 9
        section = budget.build_instruction_section()
        assert "1 turn(s) remaining" in section
        assert "Wrap up now" in section

    def test_exhausted_with_extension_available(self):
        budget = TurnBudget(default_turns=5, max_extensions=1, extension_size=5, absolute_max=15)
        budget.turns_used = 5
        section = budget.build_instruction_section()
        assert "EXHAUSTED" in section
        assert "request_extension" in section

    def test_exhausted_no_extensions(self):
        budget = TurnBudget(default_turns=5, max_extensions=0, absolute_max=5)
        budget.turns_used = 5
        section = budget.build_instruction_section()
        assert "EXHAUSTED" in section
        assert "Wrap up NOW" in section


# ------------------------------------------------------------------ #
# TurnBudget — on_llm_start capability hook
# ------------------------------------------------------------------ #


class TestTurnBudgetOnLlmStart:
    def test_on_llm_start_records_turn(self):
        budget = TurnBudget()
        budget.on_llm_start(Mock(), Mock(), None, [])
        assert budget.turns_used == 1

    def test_multiple_llm_starts(self):
        budget = TurnBudget()
        for _ in range(5):
            budget.on_llm_start(Mock(), Mock(), None, [])
        assert budget.turns_used == 5

    def test_on_llm_start_emits_event_when_streaming(self):
        budget = TurnBudget(default_turns=10)
        events: list[dict] = []
        budget.on_event = events.append
        budget.on_llm_start(Mock(), Mock(), None, [])
        assert len(events) == 1
        assert events[0]["event"] == "turn_budget"
        assert events[0]["data"]["turns_used"] == 1


# ------------------------------------------------------------------ #
# Context accessor
# ------------------------------------------------------------------ #


class TestTurnBudgetContextAccessor:
    def test_set_then_get_round_trip(self):
        budget = TurnBudget(default_turns=7)
        context = SimpleNamespace()

        set_turn_budget(context, budget)

        assert get_turn_budget(context) is budget

    def test_get_returns_none_when_unset(self):
        assert get_turn_budget(SimpleNamespace()) is None

    def test_get_returns_none_on_none_context(self):
        assert get_turn_budget(None) is None

    def test_set_overwrites_previous_budget(self):
        first = TurnBudget(default_turns=3)
        second = TurnBudget(default_turns=9)
        context = SimpleNamespace()

        set_turn_budget(context, first)
        set_turn_budget(context, second)

        assert get_turn_budget(context) is second

    def test_stored_under_public_attribute(self):
        budget = TurnBudget()
        context = SimpleNamespace()

        set_turn_budget(context, budget)

        assert context.turn_budget is budget
        assert not hasattr(context, "_turn_budget")


# ------------------------------------------------------------------ #
# InstructionBuilder integration
# ------------------------------------------------------------------ #


class TestInstructionBuilderTurnBudget:
    def test_no_budget_returns_none(self):
        from sinan_agentic_core.instructions import InstructionBuilder

        builder = InstructionBuilder(None, None)
        assert builder.turn_budget_section() is None

    def test_budget_in_context_returns_section(self):
        from sinan_agentic_core.instructions import InstructionBuilder

        budget = TurnBudget(default_turns=8)
        budget.turns_used = 6

        context = SimpleNamespace()
        set_turn_budget(context, budget)

        builder = InstructionBuilder(context, None)
        section = builder.turn_budget_section()
        assert section is not None
        assert "2" in section and "8" in section

    def test_budget_section_in_build_output(self):
        from sinan_agentic_core.instructions import InstructionBuilder

        budget = TurnBudget(default_turns=10)

        context = SimpleNamespace()
        set_turn_budget(context, budget)

        class TestBuilder(InstructionBuilder):
            def persona(self):
                return "You are a test agent."

        result = TestBuilder(context, None).build()
        assert "You are a test agent." in result
        assert "10 turns" in result

    def test_budget_section_absent_without_budget(self):
        from sinan_agentic_core.instructions import InstructionBuilder

        class TestBuilder(InstructionBuilder):
            def persona(self):
                return "You are a test agent."

        result = TestBuilder(None, None).build()
        assert result == "You are a test agent."
        assert "turn" not in result.lower()


# ------------------------------------------------------------------ #
# BaseAgentRunner integration
# ------------------------------------------------------------------ #


class TestBaseAgentRunnerTurnBudget:
    @pytest.fixture
    def _registries(self):
        from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
        from sinan_agentic_core.registry.guardrail_registry import GuardrailRegistry
        from sinan_agentic_core.registry.tool_registry import ToolRegistry

        agent_reg = AgentRegistry()
        tool_reg = ToolRegistry()
        guardrail_reg = GuardrailRegistry()

        agent_reg.register(
            AgentDefinition(
                name="test_agent",
                description="test",
                instructions="You are a test agent.",
                tools=[],
            )
        )
        return agent_reg, tool_reg, guardrail_reg

    @pytest.fixture
    def runner(self, _registries):
        from sinan_agentic_core.core.base_runner import BaseAgentRunner

        agent_reg, tool_reg, guardrail_reg = _registries
        with (
            patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
            patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
            patch(
                "sinan_agentic_core.core.base_runner.get_guardrail_registry",
                return_value=guardrail_reg,
            ),
        ):
            return BaseAgentRunner()

    def test_apply_dynamic_instructions_from_static(self, runner):
        agent = Mock()
        agent.instructions = "Static instructions."
        budget = TurnBudget(default_turns=10)
        budget.turns_used = 8

        runner._apply_dynamic_instructions(agent, [budget])

        assert callable(agent.instructions)
        result = agent.instructions(Mock(), Mock())
        assert "Static instructions." in result
        assert "remaining" in result and "10" in result

    def test_apply_dynamic_instructions_from_callable(self, runner):
        agent = Mock()
        agent.instructions = lambda ctx, a: "Dynamic base."
        budget = TurnBudget(default_turns=5)

        runner._apply_dynamic_instructions(agent, [budget])

        assert callable(agent.instructions)
        result = agent.instructions(Mock(), Mock())
        assert "Dynamic base." in result
        assert "5 turns" in result

    async def test_execute_basic_with_budget(self, runner):
        from sinan_agentic_core.core.base_runner import _CompositeHooks

        budget = TurnBudget(default_turns=10, absolute_max=25)
        context = Mock()
        session = Mock()

        mock_result = Mock()
        mock_result.final_output = "test output"
        mock_result.new_items = []
        mock_result.raw_responses = []

        with patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls:
            mock_runner_cls.run = AsyncMock(return_value=mock_result)
            with patch.object(runner, "create_agent", new_callable=AsyncMock) as mock_create:
                mock_agent = Mock()
                mock_agent.tools = []
                mock_agent.instructions = "Static."
                mock_create.return_value = mock_agent

                result = await runner._execute_basic(
                    "test_agent",
                    context,
                    session,
                    25,
                    "hello",
                    capabilities=[budget],
                )

                assert result == "test output"
                # Verify hooks were passed
                call_kwargs = mock_runner_cls.run.call_args[1]
                assert isinstance(call_kwargs["hooks"], _CompositeHooks)
                assert call_kwargs["max_turns"] == 25

    async def test_execute_sets_absolute_max(self, runner):
        budget = TurnBudget(default_turns=10, absolute_max=25)
        context = Mock()
        session = Mock()

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent",
                context,
                session,
                max_turns=10,
                input_text="hello",
                turn_budget=budget,
            )

            call_args = mock_basic.call_args
            # sdk_max_turns should be absolute_max
            assert call_args[0][3] == 25  # max_turns positional arg
            assert budget in call_args[1]["capabilities"]

    async def test_execute_attaches_budget_to_context(self, runner):
        budget = TurnBudget(default_turns=10)
        context = Mock(spec=[])  # no existing attributes

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent",
                context,
                session=Mock(),
                input_text="hello",
                turn_budget=budget,
            )

            assert get_turn_budget(context) is budget

    async def test_execute_resets_budget(self, runner):
        budget = TurnBudget(default_turns=10)
        budget.turns_used = 5
        budget.extensions_used = 1

        context = Mock(spec=[])

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent",
                context,
                session=Mock(),
                input_text="hello",
                turn_budget=budget,
            )

            assert budget.turns_used == 0
            assert budget.extensions_used == 0

    async def test_execute_without_budget_uses_max_turns(self, runner):
        context = Mock()
        session = Mock()

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent",
                context,
                session,
                max_turns=15,
                input_text="hello",
            )

            call_args = mock_basic.call_args
            assert call_args[0][3] == 15  # uses original max_turns


# ------------------------------------------------------------------ #
# Capability protocol adoption
# ------------------------------------------------------------------ #


class TestTurnBudgetIsCapability:
    def test_is_capability_subclass(self):
        assert issubclass(TurnBudget, Capability)

    def test_instance_is_capability(self):
        assert isinstance(TurnBudget(), Capability)

    def test_instructions_returns_initial_section(self):
        budget = TurnBudget(default_turns=10)
        ctx = RunContextWrapper(context=None)
        section = budget.instructions(ctx)
        assert section is not None
        assert "10 turns" in section

    def test_instructions_reflects_state(self):
        budget = TurnBudget(default_turns=10, reminder_at=2)
        budget.turns_used = 5
        ctx = RunContextWrapper(context=None)
        section = budget.instructions(ctx)
        assert section is not None
        assert "5 of 10" in section

    def test_instructions_matches_build_instruction_section(self):
        budget = TurnBudget(default_turns=10, reminder_at=2)
        budget.turns_used = 9
        ctx = RunContextWrapper(context=None)
        assert budget.instructions(ctx) == budget.build_instruction_section()

    def test_tools_returns_request_extension(self):
        from sinan_agentic_core.core.turn_budget_tool import request_extension_tool

        assert TurnBudget().tools() == [request_extension_tool]


class TestTurnBudgetClone:
    def test_clone_returns_turn_budget(self):
        clone = TurnBudget().clone()
        assert isinstance(clone, TurnBudget)

    def test_clone_is_independent_instance(self):
        original = TurnBudget(default_turns=10)
        clone = original.clone()
        assert clone is not original

    def test_clone_preserves_configuration(self):
        original = TurnBudget(
            default_turns=8,
            reminder_at=1,
            max_extensions=4,
            extension_size=3,
            absolute_max=20,
        )
        clone = original.clone()
        assert clone.default_turns == 8
        assert clone.reminder_at == 1
        assert clone.max_extensions == 4
        assert clone.extension_size == 3
        assert clone.absolute_max == 20

    def test_clone_zeroes_counters(self):
        original = TurnBudget(default_turns=10)
        original.turns_used = 7
        original.extensions_used = 2
        original.extension_reasons.extend(["a", "b"])

        clone = original.clone()
        assert clone.turns_used == 0
        assert clone.extensions_used == 0
        assert clone.extension_reasons == []

    def test_clone_does_not_mutate_original(self):
        original = TurnBudget(default_turns=10)
        original.turns_used = 4
        original.extensions_used = 1
        original.extension_reasons.append("keep me")

        clone = original.clone()
        clone.record_turn()
        clone.request_extension("clone-only reason")

        assert original.turns_used == 4
        assert original.extensions_used == 1
        assert original.extension_reasons == ["keep me"]

    def test_clone_does_not_share_extension_reasons(self):
        original = TurnBudget(default_turns=10)
        original.extension_reasons.append("seed")
        clone = original.clone()
        assert clone.extension_reasons is not original.extension_reasons


# ------------------------------------------------------------------ #
# Snapshot / rehydrate
# ------------------------------------------------------------------ #


class TestTurnBudgetSnapshot:
    def test_snapshot_serializes_counters(self):
        budget = TurnBudget(default_turns=10)
        budget.record_turn()
        budget.record_turn()
        budget.request_extension("need more")

        snap = budget.to_snapshot()
        assert snap == {
            "turns_used": 2,
            "extensions_used": 1,
            "extension_reasons": ["need more"],
        }

    def test_round_trip_preserves_remaining_budget(self):
        # AC: configure to 10 turns, advance to 4, snapshot, rehydrate, expect 6 remaining.
        original = TurnBudget(default_turns=10)
        for _ in range(4):
            original.record_turn()
        snap = original.to_snapshot()

        resumed = TurnBudget(default_turns=10)
        resumed.from_snapshot(snap)
        assert resumed.turns_used == 4
        assert resumed.remaining == 6
        assert resumed.effective_max == 10

    def test_round_trip_preserves_extensions(self):
        original = TurnBudget(default_turns=5, extension_size=3, max_extensions=2)
        original.request_extension("first")
        original.record_turn()

        resumed = TurnBudget(default_turns=5, extension_size=3, max_extensions=2)
        resumed.from_snapshot(original.to_snapshot())

        assert resumed.extensions_used == 1
        assert resumed.effective_max == 8
        assert resumed.remaining == 7
        assert resumed.extension_reasons == ["first"]

    def test_from_snapshot_tolerates_missing_keys(self):
        budget = TurnBudget()
        budget.from_snapshot({})  # must not raise
        assert budget.turns_used == 0
        assert budget.extensions_used == 0
        assert budget.extension_reasons == []

    def test_snapshot_is_json_serializable(self):
        import json

        budget = TurnBudget()
        budget.record_turn()
        budget.request_extension("why")
        json.dumps(budget.to_snapshot())  # must not raise


# ------------------------------------------------------------------ #
# TurnBudgetConfig — the declared form
# ------------------------------------------------------------------ #


class TestTurnBudgetConfigDefaults:
    def test_an_empty_declaration_opts_in_with_defaults(self):
        """``turn_budget: {}`` is a declaration, not an absence."""
        config = TurnBudgetConfig()
        assert config.default_turns == DEFAULT_TURNS
        assert config.reminder_at == DEFAULT_REMINDER_AT
        assert config.max_extensions == DEFAULT_MAX_EXTENSIONS
        assert config.extension_size == DEFAULT_EXTENSION_SIZE

    def test_the_declared_defaults_match_the_runtime_defaults(self):
        """Declaring nothing and declaring an empty block must yield one budget.

        Both field lists read the same module constants, so a default changed on
        one side cannot leave the other behind.
        """
        runtime = TurnBudget()

        for name, field in TurnBudgetConfig.model_fields.items():
            assert field.default == getattr(runtime, name), name

    def test_an_unknown_key_is_rejected(self):
        """A typo must fail loudly rather than silently leave the field at its default."""
        with pytest.raises(ValidationError, match="typo"):
            TurnBudgetConfig(typo=5)

    def test_the_hard_ceiling_is_not_a_budget_field(self):
        """``absolute_max`` is the agent's ``max_turns``, not part of the declaration."""
        with pytest.raises(ValidationError, match="absolute_max"):
            TurnBudgetConfig(absolute_max=40)


class TestTurnBudgetConfigBuild:
    """The translation both declaration paths share."""

    def test_a_declared_config_becomes_a_budget(self):
        budget = TurnBudgetConfig(
            default_turns=15, reminder_at=3, max_extensions=2, extension_size=4
        ).build(absolute_max=30)

        assert budget.default_turns == 15
        assert budget.reminder_at == 3
        assert budget.max_extensions == 2
        assert budget.extension_size == 4
        assert budget.absolute_max == 30

    def test_no_ceiling_declared_falls_back_to_the_default(self):
        budget = TurnBudgetConfig(default_turns=8).build()

        assert budget.absolute_max == DEFAULT_ABSOLUTE_MAX_TURNS

    def test_every_declared_field_reaches_the_budget(self):
        """A field added to the config must land on the budget without editing build()."""
        declared = {name: 7 for name in TurnBudgetConfig.model_fields}

        budget = TurnBudgetConfig(**declared).build()

        for name, value in declared.items():
            assert getattr(budget, name) == value

    def test_each_call_returns_a_fresh_budget(self):
        """The budget carries mutable counters, so no two agents may share one."""
        config = TurnBudgetConfig(default_turns=8)

        assert config.build() is not config.build()


class TestBuildTurnBudget:
    """The translator carrying the "off unless declared" rule."""

    def test_a_declared_config_becomes_a_budget(self):
        budget = build_turn_budget(TurnBudgetConfig(default_turns=15), 30)

        assert budget is not None
        assert budget.default_turns == 15
        assert budget.absolute_max == 30

    def test_no_declaration_means_no_budget(self):
        """A budget rewrites instructions and adds a tool, so it is never implied."""
        assert build_turn_budget(None) is None

    def test_no_ceiling_declared_falls_back_to_the_default(self):
        budget = build_turn_budget(TurnBudgetConfig(default_turns=8))

        assert budget is not None
        assert budget.absolute_max == DEFAULT_ABSOLUTE_MAX_TURNS


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


class TestTopLevelImports:
    def test_turn_budget_importable(self):
        from sinan_agentic_core import TurnBudget

        assert TurnBudget is not None

    def test_turn_budget_from_core(self):
        from sinan_agentic_core.core import TurnBudget

        assert TurnBudget is not None

    def test_turn_budget_config_importable(self):
        from sinan_agentic_core import TurnBudgetConfig

        assert TurnBudgetConfig is not None

    def test_turn_budget_config_from_core(self):
        from sinan_agentic_core.core import TurnBudgetConfig

        assert TurnBudgetConfig is not None
