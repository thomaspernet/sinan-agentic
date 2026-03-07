"""Tests for the turn budget system (core/turn_budget.py + turn_budget_tool.py)."""

from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from agents_core.core.turn_budget import TurnBudget, TurnBudgetHooks


# ------------------------------------------------------------------ #
# TurnBudget dataclass
# ------------------------------------------------------------------ #


class TestTurnBudgetDefaults:
    def test_default_values(self):
        budget = TurnBudget()
        assert budget.default_turns == 10
        assert budget.reminder_at == 2
        assert budget.max_extensions == 3
        assert budget.extension_size == 5
        assert budget.absolute_max == 25

    def test_initial_state(self):
        budget = TurnBudget()
        assert budget.turns_used == 0
        assert budget.extensions_used == 0
        assert budget.extension_reasons == []

    def test_custom_values(self):
        budget = TurnBudget(default_turns=5, reminder_at=1, max_extensions=2, extension_size=3, absolute_max=15)
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
        budget = TurnBudget(default_turns=10, reminder_at=2, max_extensions=1, extension_size=5, absolute_max=20)
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
# TurnBudgetHooks
# ------------------------------------------------------------------ #


class TestTurnBudgetHooks:
    @pytest.mark.asyncio
    async def test_on_llm_start_records_turn(self):
        budget = TurnBudget()
        hooks = TurnBudgetHooks(budget)
        await hooks.on_llm_start(Mock(), Mock(), None, [])
        assert budget.turns_used == 1

    @pytest.mark.asyncio
    async def test_multiple_llm_starts(self):
        budget = TurnBudget()
        hooks = TurnBudgetHooks(budget)
        for _ in range(5):
            await hooks.on_llm_start(Mock(), Mock(), None, [])
        assert budget.turns_used == 5


# ------------------------------------------------------------------ #
# InstructionBuilder integration
# ------------------------------------------------------------------ #


class TestInstructionBuilderTurnBudget:
    def test_no_budget_returns_none(self):
        from agents_core.instructions import InstructionBuilder
        builder = InstructionBuilder(None, None)
        assert builder.turn_budget_section() is None

    def test_budget_in_context_returns_section(self):
        from agents_core.instructions import InstructionBuilder

        budget = TurnBudget(default_turns=8)
        budget.turns_used = 6

        class FakeCtx:
            _turn_budget = budget

        builder = InstructionBuilder(FakeCtx(), None)
        section = builder.turn_budget_section()
        assert section is not None
        assert "2" in section and "8" in section

    def test_budget_section_in_build_output(self):
        from agents_core.instructions import InstructionBuilder

        budget = TurnBudget(default_turns=10)

        class FakeCtx:
            _turn_budget = budget

        class TestBuilder(InstructionBuilder):
            def persona(self):
                return "You are a test agent."

        result = TestBuilder(FakeCtx(), None).build()
        assert "You are a test agent." in result
        assert "10 turns" in result

    def test_budget_section_absent_without_budget(self):
        from agents_core.instructions import InstructionBuilder

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
        from agents_core.registry.agent_registry import AgentDefinition, AgentRegistry
        from agents_core.registry.tool_registry import ToolRegistry
        from agents_core.registry.guardrail_registry import GuardrailRegistry

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
        from agents_core.core.base_runner import BaseAgentRunner

        agent_reg, tool_reg, guardrail_reg = _registries
        with (
            patch("agents_core.core.base_runner.get_agent_registry", return_value=agent_reg),
            patch("agents_core.core.base_runner.get_tool_registry", return_value=tool_reg),
            patch("agents_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg),
        ):
            return BaseAgentRunner()

    def test_make_instructions_dynamic_from_static(self, runner):
        agent = Mock()
        agent.instructions = "Static instructions."
        budget = TurnBudget(default_turns=10)
        budget.turns_used = 8

        runner._make_instructions_dynamic(agent, budget)

        assert callable(agent.instructions)
        result = agent.instructions(Mock(), Mock())
        assert "Static instructions." in result
        assert "remaining" in result and "10" in result

    def test_make_instructions_dynamic_from_callable(self, runner):
        agent = Mock()
        agent.instructions = lambda ctx, a: "Dynamic base."
        budget = TurnBudget(default_turns=5)

        runner._make_instructions_dynamic(agent, budget)

        assert callable(agent.instructions)
        result = agent.instructions(Mock(), Mock())
        assert "Dynamic base." in result
        assert "5 turns" in result

    @pytest.mark.asyncio
    async def test_execute_basic_with_budget(self, runner):
        budget = TurnBudget(default_turns=10, absolute_max=25)
        context = Mock()
        session = Mock()

        mock_result = Mock()
        mock_result.final_output = "test output"

        with patch("agents_core.core.base_runner.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            with patch.object(runner, "create_agent", new_callable=AsyncMock) as mock_create:
                mock_agent = Mock()
                mock_agent.tools = []
                mock_agent.instructions = "Static."
                mock_create.return_value = mock_agent

                result = await runner._execute_basic(
                    "test_agent", context, session, 25, "hello",
                    turn_budget=budget,
                )

                assert result == "test output"
                # Verify hooks were passed
                call_kwargs = MockRunner.run.call_args[1]
                assert isinstance(call_kwargs["hooks"], TurnBudgetHooks)
                assert call_kwargs["max_turns"] == 25

    @pytest.mark.asyncio
    async def test_execute_sets_absolute_max(self, runner):
        budget = TurnBudget(default_turns=10, absolute_max=25)
        context = Mock()
        session = Mock()

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent", context, session,
                max_turns=10,
                input_text="hello",
                turn_budget=budget,
            )

            call_args = mock_basic.call_args
            # sdk_max_turns should be absolute_max
            assert call_args[0][3] == 25  # max_turns positional arg
            assert call_args[1]["turn_budget"] is budget

    @pytest.mark.asyncio
    async def test_execute_attaches_budget_to_context(self, runner):
        budget = TurnBudget(default_turns=10)
        context = Mock(spec=[])  # no existing attributes

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent", context, session=Mock(),
                input_text="hello",
                turn_budget=budget,
            )

            assert context._turn_budget is budget

    @pytest.mark.asyncio
    async def test_execute_resets_budget(self, runner):
        budget = TurnBudget(default_turns=10)
        budget.turns_used = 5
        budget.extensions_used = 1

        context = Mock(spec=[])

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent", context, session=Mock(),
                input_text="hello",
                turn_budget=budget,
            )

            assert budget.turns_used == 0
            assert budget.extensions_used == 0

    @pytest.mark.asyncio
    async def test_execute_without_budget_uses_max_turns(self, runner):
        context = Mock()
        session = Mock()

        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent", context, session,
                max_turns=15,
                input_text="hello",
            )

            call_args = mock_basic.call_args
            assert call_args[0][3] == 15  # uses original max_turns


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


class TestTopLevelImports:
    def test_turn_budget_importable(self):
        from agents_core import TurnBudget
        assert TurnBudget is not None

    def test_turn_budget_hooks_importable(self):
        from agents_core import TurnBudgetHooks
        assert TurnBudgetHooks is not None

    def test_turn_budget_from_core(self):
        from agents_core.core import TurnBudget, TurnBudgetHooks
        assert TurnBudget is not None
        assert TurnBudgetHooks is not None
