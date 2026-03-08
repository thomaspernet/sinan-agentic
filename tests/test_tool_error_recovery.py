"""Tests for the tool error recovery system (core/tool_error_recovery.py)."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents_core.core.tool_error_recovery import (
    ToolErrorEntry,
    ToolErrorRecovery,
    ToolErrorRecoveryHooks,
)


# ------------------------------------------------------------------ #
# ToolErrorRecovery — basic tracking
# ------------------------------------------------------------------ #


class TestRecordToolResult:
    def test_success_clears_tracked_error(self):
        recovery = ToolErrorRecovery()
        # Record an error first
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        assert recovery.has_errors
        # Then a success
        recovery.record_tool_result("my_tool", json.dumps({"result": "ok"}))
        assert not recovery.has_errors

    def test_error_is_tracked(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps({"error": "something broke"}))
        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert "my_tool" in summary
        assert summary["my_tool"]["error"] == "something broke"
        assert summary["my_tool"]["call_count"] == 1

    def test_non_json_result_ignored(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", "not json at all")
        assert not recovery.has_errors

    def test_non_dict_json_ignored(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps([1, 2, 3]))
        assert not recovery.has_errors

    def test_json_without_error_key_is_success(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps({"data": "ok"}))
        assert not recovery.has_errors

    def test_multiple_errors_same_tool(self):
        recovery = ToolErrorRecovery()
        args = json.dumps({"mode": "search", "query": "test"})
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail1"}), args)
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail2"}), args)
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["call_count"] == 2
        assert summary["my_tool"]["identical_count"] == 2

    def test_different_args_resets_identical_count(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool", json.dumps({"error": "fail"}), json.dumps({"a": 1})
        )
        recovery.record_tool_result(
            "my_tool", json.dumps({"error": "fail"}), json.dumps({"a": 2})
        )
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["call_count"] == 2
        assert summary["my_tool"]["identical_count"] == 1  # reset because args differ

    def test_multiple_tools_tracked_independently(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("tool_a", json.dumps({"error": "a error"}))
        recovery.record_tool_result("tool_b", json.dumps({"error": "b error"}))
        summary = recovery.get_error_summary()
        assert "tool_a" in summary
        assert "tool_b" in summary


# ------------------------------------------------------------------ #
# ToolErrorRecovery — instruction generation
# ------------------------------------------------------------------ #


class TestBuildInstructionSection:
    def test_no_errors_returns_empty(self):
        recovery = ToolErrorRecovery()
        assert recovery.build_instruction_section() == ""

    def test_first_error_shows_hint(self):
        registry = Mock()
        tool_def = Mock()
        tool_def.recovery_hint = "Try using mode='search' instead."
        registry.get_tool.return_value = tool_def

        recovery = ToolErrorRecovery(tool_registry=registry)
        recovery.record_tool_result("my_tool", json.dumps({"error": "No IDs provided"}))

        section = recovery.build_instruction_section()
        assert "Tool Error Recovery" in section
        assert "my_tool" in section
        assert "No IDs provided" in section
        assert "Try using mode='search' instead." in section

    def test_first_error_without_hint(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        section = recovery.build_instruction_section()
        assert "my_tool" in section
        assert "fail" in section
        assert "Recovery hint" not in section

    def test_repeated_error_escalates(self):
        recovery = ToolErrorRecovery()
        args = json.dumps({"mode": "check"})
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}), args)
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}), args)
        section = recovery.build_instruction_section()
        assert "FAILED 2 TIMES" in section
        assert "identical arguments" in section

    def test_stop_after_max_identical(self):
        recovery = ToolErrorRecovery(max_identical_before_stop=3)
        args = json.dumps({"mode": "check"})
        for _ in range(3):
            recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}), args)
        section = recovery.build_instruction_section()
        assert "STOP" in section
        assert "Do NOT call this tool again" in section

    def test_general_rule_always_present(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        section = recovery.build_instruction_section()
        assert "Never retry a tool call with identical parameters" in section

    def test_args_summary_shown(self):
        recovery = ToolErrorRecovery()
        args = json.dumps({"mode": "search", "query": "test"})
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}), args)
        section = recovery.build_instruction_section()
        assert "mode=" in section
        assert "query=" in section

    def test_empty_args_shown_as_empty(self):
        recovery = ToolErrorRecovery()
        args = json.dumps({"mode": "check", "ids": ""})
        recovery.record_tool_result("my_tool", json.dumps({"error": "No IDs"}), args)
        section = recovery.build_instruction_section()
        assert "ids=<empty>" in section


# ------------------------------------------------------------------ #
# ToolErrorRecovery — properties and reset
# ------------------------------------------------------------------ #


class TestProperties:
    def test_has_errors_false_initially(self):
        assert not ToolErrorRecovery().has_errors

    def test_has_critical_errors(self):
        recovery = ToolErrorRecovery(max_identical_before_stop=2)
        args = json.dumps({"a": 1})
        recovery.record_tool_result("t", json.dumps({"error": "x"}), args)
        assert not recovery.has_critical_errors
        recovery.record_tool_result("t", json.dumps({"error": "x"}), args)
        assert recovery.has_critical_errors

    def test_reset_clears_everything(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "x"}))
        recovery.reset()
        assert not recovery.has_errors
        assert recovery.build_instruction_section() == ""


# ------------------------------------------------------------------ #
# ToolErrorRecovery — hint lookup
# ------------------------------------------------------------------ #


class TestHintLookup:
    def test_hint_from_registry(self):
        registry = Mock()
        tool_def = Mock()
        tool_def.recovery_hint = "Use mode='resolve' first."
        registry.get_tool.return_value = tool_def

        recovery = ToolErrorRecovery(tool_registry=registry)
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        section = recovery.build_instruction_section()
        assert "Use mode='resolve' first." in section

    def test_hint_from_mcp_config(self):
        recovery = ToolErrorRecovery(
            mcp_hints={"mcp_search": "Query must be at least 2 characters."}
        )
        recovery.record_tool_result("mcp_search", json.dumps({"error": "too short"}))
        section = recovery.build_instruction_section()
        assert "Query must be at least 2 characters." in section

    def test_registry_hint_takes_priority_over_mcp(self):
        registry = Mock()
        tool_def = Mock()
        tool_def.recovery_hint = "registry hint"
        registry.get_tool.return_value = tool_def

        recovery = ToolErrorRecovery(
            tool_registry=registry,
            mcp_hints={"my_tool": "mcp hint"},
        )
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        section = recovery.build_instruction_section()
        assert "registry hint" in section
        assert "mcp hint" not in section

    def test_no_registry_uses_mcp(self):
        recovery = ToolErrorRecovery(mcp_hints={"ext_tool": "external hint"})
        recovery.record_tool_result("ext_tool", json.dumps({"error": "fail"}))
        section = recovery.build_instruction_section()
        assert "external hint" in section


# ------------------------------------------------------------------ #
# ToolErrorRecovery — args hashing
# ------------------------------------------------------------------ #


class TestArgsHashing:
    def test_same_args_different_key_order(self):
        recovery = ToolErrorRecovery()
        args1 = json.dumps({"b": 2, "a": 1})
        args2 = json.dumps({"a": 1, "b": 2})
        recovery.record_tool_result("t", json.dumps({"error": "x"}), args1)
        recovery.record_tool_result("t", json.dumps({"error": "x"}), args2)
        # Should be detected as identical (normalized JSON)
        summary = recovery.get_error_summary()
        assert summary["t"]["identical_count"] == 2

    def test_empty_args_hash_stable(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "x"}), "")
        recovery.record_tool_result("t", json.dumps({"error": "x"}), "")
        summary = recovery.get_error_summary()
        assert summary["t"]["identical_count"] == 2


# ------------------------------------------------------------------ #
# ToolErrorRecoveryHooks
# ------------------------------------------------------------------ #


class TestToolErrorRecoveryHooks:
    @pytest.mark.asyncio
    async def test_on_tool_end_records_result(self):
        recovery = ToolErrorRecovery()
        hooks = ToolErrorRecoveryHooks(recovery)

        tool = Mock()
        tool.name = "my_tool"

        # Simulate on_tool_start to capture args
        ctx = Mock()
        ctx.tool_arguments = json.dumps({"mode": "check"})
        await hooks.on_tool_start(ctx, Mock(), tool)

        # Simulate on_tool_end with error
        await hooks.on_tool_end(ctx, Mock(), tool, json.dumps({"error": "fail"}))

        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["error"] == "fail"

    @pytest.mark.asyncio
    async def test_on_tool_end_success_clears_error(self):
        recovery = ToolErrorRecovery()
        hooks = ToolErrorRecoveryHooks(recovery)

        tool = Mock()
        tool.name = "my_tool"
        ctx = Mock()
        ctx.tool_arguments = "{}"

        # Error first
        await hooks.on_tool_start(ctx, Mock(), tool)
        await hooks.on_tool_end(ctx, Mock(), tool, json.dumps({"error": "fail"}))
        assert recovery.has_errors

        # Then success
        await hooks.on_tool_start(ctx, Mock(), tool)
        await hooks.on_tool_end(ctx, Mock(), tool, json.dumps({"result": "ok"}))
        assert not recovery.has_errors

    @pytest.mark.asyncio
    async def test_on_event_emitted_on_error(self):
        recovery = ToolErrorRecovery()
        events = []
        hooks = ToolErrorRecoveryHooks(recovery, on_event=events.append)

        tool = Mock()
        tool.name = "my_tool"
        ctx = Mock()
        ctx.tool_arguments = "{}"

        await hooks.on_tool_start(ctx, Mock(), tool)
        await hooks.on_tool_end(ctx, Mock(), tool, json.dumps({"error": "fail"}))

        assert len(events) == 1
        assert events[0]["event"] == "tool_error_recovery"

    @pytest.mark.asyncio
    async def test_no_event_on_success(self):
        recovery = ToolErrorRecovery()
        events = []
        hooks = ToolErrorRecoveryHooks(recovery, on_event=events.append)

        tool = Mock()
        tool.name = "my_tool"
        ctx = Mock()
        ctx.tool_arguments = "{}"

        await hooks.on_tool_start(ctx, Mock(), tool)
        await hooks.on_tool_end(ctx, Mock(), tool, json.dumps({"data": "ok"}))

        assert len(events) == 0


# ------------------------------------------------------------------ #
# BaseAgentRunner integration
# ------------------------------------------------------------------ #


class TestBaseAgentRunnerIntegration:
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

    def test_apply_dynamic_instructions_with_recovery(self, runner):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "fail"}))

        agent = Mock()
        agent.instructions = "Base instructions."
        runner._apply_dynamic_instructions(agent, recovery=recovery)

        assert callable(agent.instructions)
        result = agent.instructions(Mock(), Mock())
        assert "Base instructions." in result
        assert "Tool Error Recovery" in result

    def test_apply_dynamic_instructions_no_features(self, runner):
        agent = Mock()
        agent.instructions = "Static."
        runner._apply_dynamic_instructions(agent)
        # Should NOT replace with callable
        assert agent.instructions == "Static."

    def test_apply_dynamic_instructions_both_budget_and_recovery(self, runner):
        from agents_core.core.turn_budget import TurnBudget

        budget = TurnBudget(default_turns=10)
        budget.turns_used = 8
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "fail"}))

        agent = Mock()
        agent.instructions = "Base."
        runner._apply_dynamic_instructions(agent, budget=budget, recovery=recovery)

        result = agent.instructions(Mock(), Mock())
        assert "Base." in result
        assert "remaining" in result  # budget section
        assert "Tool Error Recovery" in result  # recovery section

    def test_build_hooks_none_when_no_features(self):
        from agents_core.core.base_runner import BaseAgentRunner
        assert BaseAgentRunner._build_hooks() is None

    def test_build_hooks_single_budget(self):
        from agents_core.core.base_runner import BaseAgentRunner
        from agents_core.core.turn_budget import TurnBudget, TurnBudgetHooks

        hooks = BaseAgentRunner._build_hooks(budget=TurnBudget())
        assert isinstance(hooks, TurnBudgetHooks)

    def test_build_hooks_single_recovery(self):
        from agents_core.core.base_runner import BaseAgentRunner

        hooks = BaseAgentRunner._build_hooks(recovery=ToolErrorRecovery())
        assert isinstance(hooks, ToolErrorRecoveryHooks)

    def test_build_hooks_composite_when_both(self):
        from agents_core.core.base_runner import BaseAgentRunner, _CompositeHooks
        from agents_core.core.turn_budget import TurnBudget

        hooks = BaseAgentRunner._build_hooks(
            budget=TurnBudget(), recovery=ToolErrorRecovery()
        )
        assert isinstance(hooks, _CompositeHooks)

    @pytest.mark.asyncio
    async def test_execute_basic_with_recovery(self, runner):
        recovery = ToolErrorRecovery()
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
                    "test_agent", context, session, 10, "hello",
                    error_recovery=recovery,
                )

                assert result == "test output"
                call_kwargs = MockRunner.run.call_args[1]
                assert isinstance(call_kwargs["hooks"], ToolErrorRecoveryHooks)

    @pytest.mark.asyncio
    async def test_execute_resets_recovery(self, runner):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "old"}))
        assert recovery.has_errors

        context = Mock(spec=[])
        with patch.object(runner, "_execute_basic", new_callable=AsyncMock) as mock_basic:
            mock_basic.return_value = "output"
            await runner.execute(
                "test_agent", context, session=Mock(),
                input_text="hello",
                error_recovery=recovery,
            )
        assert not recovery.has_errors  # reset at start


# ------------------------------------------------------------------ #
# _CompositeHooks
# ------------------------------------------------------------------ #


class TestCompositeHooks:
    @pytest.mark.asyncio
    async def test_delegates_to_all_hooks(self):
        from agents_core.core.base_runner import _CompositeHooks

        hook_a = Mock()
        hook_a.on_tool_start = AsyncMock()
        hook_a.on_tool_end = AsyncMock()
        hook_a.on_llm_start = AsyncMock()

        hook_b = Mock()
        hook_b.on_tool_start = AsyncMock()
        hook_b.on_tool_end = AsyncMock()
        hook_b.on_llm_start = AsyncMock()

        composite = _CompositeHooks([hook_a, hook_b])
        await composite.on_tool_start(Mock(), Mock(), Mock())
        await composite.on_tool_end(Mock(), Mock(), Mock(), "result")
        await composite.on_llm_start(Mock(), Mock(), None, [])

        hook_a.on_tool_start.assert_called_once()
        hook_b.on_tool_start.assert_called_once()
        hook_a.on_tool_end.assert_called_once()
        hook_b.on_tool_end.assert_called_once()
        hook_a.on_llm_start.assert_called_once()
        hook_b.on_llm_start.assert_called_once()


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


class TestTopLevelImports:
    def test_importable_from_core(self):
        from agents_core.core import ToolErrorRecovery, ToolErrorRecoveryHooks
        assert ToolErrorRecovery is not None
        assert ToolErrorRecoveryHooks is not None

    def test_importable_from_top_level(self):
        from agents_core import ToolErrorRecovery, ToolErrorRecoveryHooks
        assert ToolErrorRecovery is not None
        assert ToolErrorRecoveryHooks is not None
