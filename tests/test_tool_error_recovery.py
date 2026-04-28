"""Tests for the tool error recovery system (core/tool_error_recovery.py)."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import RunContextWrapper

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.tool_error_recovery import (
    ToolErrorEntry,
    ToolErrorRecovery,
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

    def test_status_failed_detected_as_error(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool", json.dumps({"status": "failed", "message": "Page not found"})
        )
        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["error"] == "Page not found"

    def test_status_validation_error_detected(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool",
            json.dumps({"status": "validation_error", "message": "name is required"}),
        )
        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["error"] == "name is required"

    def test_status_completed_is_success(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool",
            json.dumps({"status": "completed", "message": "Page created"}),
        )
        assert not recovery.has_errors

    def test_status_failed_without_message_uses_fallback(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool", json.dumps({"status": "failed"})
        )
        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert "failed" in summary["my_tool"]["error"]

    def test_error_key_takes_priority_over_status(self):
        """When both 'error' and 'status' are present, 'error' wins."""
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool",
            json.dumps({"error": "explicit error", "status": "failed", "message": "status msg"}),
        )
        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["error"] == "explicit error"

    def test_status_failed_clears_on_subsequent_success(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result(
            "my_tool", json.dumps({"status": "failed", "message": "bad"})
        )
        assert recovery.has_errors
        recovery.record_tool_result(
            "my_tool", json.dumps({"status": "completed", "message": "ok"})
        )
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

    def test_hint_shown_for_status_based_error(self):
        registry = Mock()
        tool_def = Mock()
        tool_def.recovery_hint = "Use search to find the UUID first."
        registry.get_tool.return_value = tool_def

        recovery = ToolErrorRecovery(tool_registry=registry)
        recovery.record_tool_result(
            "update_page",
            json.dumps({"status": "failed", "message": "Page not found: my-page"}),
        )
        section = recovery.build_instruction_section()
        assert "Use search to find the UUID first." in section
        assert "Page not found" in section


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
# ToolErrorRecovery — capability lifecycle hooks
# ------------------------------------------------------------------ #


class TestToolErrorRecoveryLifecycle:
    def test_on_tool_end_records_result(self):
        recovery = ToolErrorRecovery()

        tool = Mock()
        tool.name = "my_tool"

        ctx = Mock()
        recovery.on_tool_start(ctx, tool, json.dumps({"mode": "check"}))
        recovery.on_tool_end(ctx, tool, json.dumps({"error": "fail"}))

        assert recovery.has_errors
        summary = recovery.get_error_summary()
        assert summary["my_tool"]["error"] == "fail"

    def test_on_tool_end_success_clears_error(self):
        recovery = ToolErrorRecovery()

        tool = Mock()
        tool.name = "my_tool"
        ctx = Mock()

        recovery.on_tool_start(ctx, tool, "{}")
        recovery.on_tool_end(ctx, tool, json.dumps({"error": "fail"}))
        assert recovery.has_errors

        recovery.on_tool_start(ctx, tool, "{}")
        recovery.on_tool_end(ctx, tool, json.dumps({"result": "ok"}))
        assert not recovery.has_errors

    def test_on_event_emitted_on_error(self):
        recovery = ToolErrorRecovery()
        events: list[dict] = []
        recovery.on_event = events.append

        tool = Mock()
        tool.name = "my_tool"
        ctx = Mock()

        recovery.on_tool_start(ctx, tool, "{}")
        recovery.on_tool_end(ctx, tool, json.dumps({"error": "fail"}))

        assert len(events) == 1
        assert events[0]["event"] == "tool_error_recovery"

    def test_no_event_on_success(self):
        recovery = ToolErrorRecovery()
        events: list[dict] = []
        recovery.on_event = events.append

        tool = Mock()
        tool.name = "my_tool"
        ctx = Mock()

        recovery.on_tool_start(ctx, tool, "{}")
        recovery.on_tool_end(ctx, tool, json.dumps({"data": "ok"}))

        assert len(events) == 0


# ------------------------------------------------------------------ #
# BaseAgentRunner integration
# ------------------------------------------------------------------ #


class TestBaseAgentRunnerIntegration:
    @pytest.fixture
    def _registries(self):
        from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
        from sinan_agentic_core.registry.tool_registry import ToolRegistry
        from sinan_agentic_core.registry.guardrail_registry import GuardrailRegistry

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
            patch("sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg),
        ):
            return BaseAgentRunner()

    def test_apply_dynamic_instructions_with_recovery(self, runner):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "fail"}))

        agent = Mock()
        agent.instructions = "Base instructions."
        runner._apply_dynamic_instructions(agent, [recovery])

        assert callable(agent.instructions)
        result = agent.instructions(Mock(), Mock())
        assert "Base instructions." in result
        assert "Tool Error Recovery" in result

    def test_apply_dynamic_instructions_no_capabilities(self, runner):
        agent = Mock()
        agent.instructions = "Static."
        runner._apply_dynamic_instructions(agent, [])
        # Should NOT replace with callable
        assert agent.instructions == "Static."

    def test_apply_dynamic_instructions_both_budget_and_recovery(self, runner):
        from sinan_agentic_core.core.turn_budget import TurnBudget

        budget = TurnBudget(default_turns=10)
        budget.turns_used = 8
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("t", json.dumps({"error": "fail"}))

        agent = Mock()
        agent.instructions = "Base."
        runner._apply_dynamic_instructions(agent, [budget, recovery])

        result = agent.instructions(Mock(), Mock())
        assert "Base." in result
        assert "remaining" in result  # budget section
        assert "Tool Error Recovery" in result  # recovery section

    def test_build_hooks_none_when_no_capabilities(self):
        from sinan_agentic_core.core.base_runner import BaseAgentRunner
        assert BaseAgentRunner._build_hooks([]) is None

    def test_build_hooks_returns_composite(self):
        from sinan_agentic_core.core.base_runner import BaseAgentRunner, _CompositeHooks
        from sinan_agentic_core.core.turn_budget import TurnBudget

        hooks = BaseAgentRunner._build_hooks([TurnBudget()])
        assert isinstance(hooks, _CompositeHooks)

    def test_build_hooks_composite_when_multiple(self):
        from sinan_agentic_core.core.base_runner import BaseAgentRunner, _CompositeHooks
        from sinan_agentic_core.core.turn_budget import TurnBudget

        hooks = BaseAgentRunner._build_hooks([TurnBudget(), ToolErrorRecovery()])
        assert isinstance(hooks, _CompositeHooks)

    @pytest.mark.asyncio
    async def test_execute_basic_with_recovery(self, runner):
        from sinan_agentic_core.core.base_runner import _CompositeHooks

        recovery = ToolErrorRecovery()
        context = Mock()
        session = Mock()

        mock_result = Mock()
        mock_result.final_output = "test output"

        with patch("sinan_agentic_core.core.base_runner.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            with patch.object(runner, "create_agent", new_callable=AsyncMock) as mock_create:
                mock_agent = Mock()
                mock_agent.tools = []
                mock_agent.instructions = "Static."
                mock_create.return_value = mock_agent

                result = await runner._execute_basic(
                    "test_agent", context, session, 10, "hello",
                    capabilities=[recovery],
                )

                assert result == "test output"
                call_kwargs = MockRunner.run.call_args[1]
                assert isinstance(call_kwargs["hooks"], _CompositeHooks)

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
    async def test_delegates_to_all_capabilities(self):
        from sinan_agentic_core.core.base_runner import _CompositeHooks

        cap_a = Mock(spec=Capability)
        cap_b = Mock(spec=Capability)

        composite = _CompositeHooks([cap_a, cap_b])
        await composite.on_tool_start(Mock(), Mock(), Mock())
        await composite.on_tool_end(Mock(), Mock(), Mock(), "result")
        await composite.on_llm_start(Mock(), Mock(), None, [])

        cap_a.on_tool_start.assert_called_once()
        cap_b.on_tool_start.assert_called_once()
        cap_a.on_tool_end.assert_called_once()
        cap_b.on_tool_end.assert_called_once()
        cap_a.on_llm_start.assert_called_once()
        cap_b.on_llm_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_extracts_tool_arguments_from_context(self):
        from sinan_agentic_core.core.base_runner import _CompositeHooks

        cap = Mock(spec=Capability)
        composite = _CompositeHooks([cap])
        ctx = Mock()
        ctx.tool_arguments = json.dumps({"x": 1})
        tool = Mock()

        await composite.on_tool_start(ctx, Mock(), tool)
        cap.on_tool_start.assert_called_once_with(ctx, tool, ctx.tool_arguments)


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


class TestTopLevelImports:
    def test_importable_from_core(self):
        from sinan_agentic_core.core import ToolErrorRecovery
        assert ToolErrorRecovery is not None

    def test_importable_from_top_level(self):
        from sinan_agentic_core import ToolErrorRecovery
        assert ToolErrorRecovery is not None


# ------------------------------------------------------------------ #
# Capability protocol adoption
# ------------------------------------------------------------------ #


class TestToolErrorRecoveryIsCapability:
    def test_is_capability_subclass(self):
        assert issubclass(ToolErrorRecovery, Capability)

    def test_instance_is_capability(self):
        assert isinstance(ToolErrorRecovery(), Capability)

    def test_instructions_returns_none_when_clean(self):
        recovery = ToolErrorRecovery()
        ctx = RunContextWrapper(context=None)
        assert recovery.instructions(ctx) is None

    def test_instructions_returns_section_when_errors_tracked(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        ctx = RunContextWrapper(context=None)
        section = recovery.instructions(ctx)
        assert section is not None
        assert "Tool Error Recovery" in section

    def test_instructions_matches_build_instruction_section(self):
        recovery = ToolErrorRecovery()
        recovery.record_tool_result("my_tool", json.dumps({"error": "fail"}))
        ctx = RunContextWrapper(context=None)
        assert recovery.instructions(ctx) == recovery.build_instruction_section()

    def test_on_tool_start_then_end_records_error(self):
        recovery = ToolErrorRecovery()
        ctx = RunContextWrapper(context=None)
        tool = Mock()
        tool.name = "my_tool"

        recovery.on_tool_start(ctx, tool, json.dumps({"q": "x"}))
        recovery.on_tool_end(ctx, tool, json.dumps({"error": "fail"}))

        summary = recovery.get_error_summary()
        assert summary["my_tool"]["error"] == "fail"

    def test_on_tool_end_uses_args_from_on_tool_start(self):
        recovery = ToolErrorRecovery()
        ctx = RunContextWrapper(context=None)
        tool = Mock()
        tool.name = "my_tool"

        # Two identical-arg failures via the Capability hooks should be detected
        # as identical retries.
        args = json.dumps({"q": "x"})
        for _ in range(2):
            recovery.on_tool_start(ctx, tool, args)
            recovery.on_tool_end(ctx, tool, json.dumps({"error": "fail"}))

        summary = recovery.get_error_summary()
        assert summary["my_tool"]["identical_count"] == 2

    def test_tools_default_is_empty(self):
        assert ToolErrorRecovery().tools() == []


class TestToolErrorRecoveryClone:
    def test_clone_returns_tool_error_recovery(self):
        clone = ToolErrorRecovery().clone()
        assert isinstance(clone, ToolErrorRecovery)

    def test_clone_is_independent_instance(self):
        original = ToolErrorRecovery()
        clone = original.clone()
        assert clone is not original

    def test_clone_preserves_configuration(self):
        registry = Mock()
        original = ToolErrorRecovery(
            tool_registry=registry,
            mcp_hints={"mcp_search": "Use a longer query."},
            max_identical_before_stop=5,
        )
        clone = original.clone()
        assert clone._registry is registry
        assert clone._mcp_hints == {"mcp_search": "Use a longer query."}
        assert clone.max_identical_before_stop == 5

    def test_clone_does_not_share_mcp_hints(self):
        original = ToolErrorRecovery(mcp_hints={"a": "hint"})
        clone = original.clone()
        clone._mcp_hints["b"] = "new hint"
        assert "b" not in original._mcp_hints

    def test_clone_zeroes_error_state(self):
        original = ToolErrorRecovery()
        original.record_tool_result("t", json.dumps({"error": "old"}), json.dumps({"a": 1}))
        original.on_tool_start(RunContextWrapper(context=None), Mock(name="pending_tool"), "{}")
        assert original.has_errors

        clone = original.clone()
        assert not clone.has_errors
        assert clone.get_error_summary() == {}
        assert clone._last_args == {}
        assert clone._pending_args == {}

    def test_clone_does_not_mutate_original(self):
        original = ToolErrorRecovery()
        original.record_tool_result("t", json.dumps({"error": "keep"}), json.dumps({"a": 1}))

        clone = original.clone()
        clone.record_tool_result("t", json.dumps({"error": "fresh"}), json.dumps({"a": 2}))

        # Original keeps its single tracked error untouched
        assert original.get_error_summary()["t"]["error"] == "keep"
        assert original.get_error_summary()["t"]["call_count"] == 1
        assert clone.get_error_summary()["t"]["error"] == "fresh"


class TestResetClearsPendingArgs:
    def test_reset_clears_pending_args(self):
        recovery = ToolErrorRecovery()
        ctx = RunContextWrapper(context=None)
        tool = Mock()
        tool.name = "my_tool"
        recovery.on_tool_start(ctx, tool, json.dumps({"q": "x"}))
        assert recovery._pending_args  # in-flight

        recovery.reset()
        assert recovery._pending_args == {}


# ------------------------------------------------------------------ #
# Snapshot / rehydrate
# ------------------------------------------------------------------ #


def _record_failure(recovery: ToolErrorRecovery, tool_name: str, args: dict, message: str = "boom") -> None:
    """Helper: simulate a single failed tool call."""
    ctx = RunContextWrapper(context=None)
    tool = Mock()
    tool.name = tool_name
    recovery.on_tool_start(ctx, tool, json.dumps(args))
    recovery.on_tool_end(ctx, tool, json.dumps({"error": message}))


class TestToolErrorRecoverySnapshot:
    def test_snapshot_captures_tracked_errors(self):
        recovery = ToolErrorRecovery()
        _record_failure(recovery, "search", {"q": "x"}, "404")

        snap = recovery.to_snapshot()
        assert "search" in snap["errors"]
        assert snap["errors"]["search"]["call_count"] == 1
        assert snap["errors"]["search"]["error"] == "404"
        assert "search" in snap["last_args"]

    def test_round_trip_preserves_error_state(self):
        # AC: same round-trip behavior for ToolErrorRecovery as TurnBudget.
        original = ToolErrorRecovery(max_identical_before_stop=3)
        _record_failure(original, "search", {"q": "a"}, "fail-1")
        _record_failure(original, "search", {"q": "a"}, "fail-2")
        _record_failure(original, "fetch", {"url": "u"}, "timeout")

        snap = original.to_snapshot()

        resumed = ToolErrorRecovery(max_identical_before_stop=3)
        resumed.from_snapshot(snap)

        assert resumed.has_errors
        assert resumed._errors["search"].call_count == 2
        assert resumed._errors["search"].identical_count == 2
        assert resumed._errors["fetch"].call_count == 1
        assert resumed._last_args["search"] == original._last_args["search"]
        assert resumed._pending_args == {}

    def test_snapshot_is_json_serializable(self):
        recovery = ToolErrorRecovery()
        _record_failure(recovery, "search", {"q": "x"})
        json.dumps(recovery.to_snapshot())  # must not raise

    def test_from_snapshot_tolerates_missing_keys(self):
        recovery = ToolErrorRecovery()
        recovery.from_snapshot({})
        assert recovery._errors == {}
        assert recovery._last_args == {}

    def test_round_trip_preserves_instruction_escalation(self):
        original = ToolErrorRecovery(max_identical_before_stop=3)
        _record_failure(original, "search", {"q": "a"}, "fail")
        _record_failure(original, "search", {"q": "a"}, "fail")
        _record_failure(original, "search", {"q": "a"}, "fail")

        resumed = ToolErrorRecovery(max_identical_before_stop=3)
        resumed.from_snapshot(original.to_snapshot())
        assert resumed.has_critical_errors
        assert "STOP" in resumed.build_instruction_section()
