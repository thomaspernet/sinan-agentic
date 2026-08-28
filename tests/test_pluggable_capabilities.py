"""Integration tests for ``AgentDefinition.capabilities`` (issue #5).

Verifies the capability pipeline:
- A custom Capability wired via ``capabilities=[...]`` is dispatched correctly
  without any ``base_runner.py`` edits.
- Capability state is isolated across sequential ``execute()`` calls (clones
  + reset).
- ``_CompositeHooks`` and ``_apply_dynamic_instructions`` are generic over
  capabilities (no TurnBudget/ToolErrorRecovery names).
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import RunContextWrapper, Tool

from sinan_agentic_core.core import base_runner as base_runner_module
from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from sinan_agentic_core.registry.guardrail_registry import GuardrailRegistry
from sinan_agentic_core.registry.tool_registry import ToolRegistry


class LoggingCapability(Capability):
    """Records every lifecycle call — used to verify the pipeline."""

    def __init__(self) -> None:
        self.tool_starts: list[tuple[str, str]] = []
        self.tool_ends: list[tuple[str, str]] = []
        self.llm_starts: int = 0
        self.instruction_calls: int = 0

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        self.instruction_calls += 1
        return f"[logging] tool_starts={len(self.tool_starts)}"

    def on_tool_start(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        args: str,
    ) -> None:
        name = getattr(tool, "name", str(tool))
        self.tool_starts.append((name, args))

    def on_tool_end(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        result: str,
    ) -> None:
        name = getattr(tool, "name", str(tool))
        self.tool_ends.append((name, result))

    def on_llm_start(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        system_prompt: str | None,
        input_items: Any,
    ) -> None:
        self.llm_starts += 1

    def reset(self) -> None:
        self.tool_starts.clear()
        self.tool_ends.clear()
        self.llm_starts = 0
        self.instruction_calls = 0


@pytest.fixture
def runner_with_capability_agent():
    """Spin up a runner whose registered agent declares a LoggingCapability."""
    cap = LoggingCapability()

    agent_reg = AgentRegistry()
    agent_reg.register(
        AgentDefinition(
            name="logger_agent",
            description="agent under test",
            instructions="You are a test agent.",
            tools=[],
            capabilities=[cap],
        )
    )
    tool_reg = ToolRegistry()
    guardrail_reg = GuardrailRegistry()

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        from sinan_agentic_core.core.base_runner import BaseAgentRunner

        runner = BaseAgentRunner()
        yield runner, cap


class TestCustomCapabilityIntegration:
    async def test_capability_lifecycle_dispatches(self, runner_with_capability_agent) -> None:
        runner, declarative_cap = runner_with_capability_agent

        mock_result = Mock()
        mock_result.final_output = "ok"
        mock_result.new_items = []
        mock_result.raw_responses = []

        captured_hooks: dict[str, Any] = {}
        captured_agent: dict[str, Any] = {}

        async def fake_run(**kwargs):
            captured_hooks["hooks"] = kwargs.get("hooks")
            captured_agent["agent"] = kwargs["starting_agent"]
            return mock_result

        with patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls:
            mock_runner_cls.run = AsyncMock(side_effect=fake_run)
            with patch.object(runner, "create_agent", new_callable=AsyncMock) as mock_create:
                mock_agent = Mock()
                mock_agent.tools = []
                mock_agent.instructions = "Static."
                mock_create.return_value = mock_agent

                await runner.execute(
                    "logger_agent",
                    context=Mock(spec=[]),
                    session=Mock(),
                    input_text="hello",
                )

        # The hooks attached to the run dispatch to the *cloned* capability, not
        # the declarative template instance.
        composite = captured_hooks["hooks"]
        assert composite is not None
        clones = composite._capabilities
        assert len(clones) == 1
        run_cap = clones[0]
        assert isinstance(run_cap, LoggingCapability)
        assert run_cap is not declarative_cap

        # Drive lifecycle through the SDK-shaped adapter and confirm the clone
        # received the calls.
        ctx = Mock()
        ctx.tool_arguments = '{"q": "hi"}'
        tool = Mock()
        tool.name = "search"

        await composite.on_tool_start(ctx, mock_agent, tool)
        await composite.on_tool_end(ctx, mock_agent, tool, '{"result":"x"}')
        await composite.on_llm_start(ctx, mock_agent, None, [])

        assert run_cap.tool_starts == [("search", '{"q": "hi"}')]
        assert run_cap.tool_ends == [("search", '{"result":"x"}')]
        assert run_cap.llm_starts == 1

        # Declarative template stays clean.
        assert declarative_cap.tool_starts == []
        assert declarative_cap.tool_ends == []
        assert declarative_cap.llm_starts == 0

    async def test_capability_instructions_are_merged(self, runner_with_capability_agent) -> None:
        runner, _ = runner_with_capability_agent

        mock_result = Mock()
        mock_result.final_output = "ok"
        mock_result.new_items = []
        mock_result.raw_responses = []

        captured: dict[str, Any] = {}

        async def fake_run(**kwargs):
            captured["agent"] = kwargs["starting_agent"]
            return mock_result

        with patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls:
            mock_runner_cls.run = AsyncMock(side_effect=fake_run)
            with patch.object(runner, "create_agent", new_callable=AsyncMock) as mock_create:
                mock_agent = Mock()
                mock_agent.tools = []
                mock_agent.instructions = "Base instructions."
                mock_create.return_value = mock_agent

                await runner.execute(
                    "logger_agent",
                    context=Mock(spec=[]),
                    session=Mock(),
                    input_text="hello",
                )

        # ``_apply_dynamic_instructions`` wraps the agent's instructions with a
        # callable that merges every capability fragment.
        wrapped = captured["agent"].instructions
        assert callable(wrapped)
        rendered = wrapped(RunContextWrapper(context=None), captured["agent"])
        assert "Base instructions." in rendered
        assert "[logging]" in rendered


class TestCapabilityStateIsolation:
    async def test_state_does_not_leak_between_runs(self, runner_with_capability_agent) -> None:
        runner, declarative_cap = runner_with_capability_agent

        mock_result = Mock()
        mock_result.final_output = "ok"
        mock_result.new_items = []
        mock_result.raw_responses = []

        captured_clones: list[LoggingCapability] = []

        async def fake_run(**kwargs):
            composite = kwargs["hooks"]
            run_cap = composite._capabilities[0]
            assert isinstance(run_cap, LoggingCapability)
            captured_clones.append(run_cap)
            # Simulate work during the run: bump state.
            run_cap.tool_starts.append(("simulated", "{}"))
            run_cap.llm_starts += 1
            return mock_result

        with patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls:
            mock_runner_cls.run = AsyncMock(side_effect=fake_run)
            with patch.object(runner, "create_agent", new_callable=AsyncMock) as mock_create:
                mock_agent = Mock()
                mock_agent.tools = []
                mock_agent.instructions = "Static."
                mock_create.return_value = mock_agent

                # Two sequential runs share the same registered AgentDefinition
                # — yet state must not bleed across them.
                await runner.execute(
                    "logger_agent",
                    context=Mock(spec=[]),
                    session=Mock(),
                    input_text="hello",
                )
                await runner.execute(
                    "logger_agent",
                    context=Mock(spec=[]),
                    session=Mock(),
                    input_text="world",
                )

        assert len(captured_clones) == 2
        run1, run2 = captured_clones
        # Each run got its own clone — verifies clone() wired through.
        assert run1 is not run2
        assert run1 is not declarative_cap
        assert run2 is not declarative_cap
        # Run-1's mutations did not leak into run-2 — verifies reset() at the
        # start of the second execute().
        assert run2.tool_starts == [("simulated", "{}")]
        assert run2.llm_starts == 1
        # The declarative template stays pristine across both runs.
        assert declarative_cap.tool_starts == []
        assert declarative_cap.llm_starts == 0


class TestCompositeHooksGeneric:
    """Verify ``_CompositeHooks`` and the dynamic-instructions builder no
    longer reference TurnBudget or ToolErrorRecovery by name."""

    def test_composite_hooks_source_has_no_turn_budget_branch(self) -> None:
        src = inspect.getsource(base_runner_module._CompositeHooks)
        assert "TurnBudget" not in src
        assert "ToolErrorRecovery" not in src

    def test_dynamic_instructions_helper_has_no_capability_branches(self) -> None:
        src = inspect.getsource(base_runner_module._merge_capability_instructions)
        assert "TurnBudget" not in src
        assert "ToolErrorRecovery" not in src
