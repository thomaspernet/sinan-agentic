"""Tests for run-level SDK settings derived from a built agent (core/run_config.py)."""

from __future__ import annotations

import copy

import pytest
from agents import (
    Agent,
    ToolGuardrailFunctionOutput,
    WebSearchTool,
    function_tool,
    tool_input_guardrail,
)

from sinan_agentic_core.core.run_config import build_run_config, tool_input_pre_approval


@pytest.fixture
def echo_tool():
    """A local function tool, fresh per test so attachments cannot leak."""

    @function_tool
    def echo(value: str) -> str:
        """Echo a value.

        Args:
            value: Text to echo back.
        """
        return value

    return echo


@pytest.fixture
def guard():
    @tool_input_guardrail
    def block_nothing(data):
        return ToolGuardrailFunctionOutput()

    return block_nothing


def _guarded(tool, guardrail):
    """Copy *tool* with *guardrail* attached, the way the agent factory does."""
    guarded = copy.copy(tool)
    guarded.tool_input_guardrails = [guardrail]
    return guarded


class TestBuildRunConfig:
    def test_agent_with_tool_input_guardrail_enables_pre_approval(self, echo_tool, guard):
        agent = Agent(name="a", tools=[_guarded(echo_tool, guard)])

        run_config = build_run_config(agent)

        assert run_config is not None
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    def test_agent_without_guardrails_gets_no_run_config(self, echo_tool):
        assert build_run_config(Agent(name="a", tools=[echo_tool])) is None

    def test_agent_without_tools_gets_no_run_config(self):
        assert build_run_config(Agent(name="a")) is None

    def test_hosted_tools_do_not_enable_pre_approval(self):
        agent = Agent(name="a", tools=[WebSearchTool()])

        assert build_run_config(agent) is None

    def test_one_guarded_tool_among_many_enables_pre_approval(self, echo_tool, guard):
        agent = Agent(name="a", tools=[echo_tool, WebSearchTool(), _guarded(echo_tool, guard)])

        run_config = build_run_config(agent)

        assert run_config is not None
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    def test_only_tool_execution_is_set(self, echo_tool, guard):
        agent = Agent(name="a", tools=[_guarded(echo_tool, guard)])

        run_config = build_run_config(agent)

        assert run_config.call_model_input_filter is None
        assert run_config.model is None


class TestToolInputPreApproval:
    def test_enables_pre_approval(self):
        assert tool_input_pre_approval().pre_approval_tool_input_guardrails is True

    def test_leaves_other_settings_at_sdk_defaults(self):
        assert tool_input_pre_approval().max_function_tool_concurrency is None

    def test_each_call_returns_a_fresh_instance(self):
        first = tool_input_pre_approval()
        second = tool_input_pre_approval()

        first.max_function_tool_concurrency = 1

        assert first is not second
        assert second.max_function_tool_concurrency is None
