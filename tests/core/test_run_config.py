"""Tests for run-level SDK settings derived from a built agent (core/run_config.py)."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import pytest
from agents import (
    Agent,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    WebSearchTool,
    function_tool,
    tool_input_guardrail,
)
from agents.extensions import ToolOutputTrimmer

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.capabilities.steering import (
    STEERING_ITEM_ROLE,
    CapabilitySteering,
)
from sinan_agentic_core.core.run_config import (
    build_model_input_filter,
    build_run_config,
    tool_input_pre_approval,
)
from sinan_agentic_core.core.tool_output_trim import ToolOutputTrimConfig
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from tests.conftest import drive_model_input_filter


class Speaking(Capability):
    """A capability with one fixed fragment, enough to place the steering item."""

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        return "stay on task"


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


@pytest.fixture
def trim_registry():
    """An isolated registry, patched inside the shared by-name definition resolver."""
    registry = AgentRegistry()
    registry.register(
        AgentDefinition(
            name="trimming_agent",
            description="trims",
            instructions="answer",
            tool_output_trim=ToolOutputTrimConfig(recent_turns=3, max_output_chars=4000),
        )
    )
    registry.register(
        AgentDefinition(name="plain_agent", description="plain", instructions="answer")
    )

    with patch(
        "sinan_agentic_core.registry.agent_registry.get_agent_registry", return_value=registry
    ):
        yield registry


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

    def test_capabilities_install_the_steering_filter(self, trim_registry):
        run_config = build_run_config(Agent(name="plain_agent"), [Speaking()])

        assert isinstance(run_config.call_model_input_filter, CapabilitySteering)

    def test_no_capabilities_steer_nothing(self, trim_registry):
        """The chat service passes none, so its runs are unchanged."""
        assert build_run_config(Agent(name="plain_agent")) is None


class TestDeclaredToolOutputTrim:
    """Trimming has no slot on ``Agent``, so it is resolved through the registry.

    Without it, an agent that declares ``tool_output_trim`` is trimmed when the
    runner executes it and left untrimmed when the chat service does — the same
    declaration honored on one path and dropped on the other.
    """

    def test_a_declared_policy_installs_the_filter(self, trim_registry):
        run_config = build_run_config(Agent(name="trimming_agent"))

        assert run_config is not None
        assert run_config.call_model_input_filter.recent_turns == 3
        assert run_config.call_model_input_filter.max_output_chars == 4000

    def test_an_agent_declaring_nothing_gets_no_run_config(self, trim_registry):
        assert build_run_config(Agent(name="plain_agent")) is None

    def test_an_unregistered_agent_declares_nothing(self, trim_registry):
        """A hand-assembled agent under an unknown name has no policy to honor."""
        assert build_run_config(Agent(name="assembled_by_hand")) is None

    def test_trimming_composes_with_tool_input_pre_approval(self, trim_registry, echo_tool, guard):
        """Both settings share one RunConfig — declaring either must not drop the other."""
        agent = Agent(name="trimming_agent", tools=[_guarded(echo_tool, guard)])

        run_config = build_run_config(agent)

        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True
        assert run_config.call_model_input_filter.max_output_chars == 4000

    def test_each_call_returns_a_fresh_filter(self, trim_registry):
        """The SDK dataclass is mutable, so two runs of one agent must not share it."""
        agent = Agent(name="trimming_agent")

        first = build_run_config(agent).call_model_input_filter
        second = build_run_config(agent).call_model_input_filter

        assert first is not second


class TestBuildModelInputFilter:
    """The one ``call_model_input_filter`` slot, shared by trimming and steering."""

    def test_neither_declared_installs_nothing(self):
        assert build_model_input_filter(None, []) is None

    def test_trimming_alone_installs_the_trimmer(self):
        built = build_model_input_filter(ToolOutputTrimConfig(recent_turns=3), [])

        assert isinstance(built, ToolOutputTrimmer)

    def test_steering_alone_installs_the_steering_filter(self):
        assert isinstance(build_model_input_filter(None, [Speaking()]), CapabilitySteering)

    def test_declaring_both_runs_both(self):
        built = build_model_input_filter(
            ToolOutputTrimConfig(recent_turns=1, max_output_chars=10, preview_chars=4),
            [Speaking()],
        )

        steered = drive_model_input_filter(built, input_items=_TRIMMABLE_CONVERSATION)

        assert steered.input[-1]["content"] == "stay on task"
        assert _tool_output(steered.input) != _LONG_TOOL_OUTPUT

    def test_trimming_runs_first_so_steering_stays_last(self):
        """Trimming reads the real conversation, not one steering has extended."""
        built = build_model_input_filter(
            ToolOutputTrimConfig(recent_turns=1, max_output_chars=10, preview_chars=4),
            [Speaking()],
        )

        steered = drive_model_input_filter(built, input_items=_TRIMMABLE_CONVERSATION)

        assert steered.input[-1]["role"] == STEERING_ITEM_ROLE
        assert len(steered.input) == len(_TRIMMABLE_CONVERSATION) + 1

    def test_each_call_returns_fresh_callables(self):
        first = build_model_input_filter(ToolOutputTrimConfig(), [Speaking()])
        second = build_model_input_filter(ToolOutputTrimConfig(), [Speaking()])

        assert first is not second


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


# A conversation the trimmer will actually shorten: two user messages, so its
# one-message window leaves the first turn's oversized tool output eligible.
_LONG_TOOL_OUTPUT = "x" * 400

_TRIMMABLE_CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "first question"},
    {
        "type": "function_call",
        "call_id": "call-1",
        "name": "search",
        "arguments": "{}",
    },
    {"type": "function_call_output", "call_id": "call-1", "output": _LONG_TOOL_OUTPUT},
    {"role": "user", "content": "second question"},
]


def _tool_output(items: list[Any]) -> str:
    """The single tool output in a filtered conversation."""
    outputs = [item["output"] for item in items if item.get("type") == "function_call_output"]
    assert len(outputs) == 1
    return str(outputs[0])
