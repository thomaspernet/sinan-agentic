"""Run-level SDK settings derived from a built agent.

Some SDK settings belong to the *run*, not the agent: they live on ``RunConfig``
and are passed to ``Runner.run()`` / ``Runner.run_streamed()``, so building an
agent is not enough to get them. Tool-input pre-approval and tool-output trimming
are both of these — an agent can carry tool-input guardrails on its function
tools, and declare ``tool_output_trim`` in ``agents.yaml``, while the run they
execute in leaves both behind.

``BaseAgentRunner`` derives its run config from the agent *definition*, which it
has on every execution branch. Call sites that only hold a built ``Agent`` — the
chat service, and consumers driving ``Runner`` themselves — resolve the same
settings here instead, so a pre-built agent runs on the same terms as a
registry-resolved one. Pre-approval is read off the agent's own tools; trimming
has no slot on ``Agent``, so it is read off the definition registered under the
agent's name.

``RunConfig`` has a single ``call_model_input_filter`` slot and two things want
it — the declared tool-output trimmer and capability steering — so both
run-config-building paths go through :func:`build_model_input_filter` for it
rather than each deciding what the slot holds.

Usage:
    from sinan_agentic_core import build_run_config
    from agents import Runner

    agent = create_agent_from_registry("my_agent")
    run_config = build_run_config(agent)
    result = await Runner.run(agent, "Hello!", run_config=run_config)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from agents import Agent, RunConfig, ToolExecutionConfig
from agents.extensions import ToolOutputTrimmer
from agents.run_config import CallModelData, CallModelInputFilter, ModelInputData

from ..registry.agent_registry import resolve_agent_definition
from ..registry.guardrail_registry import has_tool_input_guardrails
from .capabilities import Capability, CapabilitySteering, build_capability_steering
from .tool_output_trim import ToolOutputTrimConfig, build_tool_output_trimmer


def build_run_config(
    agent: Agent,
    capabilities: Sequence[Capability] = (),
) -> RunConfig | None:
    """Build the SDK run config *agent* needs, or None when defaults apply.

    An agent whose function tools carry tool-input guardrails runs with
    pre-approval on, so a rejected call never reaches a human approver. An agent
    whose definition declares ``tool_output_trim`` runs with the SDK's
    tool-output filter, so bulky outputs from older turns shrink before each
    model call instead of growing the input until the run overflows. A run given
    capabilities runs with steering, so their fragments reach the model at the
    tail of the input.

    Args:
        agent: A built agent, from the registry factory or assembled by hand.
            The trim policy is looked up by ``agent.name``, so an agent
            assembled by hand picks up the policy declared for that name.
        capabilities: The run's capabilities, already cloned and reset by
            whoever owns the run. Left empty — as the chat service leaves it,
            having no capabilities to run — nothing is steered.

    Returns:
        Configured RunConfig, or None when no setting differs from the default.
    """
    config_kwargs: dict[str, Any] = {}

    if _has_tool_input_guardrails(agent):
        config_kwargs["tool_execution"] = tool_input_pre_approval()

    model_input_filter = build_model_input_filter(_declared_tool_output_trim(agent), capabilities)
    if model_input_filter is not None:
        config_kwargs["call_model_input_filter"] = model_input_filter

    if not config_kwargs:
        return None

    return RunConfig(**config_kwargs)


def build_model_input_filter(
    trim: ToolOutputTrimConfig | None,
    capabilities: Sequence[Capability],
) -> CallModelInputFilter | None:
    """Build the one filter a run installs in ``call_model_input_filter``, or None.

    Both run-config-building paths call this — ``BaseAgentRunner._build_run_config``
    and :func:`build_run_config` — so declaring trimming and running with
    capabilities install both filters whichever path assembled the config, rather
    than the second silently taking the slot from the first.

    Trimming runs first. It decides what to shrink from the items it is handed
    and measures its untouched window in user messages, so it reads the real
    conversation rather than one a steering item has already extended. Steering
    then appends to whatever trimming returned, which leaves its item last in
    the input on every call.

    Args:
        trim: The agent's declared trim policy, or None when it opts out.
        capabilities: The run's capabilities. Empty means nothing to steer with.

    Returns:
        The filter to install, or None when neither part applies. Each call
        builds fresh callables, so no two runs share filter state.
    """
    trimmer = build_tool_output_trimmer(trim)
    steering = build_capability_steering(capabilities)

    if steering is None:
        return trimmer
    if trimmer is None:
        return steering

    return _TrimThenSteer(trimmer, steering)


class _TrimThenSteer:
    """Trim tool outputs, then append the steering item to what trimming left."""

    def __init__(self, trimmer: ToolOutputTrimmer, steering: CapabilitySteering) -> None:
        self._trimmer = trimmer
        self._steering = steering

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        trimmed = replace(data, model_data=self._trimmer(data))
        return self._steering(trimmed)


def tool_input_pre_approval() -> ToolExecutionConfig:
    """Build the tool-execution setting that runs tool-input guardrails first.

    With it, a tool call that needs human approval has its tool-input guardrails
    run *before* the SDK emits the pending-approval interruption, and a rejecting
    guardrail returns its message as the tool output — so a call the guardrail
    would refuse never reaches an approver. Without it, those guardrails run only
    once the approval is resolved, just before the tool executes
    (``openai-agents`` 0.20.0, ``agents.run_internal.tool_execution``).

    Returns:
        A fresh ToolExecutionConfig with pre-approval enabled. The SDK dataclass
        is mutable, so callers get their own instance rather than a shared one.
    """
    return ToolExecutionConfig(pre_approval_tool_input_guardrails=True)


def _has_tool_input_guardrails(agent: Agent) -> bool:
    """Whether any of *agent*'s local function tools carries a tool-input guardrail."""
    return any(has_tool_input_guardrails(tool) for tool in agent.tools)


def _declared_tool_output_trim(agent: Agent) -> ToolOutputTrimConfig | None:
    """The trim policy declared for *agent*, read off its registered definition.

    Trimming is a run-level setting with no slot on ``Agent``, so unlike
    pre-approval it cannot be read off the built agent — the declaration is
    resolved through :func:`~sinan_agentic_core.registry.agent_registry.resolve_agent_definition`,
    the shared by-name reader every run-level setting goes through. An agent
    assembled under a name the registry does not know declares nothing.
    """
    agent_def = resolve_agent_definition(agent)
    if agent_def is None:
        return None

    return agent_def.tool_output_trim
