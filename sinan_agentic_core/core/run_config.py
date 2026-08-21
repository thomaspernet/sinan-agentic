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

Usage:
    from sinan_agentic_core import build_run_config
    from agents import Runner

    agent = create_agent_from_registry("my_agent")
    run_config = build_run_config(agent)
    result = await Runner.run(agent, "Hello!", run_config=run_config)
"""

from __future__ import annotations

from typing import Any

from agents import Agent, FunctionTool, RunConfig, ToolExecutionConfig

from ..registry.agent_registry import resolve_agent_definition
from .tool_output_trim import ToolOutputTrimConfig, build_tool_output_trimmer


def build_run_config(agent: Agent) -> RunConfig | None:
    """Build the SDK run config *agent* needs, or None when defaults apply.

    An agent whose function tools carry tool-input guardrails runs with
    pre-approval on, so a rejected call never reaches a human approver. An agent
    whose definition declares ``tool_output_trim`` runs with the SDK's
    tool-output filter, so bulky outputs from older turns shrink before each
    model call instead of growing the input until the run overflows.

    Args:
        agent: A built agent, from the registry factory or assembled by hand.
            The trim policy is looked up by ``agent.name``, so an agent
            assembled by hand picks up the policy declared for that name.

    Returns:
        Configured RunConfig, or None when no setting differs from the default.
    """
    config_kwargs: dict[str, Any] = {}

    if _has_tool_input_guardrails(agent):
        config_kwargs["tool_execution"] = tool_input_pre_approval()

    trimmer = build_tool_output_trimmer(_declared_tool_output_trim(agent))
    if trimmer is not None:
        config_kwargs["call_model_input_filter"] = trimmer

    if not config_kwargs:
        return None

    return RunConfig(**config_kwargs)


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
    return any(
        isinstance(tool, FunctionTool) and tool.tool_input_guardrails for tool in agent.tools
    )


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
