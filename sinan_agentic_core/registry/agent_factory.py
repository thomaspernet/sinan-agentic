"""Agent Factory - Build Agent instances from the registry.

Resolves an AgentDefinition + its registered tools and guardrails into a
ready-to-use OpenAI Agents SDK ``Agent`` instance.

Usage:
    from sinan_agentic_core import create_agent_from_registry

    agent = create_agent_from_registry("weather_assistant")
    result = await Runner.run(agent, "What's the weather?")
"""

from agents import Agent

from .agent_registry import get_agent_registry
from .guardrail_registry import attach_tool_input_guardrails, get_guardrail_registry
from .tool_registry import get_tool_registry


def create_agent_from_registry(
    agent_name: str,
    model_override: str | None = None,
) -> Agent:
    """Build an Agent instance from registry definitions.

    Looks up the AgentDefinition by name, resolves its tool references through
    the ToolRegistry and its guardrail references through the GuardrailRegistry,
    and returns a fully configured Agent. Each guardrail lands in the SDK slot
    its category maps to: ``input`` and ``output`` on the agent, ``tool_input``
    on the agent's local function tools.

    NOTE: the run-level ``pre_approval_tool_input_guardrails`` setting is not
    wired here — it lives on ``RunConfig``, which belongs to the run, not the
    agent. Declared tool-input guardrails still run before their tool executes;
    without that setting they run after the SDK emits a pending human-approval
    interruption rather than before it (``openai-agents`` 0.18.3,
    ``agents.run_internal.tool_execution``). Callers that need the earlier
    ordering pass their own ``RunConfig`` to ``Runner.run()``;
    ``BaseAgentRunner.execute()`` sets it automatically.

    Args:
        agent_name: Name of a previously registered agent.
        model_override: Use a different model than the one in the definition.

    Returns:
        A configured ``Agent`` ready for ``Runner.run()``.

    Raises:
        ValueError: If the agent name is not found in the registry.

    Example::

        from sinan_agentic_core import create_agent_from_registry
        from agents import Runner

        agent = create_agent_from_registry("my_agent")
        result = await Runner.run(agent, "Hello!")
    """
    agent_registry = get_agent_registry()
    tool_registry = get_tool_registry()
    guardrail_registry = get_guardrail_registry()

    agent_def = agent_registry.get(agent_name)
    if not agent_def:
        available = agent_registry.list_all()
        raise ValueError(f"Agent '{agent_name}' not found. Available: {available}")

    guardrails = guardrail_registry.resolve(agent_def.guardrails)
    tools = attach_tool_input_guardrails(
        tool_registry.get_tool_functions(agent_def.tools),
        guardrails.tool_input_guardrails,
    )

    return Agent(
        name=agent_def.name,
        instructions=agent_def.instructions,
        model=model_override or agent_def.model,
        tools=tools,
        input_guardrails=guardrails.input_guardrails,
        output_guardrails=guardrails.output_guardrails,
    )
