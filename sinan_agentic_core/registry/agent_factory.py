"""Agent Factory - Build Agent instances from the registry.

Resolves an AgentDefinition + its registered tools into a ready-to-use
OpenAI Agents SDK ``Agent`` instance.

Usage:
    from sinan_agentic_core import create_agent_from_registry

    agent = create_agent_from_registry("weather_assistant")
    result = await Runner.run(agent, "What's the weather?")
"""

import logging
from typing import Optional

from agents import Agent

from .agent_registry import get_agent_registry
from .tool_registry import get_tool_registry

logger = logging.getLogger(__name__)


def create_agent_from_registry(
    agent_name: str,
    model_override: Optional[str] = None,
) -> Agent:
    """Build an Agent instance from registry definitions.

    Looks up the AgentDefinition by name, resolves all its tool references
    through the ToolRegistry, and returns a fully configured Agent.

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

    agent_def = agent_registry.get(agent_name)
    if not agent_def:
        available = agent_registry.list_all()
        raise ValueError(
            f"Agent '{agent_name}' not found. Available: {available}"
        )

    tools = tool_registry.get_tool_functions(agent_def.tools)

    return Agent(
        name=agent_def.name,
        instructions=agent_def.instructions,
        model=model_override or agent_def.model,
        tools=tools,
    )
