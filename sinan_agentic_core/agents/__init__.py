"""Agents package.

Define your agents here by creating AgentDefinition instances and registering them.

Example:
    from sinan_agentic_core.registry import AgentDefinition, register_agent
    
    my_agent = AgentDefinition(
        name="analyzer",
        description="Analyzes data and provides insights",
        instructions="You are a data analyst...",
        tools=["fetch_data", "analyze"],
        model="gpt-4o-mini"
    )
    register_agent(my_agent)
"""

__all__ = []
