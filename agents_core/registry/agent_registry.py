"""Agent Registry - Centralized definition of all agents."""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Union


@dataclass
class AgentDefinition:
    """Schema for an agent."""
    name: str
    description: str  # Description of agent's purpose
    #instructions_template: Optional[str] = None  # Static template (deprecated, use instructions)
    instructions: Optional[Union[str, Callable]] = None  # Static string or dynamic function
    # Optional fields (default to empty lists)
    tools: List[str] = field(default_factory=list)         # Tool names from registry
    guardrails: List[str] = field(default_factory=list)    # Guardrail names
    handoffs: List[str] = field(default_factory=list)
    hosted_tools: List[Any] = field(default_factory=list)  # OpenAI SDK hosted tools (WebSearchTool, etc.)
    output_dataclass: Optional[Any] = None  # Dataclass type for structured output
    model_settings_fn: Optional[Callable] = None  # Dynamic model settings function
    
    model: str = "gpt-4o-mini"
    requires_schema_injection: bool = False  # If True, inject {schema} dynamically
    knowledge_text: str = ""  # Domain knowledge from catalog (injected via domain_knowledge())
    as_tool_parameters: Optional[Any] = None  # Dataclass/Pydantic model for structured agent-as-tool input
    as_tool_max_turns: Optional[int] = None  # Max turns when running as sub-agent via as_tool()
    
    def __post_init__(self):
        """Ensure either instructions_template or instructions is provided."""
        # Support both old (instructions_template) and new (instructions) patterns
        if self.instructions is None:
            raise ValueError(f"Agent {self.name} must have either instructions or instructions_template")
        
        # If using new pattern, copy to old field for backward compatibility
        #if self.instructions is not None:
        #    if isinstance(self.instructions, str):
        #        self.instructions_template = self.instructions


@dataclass
class AgentRegistry:
    """Central registry of all agents in the system."""
    
    _agents: dict[str, AgentDefinition] = field(default_factory=dict)
    
    def register(self, agent_def: AgentDefinition):
        """Register an agent."""
        self._agents[agent_def.name] = agent_def
    
    def get(self, name: str) -> Optional[AgentDefinition]:
        """Get agent definition by name."""
        return self._agents.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())


# Global agent registry
_global_agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _global_agent_registry


def register_agent(agent_def: AgentDefinition):
    """Register an agent in the global registry."""
    _global_agent_registry.register(agent_def)
