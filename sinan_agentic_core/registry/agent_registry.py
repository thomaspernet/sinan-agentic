"""Agent Registry - Centralized definition of all agents."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.capabilities import Capability
from ..core.model_retry import ModelRetryConfig
from ..core.tool_output_trim import ToolOutputTrimConfig


@dataclass
class AgentDefinition:
    """Schema for an agent."""

    name: str
    description: str  # Description of agent's purpose
    instructions: str | Callable[..., Any] | None = None  # Static string or dynamic function
    # Optional fields (default to empty lists)
    tools: list[str] = field(default_factory=list)  # Tool names from registry
    guardrails: list[str] = field(default_factory=list)  # Guardrail names
    handoffs: list[str] = field(default_factory=list)
    hosted_tools: list[Any] = field(
        default_factory=list
    )  # OpenAI SDK hosted tools (WebSearchTool, etc.)
    output_dataclass: Any | None = None  # Dataclass type for structured output
    model_settings_fn: Callable[..., Any] | None = None  # Dynamic model settings function
    capabilities: list[Capability] = field(default_factory=list)  # Pluggable agent behaviors

    model: str = "gpt-4o-mini"
    # Re-parse a final message whose structured payload is wrapped in prose or
    # a code fence instead of failing the run. Never fabricates a value.
    invalid_output_recovery: bool = True
    # Retry transient model-API failures inside the SDK instead of failing the
    # run. Off unless declared — retries cost latency and duplicate requests.
    model_retry: ModelRetryConfig | None = None
    # Shrink oversized tool outputs from older turns before each model call, so
    # the run overflows less often. Off unless declared — trimming removes
    # content the model would otherwise have seen.
    tool_output_trim: ToolOutputTrimConfig | None = None
    requires_schema_injection: bool = False  # If True, inject {schema} dynamically
    knowledge_text: str = ""  # Domain knowledge from catalog (injected via domain_knowledge())
    as_tool_parameters: Any | None = (
        None  # Dataclass/Pydantic model for structured agent-as-tool input
    )
    as_tool_max_turns: int | None = None  # Max turns when running as sub-agent via as_tool()
    as_tool_turn_budget: Any | None = None  # TurnBudget instance for sub-agent budget management

    def __post_init__(self) -> None:
        """Ensure instructions is provided."""
        if self.instructions is None:
            raise ValueError(f"Agent {self.name} must have instructions")


@dataclass
class AgentRegistry:
    """Central registry of all agents in the system."""

    _agents: dict[str, AgentDefinition] = field(default_factory=dict)

    def register(self, agent_def: AgentDefinition) -> None:
        """Register an agent."""
        self._agents[agent_def.name] = agent_def

    def get(self, name: str) -> AgentDefinition | None:
        """Get agent definition by name."""
        return self._agents.get(name)

    def list_all(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())


# Global agent registry
_global_agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _global_agent_registry


def register_agent(agent_def: AgentDefinition) -> None:
    """Register an agent in the global registry."""
    _global_agent_registry.register(agent_def)
