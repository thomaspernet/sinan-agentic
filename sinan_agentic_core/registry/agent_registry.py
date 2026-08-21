"""Agent Registry - Centralized definition of all agents."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from agents import Agent

from ..core.capabilities import Capability
from ..core.model_retry import ModelRetryConfig
from ..core.tool_output_trim import ToolOutputTrimConfig


@dataclass
class AgentDefinition:
    """Schema for an agent.

    The definition owns its ``tools``, ``guardrails``, ``handoffs``,
    ``hosted_tools``, and ``capabilities`` lists: it is a fixed record of what
    was declared, so each is copied on construction rather than aliased to
    whoever supplied it.
    """

    name: str
    description: str  # Description of agent's purpose
    # Static string, or a sync/async callable resolved per run by the runner
    instructions: str | Callable[..., Any] | None = None
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
        """Validate the declaration, then take ownership of its list fields.

        ``default_factory=list`` only covers the omitted-argument case. A caller
        that passes its own list keeps it aliased into the definition, and the
        definition outlives that caller in the process-wide registry:
        :func:`create_agent_from_registry` re-reads ``tools`` and ``guardrails``
        on every build, and the runner re-reads ``capabilities`` on every run.
        Copying here means a later append by the caller cannot change which
        tools an agent exposes or which capabilities dispatch, and two
        definitions built from one list do not share it.

        Only the containers are copied. The elements stay shared on purpose —
        ``capabilities`` holds live per-run objects the runner clones and reads
        state back off, and ``hosted_tools`` holds SDK tool objects a caller
        matches against its own.
        """
        if self.instructions is None:
            raise ValueError(f"Agent {self.name} must have instructions")
        self.tools = list(self.tools)
        self.guardrails = list(self.guardrails)
        self.handoffs = list(self.handoffs)
        self.hosted_tools = list(self.hosted_tools)
        self.capabilities = list(self.capabilities)


@dataclass
class AgentRegistry:
    """Central registry of all agents in the system."""

    _agents: dict[str, AgentDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own the mapping the registry accumulates into.

        ``_agents`` is a dataclass constructor parameter, so
        ``AgentRegistry(_agents=existing)`` would alias the caller's dict and
        every later :meth:`register` would land in it. The shallow copy detaches
        the container while leaving the ``AgentDefinition`` values shared, so a
        caller still resolves the very object it registered.
        """
        self._agents = dict(self._agents)

    def register(self, agent_def: AgentDefinition) -> None:
        """Register an agent."""
        self._agents[agent_def.name] = agent_def

    def get(self, name: str) -> AgentDefinition | None:
        """Get a snapshot of the agent definition registered under *name*.

        Returns a copy, not the stored record. ``AgentDefinition`` carries five
        mutable list fields and the registry is process-wide, so a reader that
        appended to ``definition.tools`` would rewrite what every later lookup
        resolves — :func:`create_agent_from_registry` re-reads ``tools`` and
        ``guardrails`` on every build. :func:`dataclasses.replace` re-runs
        ``__post_init__``, which copies the containers, so the caller owns its
        lists while the elements stay shared (the same split ``__post_init__``
        draws on the inbound side).
        """
        definition = self._agents.get(name)
        if definition is None:
            return None
        return replace(definition)

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


def resolve_agent_definition(agent: Agent) -> AgentDefinition | None:
    """Get the definition registered under *agent*'s name, or None when unknown.

    A setting declared in ``agents.yaml`` that has no slot on the SDK's ``Agent``
    — a ``tool_output_trim`` policy, an ``invalid_output_recovery`` flag — cannot
    be read back off a built agent. Every run-level reader that holds only a
    built ``Agent`` resolves it here instead, so the correspondence between a
    built agent and its declaration is written once rather than once per setting.

    Args:
        agent: A built agent, from :func:`create_agent_from_registry` or
            assembled by hand. Only its name is read, so a hand-assembled agent
            resolves to the definition registered under the name it was given.

    Returns:
        The registered definition, or None when no agent is registered under
        that name. Callers decide what an absent declaration means for their own
        setting — the two differ, so this resolver does not choose for them.
    """
    return get_agent_registry().get(agent.name)
