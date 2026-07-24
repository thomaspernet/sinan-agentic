"""Guardrail Registry - Centralized definition of all available guardrails."""

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from agents import FunctionTool, InputGuardrail, OutputGuardrail, ToolInputGuardrail

logger = logging.getLogger(__name__)


class GuardrailCategory(str, Enum):
    """The SDK slot a guardrail is wired into.

    - ``INPUT`` -> ``Agent(input_guardrails=...)``: validates the run input
      before the agent starts.
    - ``OUTPUT`` -> ``Agent(output_guardrails=...)``: validates the final output
      after the agent finishes.
    - ``TOOL_INPUT`` -> ``FunctionTool(tool_input_guardrails=...)``: validates a
      tool call *before* the tool runs, and before the SDK emits a pending
      human-approval interruption. A rejecting guardrail returns its message as
      the tool output instead of executing the tool.
    """

    INPUT = "input"
    OUTPUT = "output"
    TOOL_INPUT = "tool_input"


@dataclass
class GuardrailDefinition:
    """Schema for a single guardrail."""

    name: str
    description: str
    function: Callable[..., Any]
    category: GuardrailCategory

    def __post_init__(self) -> None:
        """Coerce the category to :class:`GuardrailCategory` and reject unknowns.

        This is the only place a category is validated. Callers that hold a raw
        string (a config loader, an untyped consumer) may pass it straight in —
        a valid one is normalized to the enum, an invalid one raises here.
        """
        try:
            self.category = GuardrailCategory(self.category)
        except ValueError as exc:
            valid = ", ".join(c.value for c in GuardrailCategory)
            raise ValueError(
                f"Guardrail '{self.name}' has unknown category '{self.category}'. "
                f"Valid categories: {valid}"
            ) from exc


@dataclass(frozen=True)
class ResolvedGuardrails:
    """Guardrails split into the SDK slots their categories map to.

    A fixed record of one resolution: each of the three lists is copied on
    construction, so a caller that keeps and mutates the list it passed in
    cannot reach back into the bundle, and two bundles built from one list do
    not share it. Only the outer list is detached — the elements are SDK
    guardrail objects a caller matches by identity, the same split
    :class:`AgentDefinition` draws for ``hosted_tools``.
    """

    input_guardrails: list[InputGuardrail[Any]] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail[Any]] = field(default_factory=list)
    tool_input_guardrails: list[ToolInputGuardrail[Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # frozen=True forbids ``self.x = ...``, so the copies go in via
        # ``object.__setattr__``. ``default_factory`` only covers the omitted
        # argument; a supplied list is aliased without this.
        object.__setattr__(self, "input_guardrails", list(self.input_guardrails))
        object.__setattr__(self, "output_guardrails", list(self.output_guardrails))
        object.__setattr__(self, "tool_input_guardrails", list(self.tool_input_guardrails))


@dataclass
class GuardrailRegistry:
    """Central registry of all guardrails available to agents.

    Dataclass-driven design makes it easy to add new guardrails.
    """

    _guardrails: dict[str, GuardrailDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own the mapping the registry accumulates into.

        ``_guardrails`` is a dataclass constructor parameter, so
        ``GuardrailRegistry(_guardrails=existing)`` would alias the caller's dict
        and every later :meth:`register` would land in it. The shallow copy
        detaches the container while leaving the ``GuardrailDefinition`` values
        shared, so a caller still resolves the very object it registered.
        """
        self._guardrails = dict(self._guardrails)

    def register(self, guardrail_def: GuardrailDefinition) -> None:
        """Register a new guardrail."""
        self._guardrails[guardrail_def.name] = guardrail_def

    def get_guardrail(self, name: str) -> GuardrailDefinition | None:
        """Get a snapshot of the guardrail registered under *name*.

        Returns a copy, not the stored record. ``GuardrailDefinition`` carries
        no nested container — the exposure is field reassignment
        (``description``, ``category``) on a process-wide object, not a nested
        edit — but the registry answers the read-boundary question the same way
        its two siblings do: hand out a record, never the live definition.
        Unlike :class:`ToolRegistry`, no enrichment path patches these records,
        so nothing depends on them being live.
        """
        definition = self._guardrails.get(name)
        if definition is None:
            return None
        return replace(definition)

    def get_guardrails_by_category(self, category: GuardrailCategory) -> list[GuardrailDefinition]:
        """Get a snapshot of every guardrail in *category*."""
        return [replace(g) for g in self._guardrails.values() if g.category == category]

    def get_guardrail_functions(self, guardrail_names: list[str]) -> list[Callable[..., Any]]:
        """Get actual function objects for given guardrail names."""
        return [
            self._guardrails[name].function for name in guardrail_names if name in self._guardrails
        ]

    def get_all_functions(self) -> dict[str, Callable[..., Any]]:
        """Get all registered guardrail functions as a mapping."""
        return {name: gdef.function for name, gdef in self._guardrails.items()}

    def list_names(self) -> list[str]:
        """List all registered guardrail names."""
        return list(self._guardrails)

    def resolve(self, guardrail_names: list[str]) -> ResolvedGuardrails:
        """Split named guardrails into the SDK slots their categories map to.

        Args:
            guardrail_names: Guardrail names from an agent definition.

        Returns:
            The named guardrails bucketed by category. Unknown names are logged
            and skipped so one bad reference does not break agent creation.
        """
        buckets: dict[GuardrailCategory, list[Any]] = {c: [] for c in GuardrailCategory}

        for name in guardrail_names:
            definition = self._guardrails.get(name)
            if definition is None:
                logger.warning("Guardrail '%s' not found in registry", name)
                continue
            buckets[definition.category].append(definition.function)

        return ResolvedGuardrails(
            input_guardrails=buckets[GuardrailCategory.INPUT],
            output_guardrails=buckets[GuardrailCategory.OUTPUT],
            tool_input_guardrails=buckets[GuardrailCategory.TOOL_INPUT],
        )

    def has_category(self, guardrail_names: list[str], category: GuardrailCategory) -> bool:
        """Check whether any of *guardrail_names* is registered under *category*."""
        for name in guardrail_names:
            definition = self._guardrails.get(name)
            if definition is not None and definition.category == category:
                return True
        return False


def attach_tool_input_guardrails(
    tools: list[Any], guardrails: list[ToolInputGuardrail[Any]]
) -> list[Any]:
    """Attach tool-input guardrails to every local function tool.

    Registry tools are shared across agents, so each one is copied before its
    guardrails are set — mutating in place would leak one agent's guardrails
    into every other agent using the same tool. Non-function tools (hosted
    tools) pass through untouched: the SDK runs tool-input guardrails for local
    function tools only.

    Args:
        tools: Tools already resolved for the agent
        guardrails: Tool-input guardrails to run before each tool executes

    Returns:
        The tools list with guardrails attached to its function tools
    """
    if not guardrails:
        return tools

    guarded_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            guarded_tools.append(tool)
            continue
        guarded = copy.copy(tool)
        guarded.tool_input_guardrails = [*(tool.tool_input_guardrails or []), *guardrails]
        guarded_tools.append(guarded)

    return guarded_tools


# Global guardrail registry instance
_global_registry = GuardrailRegistry()


def get_guardrail_registry() -> GuardrailRegistry:
    """Get the global guardrail registry."""
    return _global_registry


def register_guardrail(
    name: str,
    description: str,
    category: GuardrailCategory,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a guardrail.

    Usage:
        @register_guardrail(
            name="validate_cypher_syntax",
            description="Validate Cypher query syntax",
            category=GuardrailCategory.INPUT,
        )
        @input_guardrail
        async def validate_cypher_syntax(ctx, value):
            ...

    Tool-input guardrails run before the tool executes, so a rejected call
    never reaches the tool or a human approver:

        @register_guardrail(
            name="block_destructive_cypher",
            description="Reject write clauses in read-only queries",
            category=GuardrailCategory.TOOL_INPUT,
        )
        @tool_input_guardrail
        def block_destructive_cypher(data):
            ...

    Raises:
        ValueError: If *category* is not a known :class:`GuardrailCategory`.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        guardrail_def = GuardrailDefinition(
            name=name,
            description=description,
            function=func,
            category=category,
        )
        _global_registry.register(guardrail_def)
        return func

    return decorator
