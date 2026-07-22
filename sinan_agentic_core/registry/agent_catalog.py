"""Agent YAML catalog — load agent definitions from a YAML file.

Static agent config (model, description, tools) lives in YAML.
Dynamic parts (instructions, output_dataclass, hosted_tools) stay in Python.

Features:
  - tool_groups: reusable named tool sets, referenced via ``group: name``
  - Conditional tools: ``tool: name`` + ``when: dot.path`` (resolved against config)
  - Agent-level conditions: ``when: dot.path`` on the agent entry
  - Knowledge files: optional ``knowledge/`` directory with per-scope YAML files.
    Agents reference scopes via ``knowledge: [global, chatbot]`` in their entry.
    Content is loaded once at startup and cached on ``AgentYamlEntry.knowledge_text``.

Usage::

    from sinan_agentic_core import load_agent_catalog

    catalog = load_agent_catalog("agents.yaml", knowledge_dir="knowledge/")

    # Resolve tools (expand groups, evaluate conditions)
    cfg = catalog.get("chatbot_agent", config=my_config)
    cfg.model   # "reasoning"
    cfg.tools   # ["think", "discover", ...] — groups expanded, conditions evaluated
    cfg.knowledge_text  # concatenated knowledge from scopes listed in the agent entry

    # Check agent-level condition
    if catalog.is_enabled("web_search_agent", config=my_config):
        ...
"""

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from ..core.capabilities import Capability
from ..core.model_retry import ModelRetryConfig
from ..core.tool_error_recovery import ToolErrorRecovery, build_tool_error_recovery
from ..core.tool_output_trim import ToolOutputTrimConfig
from ..core.turn_budget import TurnBudget, TurnBudgetConfig, build_turn_budget
from .capability_registry import (
    CapabilityNotFoundError,
    get_capability_registry,
)

if TYPE_CHECKING:
    from ..mcp.yaml_schema import MCPServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public resolved type
# ---------------------------------------------------------------------------


class CapabilityRef(BaseModel):
    """Reference to a registered capability with optional config.

    Parsed from the explicit ``capabilities:`` list in ``agents.yaml``::

        capabilities:
          - name: turn_budget
            config: { default_turns: 10 }
          - name: my_custom_capability
    """

    name: str
    config: dict[str, Any] = {}


class AgentYamlEntry(BaseModel):
    """Resolved agent entry — tools are plain strings."""

    model: str
    description: str
    tools: list[str] = []
    guardrails: list[str] = []
    knowledge_text: str = ""
    max_turns: int | None = None
    turn_budget: TurnBudgetConfig | None = None
    error_recovery: bool = True
    invalid_output_recovery: bool = True
    model_retry: ModelRetryConfig | None = None
    tool_output_trim: ToolOutputTrimConfig | None = None
    capabilities: list[CapabilityRef] = []
    effort: str | None = None
    tool_rules: dict[str, dict[str, Any]] = {}

    def build_turn_budget(self) -> TurnBudget | None:
        """Create a TurnBudget from config, or None if not configured.

        The translation itself lives beside :class:`TurnBudget`; this entry only
        supplies the two things it holds — the declared budget and the agent's
        ``max_turns`` ceiling.
        """
        return build_turn_budget(self.turn_budget, self.max_turns)

    def build_error_recovery(self) -> ToolErrorRecovery | None:
        """Create a ToolErrorRecovery if enabled, or None if disabled.

        The translation itself lives beside :class:`ToolErrorRecovery`; this
        entry only supplies the one thing it holds — whether the agent declared
        recovery on.
        """
        return build_tool_error_recovery(self.error_recovery)

    def build_capabilities(self) -> list[Capability]:
        """Build all capabilities for this agent from YAML.

        Combines the built-in shorthand keys (``turn_budget``,
        ``error_recovery``) with any entries from the explicit
        ``capabilities:`` list, in that order. The list form goes through
        :class:`CapabilityRegistry` so user-registered capabilities work
        identically to built-ins.
        """
        built: list[Capability] = []

        budget = self.build_turn_budget()
        if budget is not None:
            built.append(budget)

        recovery = self.build_error_recovery()
        if recovery is not None:
            built.append(recovery)

        registry = get_capability_registry()
        for ref in self.capabilities:
            built.append(registry.build(ref.name, ref.config))
        return built


# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------


def _resolve_dot_path(obj: object, path: str) -> Any:
    """Navigate a dot-separated attribute path on *obj*.

    Returns ``None`` when any segment is missing.
    """
    current: Any = obj
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError:
            return None
    return current


def _check_condition(when: str | None, config: object | None) -> bool:
    """Evaluate a ``when`` condition against *config*."""
    if not when:
        return True
    if config is None:
        return False
    return bool(_resolve_dot_path(config, when))


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------


def _resolve_tools(
    raw_tools: list[Any],
    tool_groups: dict[str, list[str]],
    config: object | None,
) -> list[str]:
    """Resolve mixed tool entries into a plain string list.

    Supported entry formats:
      - ``"tool_name"`` — included as-is
      - ``{group: "group_name"}`` — expanded from *tool_groups*
      - ``{tool: "tool_name", when: "dot.path"}`` — included if condition is truthy
    """
    resolved: list[str] = []
    for item in raw_tools:
        if isinstance(item, str):
            resolved.append(item)
        elif isinstance(item, dict):
            if "group" in item:
                group_name = item["group"]
                if group_name not in tool_groups:
                    available = ", ".join(sorted(tool_groups.keys()))
                    raise KeyError(
                        f"Tool group '{group_name}' not found. " f"Available: {available}"
                    )
                resolved.extend(tool_groups[group_name])
            elif "tool" in item:
                if _check_condition(item.get("when"), config):
                    resolved.append(item["tool"])
    return resolved


# ---------------------------------------------------------------------------
# Capability resolution
# ---------------------------------------------------------------------------


def _parse_capabilities(agent_name: str, raw: Any) -> list[CapabilityRef]:
    """Parse the ``capabilities:`` list on an agent entry and validate names.

    Accepts the explicit list form documented on :class:`CapabilityRef`.
    Each entry must be either a string (name only) or a mapping with a
    ``name`` key and optional ``config`` mapping. Unknown capability names
    raise :class:`CapabilityNotFoundError`.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        raise TypeError(
            f"Agent '{agent_name}' has invalid 'capabilities' — "
            f"expected a list, got {type(raw).__name__}."
        )

    registry = get_capability_registry()
    refs: list[CapabilityRef] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            ref = CapabilityRef(name=item)
        elif isinstance(item, dict):
            if "name" not in item:
                raise ValueError(
                    f"Agent '{agent_name}' capabilities[{index}] is missing "
                    f"the required 'name' key."
                )
            ref = CapabilityRef(name=item["name"], config=item.get("config", {}) or {})
        else:
            raise TypeError(
                f"Agent '{agent_name}' capabilities[{index}] must be a "
                f"string or mapping, got {type(item).__name__}."
            )

        if not registry.is_registered(ref.name):
            available = ", ".join(registry.list_names()) or "<none>"
            raise CapabilityNotFoundError(
                f"Agent '{agent_name}' references unknown capability "
                f"'{ref.name}'. Available: {available}"
            )
        refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# Optional config blocks
# ---------------------------------------------------------------------------

_ConfigT = TypeVar("_ConfigT", bound=BaseModel)


def _parse_optional_block(
    raw: dict[str, Any],
    key: str,
    model: type[_ConfigT],
) -> _ConfigT | None:
    """Build *model* from an optional config block on an agent entry.

    Every optional block is fully defaulted, so an empty mapping is a
    declaration — ``turn_budget: {}`` opts in with defaults. Only a missing
    key (or an explicitly null one) means the agent opts out.
    """
    block = raw.get(key)
    if block is None:
        return None
    return model(**block)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class AgentCatalog:
    """Agent catalog loaded from ``agents.yaml``.

    Holds raw YAML data and resolves tool groups / conditions on ``get()``.
    Knowledge scopes are pre-loaded from YAML files and cached.
    Also stores ``mcp_servers`` definitions for MCP server building.

    A catalog is a fixed in-process view of the parsed YAML: every mapping it
    is given is copied on the way in, so what it resolves never changes after
    construction.
    """

    def __init__(
        self,
        tool_groups: dict[str, list[str]],
        raw_agents: dict[str, dict[str, Any]],
        knowledge: dict[str, str] | None = None,
        raw_mcp_servers: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Build a catalog over already-parsed YAML data.

        Every mapping is copied, so a later edit to a caller's dict does not
        reach the catalog and two catalogs built from the same data never share
        one mapping. ``knowledge`` holds plain strings, which a shallow copy
        already detaches; the other three nest a mutable value — a group's tool
        list, an agent block, an MCP server block — so they are copied all the
        way down.

        Args:
            tool_groups: Named tool sets, referenced via ``group: name``.
            raw_agents: Raw agent entries, keyed by agent name.
            knowledge: Pre-loaded knowledge text, keyed by scope name.
            raw_mcp_servers: Raw ``mcp_servers`` blocks, keyed by server name.
        """
        self._tool_groups: dict[str, list[str]] = copy.deepcopy(tool_groups)
        self._raw_agents: dict[str, dict[str, Any]] = copy.deepcopy(raw_agents)
        self._knowledge: dict[str, str] = dict(knowledge or {})
        self._raw_mcp_servers: dict[str, dict[str, Any]] = copy.deepcopy(raw_mcp_servers or {})

    def get(
        self,
        name: str,
        config: object | None = None,
    ) -> AgentYamlEntry:
        """Get a resolved agent entry.

        Groups are expanded and conditional tools are evaluated against
        *config*.  If *config* is ``None``, conditional tools are skipped.
        Knowledge scopes listed in the agent's ``knowledge`` key are
        concatenated into ``knowledge_text``.
        """
        if name not in self._raw_agents:
            available = ", ".join(sorted(self._raw_agents.keys()))
            raise KeyError(f"Agent '{name}' not found in agents.yaml. " f"Available: {available}")
        raw = self._raw_agents[name]
        budget_cfg = _parse_optional_block(raw, "turn_budget", TurnBudgetConfig)
        retry_cfg = _parse_optional_block(raw, "model_retry", ModelRetryConfig)
        trim_cfg = _parse_optional_block(raw, "tool_output_trim", ToolOutputTrimConfig)

        capabilities = _parse_capabilities(name, raw.get("capabilities", []))

        return AgentYamlEntry(
            model=raw["model"],
            description=raw["description"],
            tools=_resolve_tools(raw.get("tools", []), self._tool_groups, config),
            guardrails=raw.get("guardrails", []),
            knowledge_text=_resolve_knowledge(raw.get("knowledge", []), self._knowledge),
            max_turns=raw.get("max_turns"),
            turn_budget=budget_cfg,
            error_recovery=raw.get("error_recovery", True),
            invalid_output_recovery=raw.get("invalid_output_recovery", True),
            model_retry=retry_cfg,
            tool_output_trim=trim_cfg,
            capabilities=capabilities,
            tool_rules=raw.get("tool_rules", {}),
            effort=raw.get("effort"),
        )

    def is_enabled(
        self,
        name: str,
        config: object | None = None,
    ) -> bool:
        """Check if an agent passes its ``when`` condition.

        Returns ``True`` when the agent has no ``when`` clause.
        Returns ``False`` when the agent is not in the catalog.
        """
        if name not in self._raw_agents:
            return False
        when = self._raw_agents[name].get("when")
        if not when:
            return True
        return _check_condition(when, config)

    def list_agents(self) -> list[str]:
        """List all agent names in the catalog."""
        return list(self._raw_agents.keys())

    def get_mcp_server(
        self,
        name: str,
        config: object | None = None,
    ) -> "MCPServerConfig":
        """Get a resolved MCP server definition.

        Tool groups are expanded and conditional tools evaluated, same as
        agent tool resolution.

        Raises:
            KeyError: If the MCP server is not defined in agents.yaml.
            ValidationError: If the block declares a key the model does not
                carry, or a value of the wrong shape.
        """
        from ..mcp.yaml_schema import MCPServerConfig

        if name not in self._raw_mcp_servers:
            available = ", ".join(sorted(self._raw_mcp_servers.keys()))
            raise KeyError(
                f"MCP server '{name}' not found in agents.yaml. " f"Available: {available}"
            )

        raw = self._raw_mcp_servers[name]

        # The whole raw block is forwarded so an unrecognized key reaches the
        # model's extra="forbid" gate instead of being dropped here. Only the two
        # tool lists need resolving first (groups expanded, conditions evaluated);
        # ``resources`` and ``prompts`` are validated by the model's own fields,
        # and ``name`` comes from the mapping key rather than the block.
        resolved = dict(raw)
        resolved.update(
            name=name,
            tools=_resolve_tools(raw.get("tools", []), self._tool_groups, config),
            write_tools=_resolve_tools(raw.get("write_tools", []), self._tool_groups, config),
        )
        return MCPServerConfig(**resolved)

    def list_mcp_servers(self) -> list[str]:
        """List all MCP server names in the catalog."""
        return list(self._raw_mcp_servers.keys())


# ---------------------------------------------------------------------------
# Knowledge resolution
# ---------------------------------------------------------------------------


def _resolve_knowledge(
    scopes: list[str],
    knowledge: dict[str, str],
) -> str:
    """Concatenate knowledge content for the listed scopes."""
    if not scopes:
        return ""
    parts: list[str] = []
    for scope in scopes:
        text = knowledge.get(scope, "")
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts)


def _load_knowledge_dir(knowledge_dir: Path) -> dict[str, str]:
    """Load all YAML files from a knowledge directory.

    Each file's stem becomes the scope name (e.g., ``global.yaml`` -> ``"global"``).
    Each file must have a ``content`` key with the knowledge text.
    """
    try:
        import yaml
    except ImportError:
        return {}

    if not knowledge_dir.is_dir():
        return {}

    knowledge: dict[str, str] = {}
    for yaml_file in sorted(knowledge_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        content = data.get("content", "")
        if content:
            knowledge[yaml_file.stem] = content
    return knowledge


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_agent_catalog(
    path: str | Path,
    knowledge_dir: str | Path | None = None,
) -> AgentCatalog:
    """Load agent catalog from a YAML file.

    Args:
        path: Path to the agents.yaml file.
        knowledge_dir: Optional path to a directory containing knowledge
            YAML files (one per scope). Each file must have a ``content``
            key. If relative, resolved against the agents.yaml parent dir.

    Returns:
        AgentCatalog with all agent entries and pre-loaded knowledge.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for agent catalog loading. " "Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        logger.warning("agents.yaml not found at %s, using empty catalog", path)
        return AgentCatalog(tool_groups={}, raw_agents={})

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    knowledge: dict[str, str] = {}
    if knowledge_dir is not None:
        kdir = Path(knowledge_dir)
        if not kdir.is_absolute():
            kdir = path.parent / kdir
        knowledge = _load_knowledge_dir(kdir)

    return AgentCatalog(
        tool_groups=data.get("tool_groups", {}),
        raw_agents=data.get("agents", {}),
        knowledge=knowledge,
        raw_mcp_servers=data.get("mcp_servers", {}),
    )
