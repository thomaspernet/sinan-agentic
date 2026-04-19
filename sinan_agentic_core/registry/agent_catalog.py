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

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public resolved type
# ---------------------------------------------------------------------------


class TurnBudgetConfig(BaseModel):
    """Turn budget configuration from agents.yaml."""

    default_turns: int = 10
    reminder_at: int = 2
    max_extensions: int = 3
    extension_size: int = 5


class AgentYamlEntry(BaseModel):
    """Resolved agent entry — tools are plain strings."""

    model: str
    description: str
    tools: list[str] = []
    knowledge_text: str = ""
    max_turns: int | None = None
    turn_budget: TurnBudgetConfig | None = None
    error_recovery: bool = True
    effort: str | None = None
    tool_rules: dict[str, dict[str, Any]] = {}

    def build_turn_budget(self) -> "TurnBudget | None":
        """Create a TurnBudget from config, or None if not configured."""
        if self.turn_budget is None:
            return None
        from ..core.turn_budget import TurnBudget

        return TurnBudget(
            default_turns=self.turn_budget.default_turns,
            reminder_at=self.turn_budget.reminder_at,
            max_extensions=self.turn_budget.max_extensions,
            extension_size=self.turn_budget.extension_size,
            absolute_max=self.max_turns or 25,
        )

    def build_error_recovery(self) -> "ToolErrorRecovery | None":
        """Create a ToolErrorRecovery if enabled, or None if disabled."""
        if not self.error_recovery:
            return None
        from ..core.tool_error_recovery import ToolErrorRecovery
        from ..registry import get_tool_registry

        return ToolErrorRecovery(tool_registry=get_tool_registry())


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
                        f"Tool group '{group_name}' not found. "
                        f"Available: {available}"
                    )
                resolved.extend(tool_groups[group_name])
            elif "tool" in item:
                if _check_condition(item.get("when"), config):
                    resolved.append(item["tool"])
    return resolved


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class AgentCatalog:
    """Agent catalog loaded from ``agents.yaml``.

    Holds raw YAML data and resolves tool groups / conditions on ``get()``.
    Knowledge scopes are pre-loaded from YAML files and cached.
    Also stores ``mcp_servers`` definitions for MCP server building.
    """

    def __init__(
        self,
        tool_groups: dict[str, list[str]],
        raw_agents: dict[str, dict[str, Any]],
        knowledge: dict[str, str] | None = None,
        raw_mcp_servers: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._tool_groups = tool_groups
        self._raw_agents = raw_agents
        self._knowledge: dict[str, str] = knowledge or {}
        self._raw_mcp_servers: dict[str, dict[str, Any]] = raw_mcp_servers or {}

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
            raise KeyError(
                f"Agent '{name}' not found in agents.yaml. "
                f"Available: {available}"
            )
        raw = self._raw_agents[name]
        raw_budget = raw.get("turn_budget")
        budget_cfg = TurnBudgetConfig(**raw_budget) if raw_budget else None

        return AgentYamlEntry(
            model=raw["model"],
            description=raw["description"],
            tools=_resolve_tools(
                raw.get("tools", []), self._tool_groups, config
            ),
            knowledge_text=_resolve_knowledge(
                raw.get("knowledge", []), self._knowledge
            ),
            max_turns=raw.get("max_turns"),
            turn_budget=budget_cfg,
            error_recovery=raw.get("error_recovery", True),
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
        """
        from ..mcp.yaml_schema import MCPServerConfig, MCPResourceConfig, MCPPromptConfig

        if name not in self._raw_mcp_servers:
            available = ", ".join(sorted(self._raw_mcp_servers.keys()))
            raise KeyError(
                f"MCP server '{name}' not found in agents.yaml. "
                f"Available: {available}"
            )

        raw = self._raw_mcp_servers[name]

        # Resolve tools (expand groups, evaluate conditions)
        tools = _resolve_tools(
            raw.get("tools", []), self._tool_groups, config
        )
        write_tools = _resolve_tools(
            raw.get("write_tools", []), self._tool_groups, config
        )

        # Parse resources
        resources = [
            MCPResourceConfig(**r)
            for r in raw.get("resources", [])
        ]

        # Parse prompts
        prompts = [
            MCPPromptConfig(**p)
            for p in raw.get("prompts", [])
        ]

        return MCPServerConfig(
            name=name,
            description=raw.get("description", ""),
            tools=tools,
            write_tools=write_tools,
            resources=resources,
            prompts=prompts,
        )

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
            "PyYAML is required for agent catalog loading. "
            "Install it with: pip install pyyaml"
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
