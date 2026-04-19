"""Tool YAML catalog — load tool metadata from a YAML file.

Static tool metadata (description, category, parameters, recovery hints)
lives in YAML. The function binding stays in Python via @register_tool.

Usage::

    from sinan_agentic_core import load_tool_catalog, get_tool_registry

    catalog = load_tool_catalog("tools.yaml")
    catalog.enrich_registry(get_tool_registry())

    # Or query metadata directly
    entry = catalog.get("paper_lookup")
    entry.description  # "Find and resolve papers..."
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public resolved type
# ---------------------------------------------------------------------------


class ToolMCPConfig(BaseModel):
    """Per-tool MCP config from the ``mcp`` section in tools.yaml.

    Example YAML::

        discover:
          description: ...
          mcp:
            expose: true
            annotations:
              readOnlyHint: true
    """

    expose: bool = False
    annotations: dict[str, Any] = {}


class ToolYamlEntry(BaseModel):
    """Resolved tool entry from tools.yaml."""

    description: str = ""
    category: str = ""
    parameters_description: str = ""
    returns_description: str = ""
    recovery_hint: str = ""
    mcp: ToolMCPConfig | None = None


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class ToolCatalog:
    """Tool catalog loaded from ``tools.yaml``.

    Holds raw YAML data and provides:
    - ``get(name)`` to resolve a single tool's metadata
    - ``enrich_registry(registry)`` to patch ToolDefinitions with YAML metadata
    - ``list_tools()`` to list all tool names
    """

    def __init__(self, raw_tools: dict[str, dict[str, Any]]) -> None:
        self._raw_tools = raw_tools

    def get(self, name: str) -> ToolYamlEntry:
        """Get a resolved tool entry by name.

        Raises:
            KeyError: If the tool is not in the catalog.
        """
        if name not in self._raw_tools:
            available = ", ".join(sorted(self._raw_tools.keys()))
            raise KeyError(
                f"Tool '{name}' not found in tools.yaml. "
                f"Available: {available}"
            )
        raw = self._raw_tools[name]
        return ToolYamlEntry(**raw)

    def list_tools(self) -> list[str]:
        """List all tool names in the catalog."""
        return list(self._raw_tools.keys())

    def get_mcp_tools(self) -> list[str]:
        """List tool names that have ``mcp.expose: true`` in their config."""
        result: list[str] = []
        for name, raw in self._raw_tools.items():
            mcp_raw = raw.get("mcp")
            if mcp_raw and mcp_raw.get("expose", False):
                result.append(name)
        return result

    def enrich_registry(self, registry: ToolRegistry) -> None:
        """Patch registry ToolDefinitions with YAML metadata.

        For each tool in the catalog that exists in the registry, overwrites
        any non-empty YAML field onto the ToolDefinition. Tools in the catalog
        with no registered function are logged as warnings.

        YAML values always win over decorator values (YAML is the source of truth).
        Empty YAML fields do not overwrite existing decorator values.
        """
        for name, raw in self._raw_tools.items():
            tool_def = registry.get_tool(name)
            if tool_def is None:
                logger.warning(
                    "Tool '%s' in tools.yaml has no registered function — skipping",
                    name,
                )
                continue

            entry = ToolYamlEntry(**raw)
            if entry.description:
                tool_def.description = entry.description
            if entry.category:
                tool_def.category = entry.category
            if entry.parameters_description:
                tool_def.parameters_description = entry.parameters_description
            if entry.returns_description:
                tool_def.returns_description = entry.returns_description
            if entry.recovery_hint:
                tool_def.recovery_hint = entry.recovery_hint

        logger.info(
            "Enriched tool registry from tools.yaml (%d tool definitions)",
            len(self._raw_tools),
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_tool_catalog(path: str | Path) -> ToolCatalog:
    """Load tool catalog from a YAML file.

    Args:
        path: Path to the tools.yaml file.

    Returns:
        ToolCatalog with all tool metadata.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for tool catalog loading. "
            "Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        logger.warning("tools.yaml not found at %s, using empty catalog", path)
        return ToolCatalog(raw_tools={})

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return ToolCatalog(raw_tools=data.get("tools", {}))
