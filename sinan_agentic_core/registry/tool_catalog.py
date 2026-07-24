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

import copy
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

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

    # ``expose`` and ``annotations`` are the whole block, so an unknown key is a
    # typo — and because ``expose`` defaults to False, an accepted ``exposed:
    # true`` would leave the tool unexposed while reading as opted in. The
    # annotation names inside ``annotations`` are not gated here: this model
    # carries them as a free mapping, and ``MCPAnnotationsConfig``
    # (``mcp/yaml_schema.py``) is the typed schema that names them.
    model_config = ConfigDict(extra="forbid")

    expose: bool = False
    annotations: dict[str, Any] = {}


class ToolYamlEntry(BaseModel):
    """Resolved tool entry from tools.yaml.

    :meth:`ToolCatalog.get` builds this straight from the raw entry, so these
    fields are the whole of what a tool may declare in ``tools.yaml``.
    """

    # An unknown key is a typo or a field that belongs elsewhere (``name`` comes
    # from the mapping key, and the function binding lives on the decorator), and
    # this model is the only gate a tool entry passes through — so reject it
    # rather than drop it silently.
    model_config = ConfigDict(extra="forbid")

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

    A catalog is a fixed in-process view of the parsed YAML in both directions:
    the mapping it is given is copied on the way in, and every entry it resolves
    owns its values. So what it resolves never changes after construction, and a
    consumer editing a resolved entry cannot write back into the catalog.
    """

    def __init__(self, raw_tools: dict[str, dict[str, Any]]) -> None:
        """Build a catalog over already-parsed ``tools.yaml`` data.

        The mapping is copied, so a later edit to the caller's dict does not
        reach the catalog and two catalogs built from the same data never share
        one mapping. Each value is a tool block that nests further mutable
        values — the ``mcp`` block, its ``annotations`` — so the copy goes all
        the way down; a shallow copy would leave those nested edit paths open.

        Args:
            raw_tools: Raw tool entries, keyed by tool name.
        """
        self._raw_tools: dict[str, dict[str, Any]] = copy.deepcopy(raw_tools)

    def get(self, name: str) -> ToolYamlEntry:
        """Get a resolved tool entry by name.

        The returned entry owns everything it carries: editing it — including a
        value nested inside ``mcp.annotations`` — cannot change the catalog or
        another entry resolved from it.

        Raises:
            KeyError: If the tool is not in the catalog.
        """
        if name not in self._raw_tools:
            available = ", ".join(sorted(self._raw_tools.keys()))
            raise KeyError(f"Tool '{name}' not found in tools.yaml. " f"Available: {available}")
        # Copied on the way out for the same reason the constructor copies on the
        # way in. Pydantic rebuilds only the containers it has a declared type
        # for and stops at ``Any``: the values inside ``ToolMCPConfig.annotations``
        # are handed over as-is, so without this copy they would be the catalog's
        # own objects and editing a resolved entry would change what every later
        # ``get()`` returns.
        raw = copy.deepcopy(self._raw_tools[name])
        return ToolYamlEntry(**raw)

    def list_tools(self) -> list[str]:
        """List all tool names in the catalog."""
        return list(self._raw_tools.keys())

    def get_mcp_tools(self) -> list[str]:
        """List tool names that have ``mcp.expose: true`` in their config.

        The ``mcp`` block is resolved through :class:`ToolYamlEntry`, so an
        empty mapping is a declaration — ``mcp: {}`` opts in with
        ``ToolMCPConfig`` defaults. Only a missing key (or an explicitly null
        one) means the tool declares no MCP config, and ``expose`` is validated
        as a bool instead of read for truthiness.
        """
        result: list[str] = []
        for name in self._raw_tools:
            mcp = self.get(name).mcp
            if mcp is not None and mcp.expose:
                result.append(name)
        return result

    def enrich_registry(self, registry: ToolRegistry) -> None:
        """Patch registry ToolDefinitions with YAML metadata.

        For each tool in the catalog that exists in the registry, overwrites
        any non-empty YAML field onto a copy of the ToolDefinition and writes
        the patched record back through :meth:`ToolRegistry.register`. Tools in
        the catalog with no registered function are logged as warnings.

        YAML values always win over decorator values (YAML is the source of truth).
        Empty YAML fields do not overwrite existing decorator values.
        """
        for name, raw in self._raw_tools.items():
            # ``get_tool`` hands out a copy, so patching it and re-registering
            # the patched record is what makes enrichment stick — mutating the
            # returned copy alone would not reach the registry.
            tool_def = registry.get_tool(name)
            if tool_def is None:
                logger.warning(
                    "Tool '%s' in tools.yaml has no registered function — skipping",
                    name,
                )
                continue

            # ``raw`` is not deep-copied the way ``get()`` copies it: this entry
            # never leaves the method, and only its ``str`` fields are read off
            # it below.
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
            registry.register(tool_def)

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
            "PyYAML is required for tool catalog loading. " "Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        logger.warning("tools.yaml not found at %s, using empty catalog", path)
        return ToolCatalog(raw_tools={})

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return ToolCatalog(raw_tools=data.get("tools", {}))
