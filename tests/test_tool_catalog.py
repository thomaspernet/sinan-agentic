"""Tests for ToolCatalog — YAML-based tool metadata loading and registry enrichment."""

import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sinan_agentic_core.registry.tool_catalog import (
    ToolCatalog,
    ToolYamlEntry,
    load_tool_catalog,
)
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry

# ---------------------------------------------------------------------------
# ToolYamlEntry
# ---------------------------------------------------------------------------


class TestToolYamlEntry:
    def test_defaults(self):
        entry = ToolYamlEntry()
        assert entry.description == ""
        assert entry.category == ""
        assert entry.parameters_description == ""
        assert entry.returns_description == ""
        assert entry.recovery_hint == ""

    def test_full_construction(self):
        entry = ToolYamlEntry(
            description="Search papers",
            category="research",
            parameters_description="query (str): search text",
            returns_description="JSON with results",
            recovery_hint="Try different terms",
        )
        assert entry.description == "Search papers"
        assert entry.category == "research"
        assert entry.recovery_hint == "Try different terms"

    def test_rejects_an_unknown_key(self):
        """A typo must fail loudly rather than silently leave the field empty."""
        with pytest.raises(ValidationError, match="typo"):
            ToolYamlEntry(typo="Search papers")

    def test_rejects_a_misspelled_field(self):
        """The near-miss is the realistic case — the hint would never reach the tool."""
        with pytest.raises(ValidationError, match="recovery_hints"):
            ToolYamlEntry(recovery_hints="Try different terms")

    def test_rejects_an_unknown_mcp_key(self):
        """``mcp: {exposed: true}`` would leave the tool unexposed, silently."""
        with pytest.raises(ValidationError, match="exposed"):
            ToolYamlEntry(mcp={"exposed": True})


# ---------------------------------------------------------------------------
# ToolCatalog
# ---------------------------------------------------------------------------


class TestToolCatalog:
    def test_get_existing_tool(self):
        catalog = ToolCatalog(
            raw_tools={
                "search": {"description": "Search stuff", "category": "graph"},
            }
        )
        entry = catalog.get("search")
        assert entry.description == "Search stuff"
        assert entry.category == "graph"

    def test_get_missing_raises(self):
        catalog = ToolCatalog(raw_tools={"search": {"description": "d"}})
        with pytest.raises(KeyError, match="not found in tools.yaml"):
            catalog.get("missing_tool")

    def test_list_tools(self):
        catalog = ToolCatalog(
            raw_tools={
                "search": {"description": "a"},
                "read": {"description": "b"},
            }
        )
        assert set(catalog.list_tools()) == {"search", "read"}

    def test_empty_catalog(self):
        catalog = ToolCatalog(raw_tools={})
        assert catalog.list_tools() == []

    def test_get_rejects_an_unrecognized_tool_key(self):
        """A key tools.yaml does not define fails on resolve, not silently."""
        catalog = ToolCatalog(raw_tools={"search": {"description": "d", "categry": "graph"}})
        with pytest.raises(ValidationError, match="categry"):
            catalog.get("search")


# ---------------------------------------------------------------------------
# ToolCatalog copies the caller's map
# ---------------------------------------------------------------------------


class TestToolCatalogCopiesTheCallersMap:
    """The catalog owns its data, so a later edit to the caller's dict cannot reach it."""

    def test_a_new_tool_added_after_construction_is_not_visible(self):
        raw_tools: dict[str, dict[str, Any]] = {"search": {"description": "Search stuff"}}
        catalog = ToolCatalog(raw_tools=raw_tools)

        raw_tools["late_tool"] = {"description": "Added late"}

        assert catalog.list_tools() == ["search"]

    def test_an_edit_inside_a_tool_block_is_not_visible(self):
        raw_tools: dict[str, dict[str, Any]] = {
            "search": {"description": "Search stuff", "category": "graph"}
        }
        catalog = ToolCatalog(raw_tools=raw_tools)

        raw_tools["search"]["category"] = "edited late"

        assert catalog.get("search").category == "graph"

    def test_an_edit_inside_a_nested_mcp_block_is_not_visible(self):
        """``mcp`` nests inside the tool block, so a shallow copy would not detach it."""
        raw_tools: dict[str, dict[str, Any]] = {
            "search": {"description": "Search stuff", "mcp": {"expose": True}}
        }
        catalog = ToolCatalog(raw_tools=raw_tools)

        raw_tools["search"]["mcp"]["expose"] = False

        assert catalog.get_mcp_tools() == ["search"]

    def test_a_late_edit_does_not_reach_registry_enrichment(self):
        raw_tools: dict[str, dict[str, Any]] = {"search": {"description": "Search stuff"}}
        catalog = ToolCatalog(raw_tools=raw_tools)
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="search", function=lambda: None))

        raw_tools["search"]["description"] = "Edited late"
        catalog.enrich_registry(registry)

        assert registry.get_tool("search").description == "Search stuff"

    def test_two_catalogs_built_from_the_same_data_are_independent_snapshots(self):
        raw_tools: dict[str, dict[str, Any]] = {"search": {"description": "Search stuff"}}

        first = ToolCatalog(raw_tools=raw_tools)
        raw_tools["search"]["description"] = "Search things"
        second = ToolCatalog(raw_tools=raw_tools)

        assert first.get("search").description == "Search stuff"
        assert second.get("search").description == "Search things"


# ---------------------------------------------------------------------------
# enrich_registry
# ---------------------------------------------------------------------------


class TestEnrichRegistry:
    @staticmethod
    def _make_registry(*tools: ToolDefinition) -> ToolRegistry:
        reg = ToolRegistry()
        for t in tools:
            reg.register(t)
        return reg

    def test_enrich_rejects_an_unrecognized_tool_key(self):
        """Enrichment reads every entry, so a typo surfaces at startup."""
        reg = self._make_registry(ToolDefinition(name="search", function=lambda: None))
        catalog = ToolCatalog(raw_tools={"search": {"descriptions": "Search the graph"}})

        with pytest.raises(ValidationError, match="descriptions"):
            catalog.enrich_registry(reg)

    def test_yaml_overwrites_empty_decorator_fields(self):
        """YAML fills in metadata that the decorator left empty."""
        reg = self._make_registry(ToolDefinition(name="search", function=lambda: None))
        catalog = ToolCatalog(
            raw_tools={
                "search": {
                    "description": "Search the graph",
                    "category": "graph_nav",
                    "recovery_hint": "Try broader terms",
                },
            }
        )
        catalog.enrich_registry(reg)

        tool = reg.get_tool("search")
        assert tool.description == "Search the graph"
        assert tool.category == "graph_nav"
        assert tool.recovery_hint == "Try broader terms"

    def test_yaml_overwrites_decorator_values(self):
        """YAML wins over decorator-provided metadata."""
        reg = self._make_registry(
            ToolDefinition(
                name="search",
                function=lambda: None,
                description="old desc",
                category="old_cat",
            )
        )
        catalog = ToolCatalog(
            raw_tools={
                "search": {"description": "new desc", "category": "new_cat"},
            }
        )
        catalog.enrich_registry(reg)

        tool = reg.get_tool("search")
        assert tool.description == "new desc"
        assert tool.category == "new_cat"

    def test_empty_yaml_field_keeps_decorator_value(self):
        """Empty YAML field does not overwrite existing decorator value."""
        reg = self._make_registry(
            ToolDefinition(
                name="search",
                function=lambda: None,
                description="decorator desc",
                category="decorator_cat",
            )
        )
        catalog = ToolCatalog(
            raw_tools={
                "search": {"description": "", "category": ""},
            }
        )
        catalog.enrich_registry(reg)

        tool = reg.get_tool("search")
        assert tool.description == "decorator desc"
        assert tool.category == "decorator_cat"

    def test_tool_in_yaml_not_in_registry_logs_warning(self, caplog):
        """YAML tool with no registered function logs warning."""
        reg = self._make_registry()  # empty registry
        catalog = ToolCatalog(
            raw_tools={
                "unknown_tool": {"description": "No function registered"},
            }
        )
        with caplog.at_level("WARNING"):
            catalog.enrich_registry(reg)
        assert "unknown_tool" in caplog.text
        assert "no registered function" in caplog.text

    def test_tool_in_registry_not_in_yaml_keeps_metadata(self):
        """Tool registered via decorator but absent from YAML keeps its values."""
        reg = self._make_registry(
            ToolDefinition(
                name="custom_tool",
                function=lambda: None,
                description="custom desc",
                category="custom",
            )
        )
        catalog = ToolCatalog(raw_tools={})  # empty YAML
        catalog.enrich_registry(reg)

        tool = reg.get_tool("custom_tool")
        assert tool.description == "custom desc"
        assert tool.category == "custom"

    def test_function_reference_preserved(self):
        """enrich_registry never touches the function reference."""

        def fn():
            return "hello"

        reg = self._make_registry(ToolDefinition(name="t", function=fn))
        catalog = ToolCatalog(
            raw_tools={
                "t": {"description": "enriched"},
            }
        )
        catalog.enrich_registry(reg)
        assert reg.get_tool("t").function is fn

    def test_multiple_tools_enriched(self):
        """Multiple tools are all enriched in one pass."""
        reg = self._make_registry(
            ToolDefinition(name="a", function=lambda: None),
            ToolDefinition(name="b", function=lambda: None),
        )
        catalog = ToolCatalog(
            raw_tools={
                "a": {"description": "tool a", "category": "cat_a"},
                "b": {"description": "tool b", "category": "cat_b"},
            }
        )
        catalog.enrich_registry(reg)

        assert reg.get_tool("a").description == "tool a"
        assert reg.get_tool("b").category == "cat_b"


# ---------------------------------------------------------------------------
# load_tool_catalog
# ---------------------------------------------------------------------------


class TestLoadToolCatalog:
    def test_load_from_file(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            tools:
              search:
                description: Search the graph
                category: graph
                recovery_hint: Try different terms
              read:
                description: Read a node
                category: graph
        """)
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(yaml_content)

        catalog = load_tool_catalog(yaml_file)
        assert set(catalog.list_tools()) == {"search", "read"}

        entry = catalog.get("search")
        assert entry.description == "Search the graph"
        assert entry.recovery_hint == "Try different terms"

    def test_missing_file_returns_empty(self, tmp_path: Path):
        catalog = load_tool_catalog(tmp_path / "nonexistent.yaml")
        assert catalog.list_tools() == []

    def test_empty_file_returns_empty(self, tmp_path: Path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text("")
        catalog = load_tool_catalog(yaml_file)
        assert catalog.list_tools() == []

    def test_file_without_tools_key(self, tmp_path: Path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text("other_key: value\n")
        catalog = load_tool_catalog(yaml_file)
        assert catalog.list_tools() == []

    def test_enrich_from_loaded_file(self, tmp_path: Path):
        """End-to-end: load YAML, enrich registry, verify."""
        yaml_content = textwrap.dedent("""\
            tools:
              my_tool:
                description: My tool from YAML
                category: testing
                parameters_description: "x (int): a number"
                returns_description: "JSON result"
                recovery_hint: Check x is positive
        """)
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(yaml_content)

        def fn():
            return "result"

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="my_tool", function=fn))

        catalog = load_tool_catalog(yaml_file)
        catalog.enrich_registry(reg)

        tool = reg.get_tool("my_tool")
        assert tool.description == "My tool from YAML"
        assert tool.category == "testing"
        assert tool.parameters_description == "x (int): a number"
        assert tool.returns_description == "JSON result"
        assert tool.recovery_hint == "Check x is positive"
        assert tool.function is fn


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_import_from_registry_package(self):
        from sinan_agentic_core.registry import ToolCatalog, ToolYamlEntry, load_tool_catalog

        assert ToolCatalog is not None
        assert ToolYamlEntry is not None
        assert load_tool_catalog is not None

    def test_import_from_top_level(self):
        from sinan_agentic_core import ToolCatalog, ToolYamlEntry, load_tool_catalog

        assert ToolCatalog is not None
        assert ToolYamlEntry is not None
        assert load_tool_catalog is not None
