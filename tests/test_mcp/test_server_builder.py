"""Tests for MCPServerBuilder — building FastMCP servers from registry."""

import json
import pytest

from agents import function_tool
from agents.tool_context import ToolContext

from sinan_agentic_core.registry.tool_registry import ToolRegistry, ToolDefinition
from sinan_agentic_core.registry.tool_catalog import ToolCatalog
from sinan_agentic_core.mcp.context_protocol import MCPContextFactory
from sinan_agentic_core.mcp.yaml_schema import MCPServerConfig
from sinan_agentic_core.mcp.server_builder import build_mcp_server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeContext:
    db = "fake_db"


class FakeFactory(MCPContextFactory):
    async def create_context(self):
        return FakeContext()


@function_tool
async def discover(ctx: ToolContext, target: str = "overview") -> str:
    """Discover the graph."""
    return json.dumps({"target": target})


@function_tool
async def search(ctx: ToolContext, query: str, limit: int = 10) -> str:
    """Search the graph."""
    return json.dumps({"query": query, "limit": limit})


@function_tool
async def create_page(ctx: ToolContext, title: str, content: str = "") -> str:
    """Create a page."""
    return json.dumps({"title": title})


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="discover", function=discover, description="Discover"))
    reg.register(ToolDefinition(name="search", function=search, description="Search"))
    reg.register(ToolDefinition(name="create_page", function=create_page, description="Create page"))
    return reg


@pytest.fixture
def catalog():
    return ToolCatalog(raw_tools={
        "discover": {
            "description": "Discover what's available in the graph",
            "category": "graph_navigation",
            "mcp": {"expose": True, "annotations": {"readOnlyHint": True}},
        },
        "search": {
            "description": "Search the graph",
            "category": "graph_navigation",
            "mcp": {"expose": True, "annotations": {"readOnlyHint": True}},
        },
        "create_page": {
            "description": "Create a new page",
            "category": "action",
            "mcp": {
                "expose": True,
                "annotations": {"readOnlyHint": False, "destructiveHint": False},
            },
        },
    })


@pytest.fixture
def mcp_config():
    return MCPServerConfig(
        name="test_server",
        description="Test MCP Server",
        tools=["discover", "search"],
        write_tools=["create_page"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_server_read_only(registry, catalog, mcp_config):
    """Read-only server should have only read tools."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
        include_write_tools=False,
    )

    tools = server._tool_manager._tools
    assert "discover" in tools
    assert "search" in tools
    assert "create_page" not in tools


def test_build_server_with_writes(registry, catalog, mcp_config):
    """With include_write_tools, write tools should be registered."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
        include_write_tools=True,
    )

    tools = server._tool_manager._tools
    assert "discover" in tools
    assert "search" in tools
    assert "create_page" in tools


def test_build_server_descriptions_from_yaml(registry, catalog, mcp_config):
    """Descriptions should come from YAML (via catalog enrichment), not code."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    discover_tool = server._tool_manager._tools["discover"]
    assert discover_tool.description == "Discover what's available in the graph"


def test_build_server_annotations(registry, catalog, mcp_config):
    """MCP annotations should be applied from YAML config."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    discover_tool = server._tool_manager._tools["discover"]
    assert discover_tool.annotations is not None
    assert discover_tool.annotations.readOnlyHint is True


def test_build_server_missing_tool_skipped(registry, catalog):
    """Tools listed in config but not registered should be skipped."""
    catalog.enrich_registry(registry)

    config = MCPServerConfig(
        name="test",
        tools=["discover", "nonexistent_tool"],
    )

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=config,
        context_factory=FakeFactory(),
    )

    tools = server._tool_manager._tools
    assert "discover" in tools
    assert "nonexistent_tool" not in tools


def test_build_server_tool_schema(registry, catalog, mcp_config):
    """Tool parameters schema should be correct."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    search_tool = server._tool_manager._tools["search"]
    schema = search_tool.parameters
    assert "query" in schema["properties"]
    assert "limit" in schema["properties"]


@pytest.mark.asyncio
async def test_build_server_tool_invocable(registry, catalog, mcp_config):
    """Built MCP tools should be callable and return correct results."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    # Get the tool's handler and call it
    discover_tool = server._tool_manager._tools["discover"]
    result = await discover_tool.fn(target="entities")
    data = json.loads(result)
    assert data["target"] == "entities"
