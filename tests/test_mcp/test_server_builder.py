"""Tests for MCPServerBuilder — building FastMCP servers from registry."""

import json
from typing import Any

import pytest
from agents import function_tool
from agents.tool_context import ToolContext

from sinan_agentic_core.mcp.context_protocol import MCPContextFactory
from sinan_agentic_core.mcp.server_builder import build_mcp_server
from sinan_agentic_core.mcp.yaml_schema import MCPServerConfig
from sinan_agentic_core.registry.tool_catalog import ToolCatalog
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry

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


@function_tool
async def containers(
    ctx: ToolContext,
    labels: list[str],
    matrix: list[list[str]],
    tags: list[str] | None = None,
    grid: list[list[int]] | None = None,
) -> str:
    """Take container parameters."""
    return json.dumps({"labels": labels, "matrix": matrix, "tags": tags, "grid": grid})


# The SDK's strict schema forbids the ``additionalProperties`` key an open
# mapping needs, so the dict shapes round-trip through a non-strict tool.
@function_tool(strict_mode=False)
async def lenient_containers(
    ctx: ToolContext,
    filters: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> str:
    """Take dict parameters."""
    return json.dumps({"filters": filters, "extras": extras})


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="discover", function=discover, description="Discover"))
    reg.register(ToolDefinition(name="search", function=search, description="Search"))
    reg.register(
        ToolDefinition(name="create_page", function=create_page, description="Create page")
    )
    reg.register(ToolDefinition(name="containers", function=containers, description="Containers"))
    reg.register(
        ToolDefinition(
            name="lenient_containers",
            function=lenient_containers,
            description="Lenient containers",
        )
    )
    return reg


@pytest.fixture
def catalog():
    return ToolCatalog(
        raw_tools={
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
            "containers": {
                "description": "Take container parameters",
                "category": "graph_navigation",
                "mcp": {"expose": True},
            },
            "lenient_containers": {
                "description": "Take dict parameters",
                "category": "graph_navigation",
                "mcp": {"expose": True},
            },
        }
    )


@pytest.fixture
def mcp_config():
    return MCPServerConfig(
        name="test_server",
        description="Test MCP Server",
        tools=["discover", "search", "containers", "lenient_containers"],
        write_tools=["create_page"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def listed_tools(server):
    """Map the server's advertised MCP tools by name.

    Uses the server's public tool-listing API — the same view an MCP client
    gets — rather than reading the SDK's internal tool manager.
    """
    return {tool.name: tool for tool in await server.list_tools()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_build_server_read_only(registry, catalog, mcp_config):
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

    tools = await listed_tools(server)
    assert "discover" in tools
    assert "search" in tools
    assert "create_page" not in tools


async def test_build_server_with_writes(registry, catalog, mcp_config):
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

    tools = await listed_tools(server)
    assert "discover" in tools
    assert "search" in tools
    assert "create_page" in tools


async def test_build_server_descriptions_from_yaml(registry, catalog, mcp_config):
    """Descriptions should come from YAML (via catalog enrichment), not code."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    tools = await listed_tools(server)
    assert tools["discover"].description == "Discover what's available in the graph"


async def test_build_server_annotations(registry, catalog, mcp_config):
    """MCP annotations should be applied from YAML config."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    annotations = (await listed_tools(server))["discover"].annotations
    assert annotations is not None
    assert annotations.readOnlyHint is True


async def test_build_server_missing_tool_skipped(registry, catalog):
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

    tools = await listed_tools(server)
    assert "discover" in tools
    assert "nonexistent_tool" not in tools


async def test_build_server_tool_schema(registry, catalog, mcp_config):
    """Tool parameters schema should be correct."""
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    schema = (await listed_tools(server))["search"].inputSchema
    assert "query" in schema["properties"]
    assert "limit" in schema["properties"]


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

    # Dispatch through the server the way an MCP client would
    content, _structured = await server.call_tool("discover", {"target": "entities"})
    data = json.loads(content[0].text)
    assert data["target"] == "entities"


# ---------------------------------------------------------------------------
# Tests: container parameters survive the round-trip to an MCP client
# ---------------------------------------------------------------------------


def _without_presentation_keys(schema):
    """Drop the keys that carry no type information, recursively.

    ``title`` is generated from the parameter name on both sides, and ``default``
    is present only where a value was declared — neither says anything about the
    type a client must send.
    """
    if isinstance(schema, dict):
        return {
            key: _without_presentation_keys(value)
            for key, value in schema.items()
            if key not in ("title", "default")
        }
    if isinstance(schema, list):
        return [_without_presentation_keys(item) for item in schema]
    return schema


def _is_nullable(prop):
    return any(branch.get("type") == "null" for branch in prop.get("anyOf", []))


@pytest.mark.parametrize(
    ("tool_name", "tool_fn"),
    [("containers", containers), ("lenient_containers", lenient_containers)],
)
async def test_the_advertised_schema_matches_the_one_the_sdk_produced(
    registry, catalog, mcp_config, tool_name, tool_fn
):
    """What FastMCP advertises carries the SDK's types, with real requiredness.

    A correct handler signature that FastMCP then renders as a bare ``string``
    would pass every signature-level assertion, so the comparison runs against
    the schema an MCP client actually receives.
    """
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    sdk_schema = tool_fn.params_json_schema
    advertised = (await listed_tools(server))[tool_name].inputSchema

    assert advertised["properties"].keys() == sdk_schema["properties"].keys()
    for name, sdk_prop in sdk_schema["properties"].items():
        assert _without_presentation_keys(advertised["properties"][name]) == (
            _without_presentation_keys(sdk_prop)
        ), name

    # The SDK's strict schema lists every parameter as required regardless of
    # nullability; what reaches the client must not.
    assert advertised["required"] == [
        name for name in sdk_schema["required"] if not _is_nullable(sdk_schema["properties"][name])
    ]


async def test_a_tool_with_an_optional_list_is_invocable_through_the_server(
    registry, catalog, mcp_config
):
    catalog.enrich_registry(registry)

    server = build_mcp_server(
        server_name="Test",
        tool_registry=registry,
        tool_catalog=catalog,
        mcp_config=mcp_config,
        context_factory=FakeFactory(),
    )

    omitted, _ = await server.call_tool("containers", {"labels": ["a"], "matrix": [["m"]]})
    assert json.loads(omitted[0].text)["tags"] is None

    supplied, _ = await server.call_tool(
        "containers", {"labels": ["a"], "matrix": [["m"]], "tags": ["x", "y"]}
    )
    assert json.loads(supplied[0].text)["tags"] == ["x", "y"]
