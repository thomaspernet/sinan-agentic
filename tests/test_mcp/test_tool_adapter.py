"""Tests for MCPToolAdapter — tool invocation and handler building."""

import json
import pytest
from unittest.mock import AsyncMock

from agents import function_tool
from agents.tool_context import ToolContext

from sinan_agentic_core.registry.tool_registry import ToolRegistry, ToolDefinition
from sinan_agentic_core.mcp.context_protocol import MCPContextFactory
from sinan_agentic_core.mcp.tool_adapter import MCPToolAdapter, _get_params_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeAppContext:
    """Simulates an app context (like AgentContext with db_connector)."""

    def __init__(self):
        self.db_connector = "fake_db"
        self.filters = None


class FakeContextFactory(MCPContextFactory):
    """Test factory that produces FakeAppContext."""

    async def create_context(self):
        return FakeAppContext()

    async def cleanup(self, context):
        pass


@function_tool
async def discover_tool(ctx: ToolContext, target: str = "overview") -> str:
    """Discover what's available."""
    return json.dumps({"target": target, "db": ctx.context.db_connector})


@function_tool
async def search_tool(ctx: ToolContext, query: str, limit: int = 10) -> str:
    """Search for things."""
    return json.dumps({"query": query, "limit": limit})


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="discover",
        function=discover_tool,
        description="Discover what's available",
    ))
    reg.register(ToolDefinition(
        name="search",
        function=search_tool,
        description="Search for things",
    ))
    return reg


@pytest.fixture
def adapter(registry):
    return MCPToolAdapter(registry, FakeContextFactory())


# ---------------------------------------------------------------------------
# Tests: _get_params_schema
# ---------------------------------------------------------------------------


def test_get_params_schema_function_tool(registry):
    tool_def = registry.get_tool("discover")
    schema = _get_params_schema(tool_def)
    assert "properties" in schema
    assert "target" in schema["properties"]
    # ctx should NOT be in the schema
    assert "ctx" not in schema.get("properties", {})


def test_get_params_schema_search(registry):
    tool_def = registry.get_tool("search")
    schema = _get_params_schema(tool_def)
    props = schema["properties"]
    assert "query" in props
    assert "limit" in props
    assert props["query"]["type"] == "string"
    assert props["limit"]["type"] == "integer"


# ---------------------------------------------------------------------------
# Tests: MCPToolAdapter.invoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_function_tool(adapter):
    result = await adapter.invoke("discover", target="entities")
    data = json.loads(result)
    assert data["target"] == "entities"
    assert data["db"] == "fake_db"


@pytest.mark.asyncio
async def test_invoke_with_defaults(adapter):
    result = await adapter.invoke("discover")
    data = json.loads(result)
    assert data["target"] == "overview"


@pytest.mark.asyncio
async def test_invoke_search(adapter):
    result = await adapter.invoke("search", query="hello", limit=5)
    data = json.loads(result)
    assert data["query"] == "hello"
    assert data["limit"] == 5


@pytest.mark.asyncio
async def test_invoke_unknown_tool(adapter):
    with pytest.raises(KeyError, match="not_real"):
        await adapter.invoke("not_real")


# ---------------------------------------------------------------------------
# Tests: build_mcp_handler
# ---------------------------------------------------------------------------


def test_build_mcp_handler_signature(adapter):
    handler = adapter.build_mcp_handler("discover")
    assert handler.__name__ == "discover"
    assert handler.__doc__ == "Discover what's available"

    import inspect
    sig = inspect.signature(handler)
    assert "target" in sig.parameters
    # ctx should NOT appear
    assert "ctx" not in sig.parameters


def test_build_mcp_handler_search_signature(adapter):
    handler = adapter.build_mcp_handler("search")

    import inspect
    sig = inspect.signature(handler)
    params = sig.parameters
    assert "query" in params
    assert "limit" in params
    assert params["query"].default is inspect.Parameter.empty
    assert params["limit"].default is not inspect.Parameter.empty


@pytest.mark.asyncio
async def test_build_mcp_handler_callable(adapter):
    handler = adapter.build_mcp_handler("search")
    result = await handler(query="test", limit=3)
    data = json.loads(result)
    assert data["query"] == "test"
    assert data["limit"] == 3


def test_build_mcp_handler_unknown_tool(adapter):
    with pytest.raises(KeyError, match="not_real"):
        adapter.build_mcp_handler("not_real")
