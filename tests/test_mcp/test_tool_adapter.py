"""Tests for MCPToolAdapter — tool invocation and handler building."""

import inspect
import json
from typing import Any

import pytest
from agents import function_tool
from agents.tool_context import ToolContext

from sinan_agentic_core.mcp.context_protocol import MCPContextFactory
from sinan_agentic_core.mcp.tool_adapter import (
    MCPToolAdapter,
    _get_params_schema,
    _resolve_annotation,
    _resolve_schema,
)
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry

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


@function_tool
async def container_tool(
    ctx: ToolContext,
    labels: list[str],
    counts: list[int],
    tags: list[str] | None = None,
    matrix: list[list[str]] | None = None,
) -> str:
    """Take container parameters."""
    return json.dumps({"labels": labels, "counts": counts, "tags": tags, "matrix": matrix})


async def raw_container_tool(
    ctx: Any,
    query: str,
    labels: list[str],
    counts: list[int],
    filters: dict[str, Any],
    nothing: None,
    tags: list[str] | None = None,
    matrix: list[list[str]] | None = None,
    extras: dict[str, Any] | None = None,
    unannotated=None,
) -> str:
    """Take container parameters without a FunctionTool wrapper."""
    return json.dumps({"query": query, "labels": labels, "tags": tags})


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="raw_containers",
            function=raw_container_tool,
            description="Take container parameters without a FunctionTool wrapper",
        )
    )
    reg.register(
        ToolDefinition(
            name="containers",
            function=container_tool,
            description="Take container parameters",
        )
    )
    reg.register(
        ToolDefinition(
            name="discover",
            function=discover_tool,
            description="Discover what's available",
        )
    )
    reg.register(
        ToolDefinition(
            name="search",
            function=search_tool,
            description="Search for things",
        )
    )
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


async def test_invoke_function_tool(adapter):
    result = await adapter.invoke("discover", target="entities")
    data = json.loads(result)
    assert data["target"] == "entities"
    assert data["db"] == "fake_db"


async def test_invoke_with_defaults(adapter):
    result = await adapter.invoke("discover")
    data = json.loads(result)
    assert data["target"] == "overview"


async def test_invoke_search(adapter):
    result = await adapter.invoke("search", query="hello", limit=5)
    data = json.loads(result)
    assert data["query"] == "hello"
    assert data["limit"] == 5


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

    sig = inspect.signature(handler)
    assert "target" in sig.parameters
    # ctx should NOT appear
    assert "ctx" not in sig.parameters


def test_build_mcp_handler_search_signature(adapter):
    handler = adapter.build_mcp_handler("search")

    sig = inspect.signature(handler)
    params = sig.parameters
    assert "query" in params
    assert "limit" in params
    assert params["query"].default is inspect.Parameter.empty
    assert params["limit"].default is not inspect.Parameter.empty


async def test_build_mcp_handler_callable(adapter):
    handler = adapter.build_mcp_handler("search")
    result = await handler(query="test", limit=3)
    data = json.loads(result)
    assert data["query"] == "test"
    assert data["limit"] == 3


def test_build_mcp_handler_unknown_tool(adapter):
    with pytest.raises(KeyError, match="not_real"):
        adapter.build_mcp_handler("not_real")


# ---------------------------------------------------------------------------
# Tests: _resolve_annotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prop_def", "expected_annotation", "expected_nullable"),
    [
        ({"type": "string"}, str, False),
        ({"type": "integer"}, int, False),
        ({"type": "array", "items": {"type": "string"}}, list[str], False),
        ({"type": "array", "items": {"type": "integer"}}, list[int], False),
        ({"type": "object"}, dict[str, Any], False),
        ({"type": "array"}, list[Any], False),
        (
            {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            },
            list[list[str]],
            False,
        ),
        (
            {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ]
            },
            list[str],
            True,
        ),
        (
            {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                    },
                    {"type": "null"},
                ]
            },
            list[list[int]],
            True,
        ),
        ({"anyOf": [{"type": "object"}, {"type": "null"}]}, dict[str, Any], True),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, str | int, False),
        # No type key and no anyOf: the builder has no information to assert one.
        ({}, Any, False),
        ({"type": "unheard-of"}, Any, False),
    ],
)
def test_resolve_annotation(prop_def, expected_annotation, expected_nullable):
    annotation, nullable = _resolve_annotation(prop_def)
    assert annotation == expected_annotation
    assert nullable is expected_nullable


# ---------------------------------------------------------------------------
# Tests: _resolve_schema (the annotation-to-schema direction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected_schema", "expected_nullable"),
    [
        (str, {"type": "string"}, False),
        (int, {"type": "integer"}, False),
        (float, {"type": "number"}, False),
        (bool, {"type": "boolean"}, False),
        (list[str], {"type": "array", "items": {"type": "string"}}, False),
        (list[int], {"type": "array", "items": {"type": "integer"}}, False),
        (list, {"type": "array"}, False),
        (list[Any], {"type": "array"}, False),
        (dict[str, Any], {"type": "object"}, False),
        (dict, {"type": "object"}, False),
        (
            list[list[str]],
            {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            False,
        ),
        (
            list[str] | None,
            {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            True,
        ),
        (
            list[list[int]] | None,
            {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                    },
                    {"type": "null"},
                ]
            },
            True,
        ),
        (
            dict[str, Any] | None,
            {"anyOf": [{"type": "object"}, {"type": "null"}]},
            True,
        ),
        (str | None, {"anyOf": [{"type": "string"}, {"type": "null"}]}, True),
        # Both spellings of the null annotation, and one nested in a container.
        (None, {"type": "null"}, True),
        (type(None), {"type": "null"}, True),
        (list[None], {"type": "array", "items": {"type": "null"}}, False),
        (str | int, {"anyOf": [{"type": "string"}, {"type": "integer"}]}, False),
        # No information to assert a type from.
        (Any, {}, False),
        (inspect.Parameter.empty, {}, False),
        (object, {}, False),
    ],
)
def test_resolve_schema(annotation, expected_schema, expected_nullable):
    schema, nullable = _resolve_schema(annotation)
    assert schema == expected_schema
    assert nullable is expected_nullable


@pytest.mark.parametrize(
    ("prop_def", "annotation"),
    [
        ({"type": "string"}, str),
        ({"type": "array", "items": {"type": "string"}}, list[str]),
        (
            {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
            list[list[int]],
        ),
        ({"type": "object"}, dict[str, Any]),
        ({"type": "null"}, type(None)),
        (
            {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            list[str] | None,
        ),
    ],
)
def test_the_two_directions_round_trip(prop_def, annotation):
    resolved, nullable = _resolve_annotation(prop_def)
    if nullable:
        resolved = resolved | None
    assert resolved == annotation
    assert _resolve_schema(annotation)[0] == prop_def


# ---------------------------------------------------------------------------
# Tests: _get_params_schema on the raw-function path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param_name", "expected_prop"),
    [
        ("query", {"type": "string"}),
        ("labels", {"type": "array", "items": {"type": "string"}}),
        ("counts", {"type": "array", "items": {"type": "integer"}}),
        ("filters", {"type": "object"}),
        ("nothing", {"type": "null"}),
        (
            "tags",
            {
                "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
                "default": None,
            },
        ),
        (
            "matrix",
            {
                "anyOf": [
                    {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    {"type": "null"},
                ],
                "default": None,
            },
        ),
        (
            "extras",
            {"anyOf": [{"type": "object"}, {"type": "null"}], "default": None},
        ),
        # No annotation: no type is asserted, rather than defaulting to string.
        ("unannotated", {"default": None}),
    ],
)
def test_raw_function_containers_keep_their_type(registry, param_name, expected_prop):
    schema = _get_params_schema(registry.get_tool("raw_containers"))

    assert schema["properties"][param_name] == expected_prop


def test_raw_function_optional_params_are_not_required(registry):
    """A nullable param stays out of ``required`` even with no default."""
    schema = _get_params_schema(registry.get_tool("raw_containers"))

    assert schema["required"] == ["query", "labels", "counts", "filters"]
    assert "ctx" not in schema["properties"]


def test_a_raw_function_handler_keeps_its_container_annotations(adapter):
    handler = adapter.build_mcp_handler("raw_containers")
    params = inspect.signature(handler).parameters

    assert params["labels"].annotation == list[str]
    assert params["labels"].default is inspect.Parameter.empty
    assert params["tags"].annotation == list[str] | None
    assert params["tags"].default is None
    assert params["extras"].annotation == dict[str, Any] | None


# ---------------------------------------------------------------------------
# Tests: container parameters end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param_name", "expected_annotation", "expected_optional"),
    [
        ("labels", list[str], False),
        ("counts", list[int], False),
        ("tags", list[str] | None, True),
        ("matrix", list[list[str]] | None, True),
    ],
)
def test_container_params_keep_their_type(
    adapter, param_name, expected_annotation, expected_optional
):
    handler = adapter.build_mcp_handler("containers")
    param = inspect.signature(handler).parameters[param_name]

    assert param.annotation == expected_annotation
    if expected_optional:
        assert param.default is None
    else:
        assert param.default is inspect.Parameter.empty


def test_object_property_becomes_a_dict_param(adapter):
    from sinan_agentic_core.mcp.tool_adapter import _build_mcp_handler

    schema = {
        "type": "object",
        "properties": {"filters": {"type": "object"}},
        "required": ["filters"],
    }
    handler = _build_mcp_handler("filtered", schema, "Filtered", adapter)
    param = inspect.signature(handler).parameters["filters"]

    assert param.annotation == dict[str, Any]
    assert param.default is inspect.Parameter.empty


async def test_optional_container_can_be_omitted(adapter):
    handler = adapter.build_mcp_handler("containers")
    data = json.loads(await handler(labels=["a"], counts=[1]))

    assert data["labels"] == ["a"]
    assert data["counts"] == [1]
    assert data["tags"] is None


async def test_optional_container_accepts_a_list(adapter):
    handler = adapter.build_mcp_handler("containers")
    data = json.loads(await handler(labels=["a"], counts=[1], tags=["x", "y"], matrix=[["n"]]))

    assert data["tags"] == ["x", "y"]
    assert data["matrix"] == [["n"]]
