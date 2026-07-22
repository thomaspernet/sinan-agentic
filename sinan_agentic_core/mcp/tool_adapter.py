"""MCP tool adapter — invoke registered tools from MCP context.

Bridges the gap between MCP (no RunContextWrapper) and the existing tool
functions (which expect RunContextWrapper/ToolContext as first argument).

Two invocation paths:

1. **Tool has ``_impl`` attribute** — call the standalone implementation
   directly with dependencies from the context factory. Preferred path
   for tools that have been refactored for protocol-agnostic use.

2. **Fallback** — build a synthetic ``ToolContext`` from the app's
   ``MCPContextFactory`` and call ``FunctionTool.on_invoke_tool()``.
   Works with any existing tool without code changes.
"""

import inspect
import json
import logging
import types
import uuid
from typing import Any, Union, get_args, get_origin

from ..registry.tool_registry import ToolDefinition, ToolRegistry
from .context_protocol import MCPContextFactory

logger = logging.getLogger(__name__)


def _has_on_invoke_tool(obj: Any) -> bool:
    """Check if *obj* is a FunctionTool with ``on_invoke_tool``."""
    return hasattr(obj, "on_invoke_tool") and callable(obj.on_invoke_tool)


_JSON_NULL = "null"
_JSON_ARRAY = "array"
_JSON_OBJECT = "object"

# The single authoritative scalar mapping. Both translation directions read it:
# ``_resolve_annotation`` (schema to annotation) forwards, ``_resolve_schema``
# (annotation to schema) through the inverse below. A scalar added here reaches
# both directions at once, so the two cannot drift.
_JSON_SCALARS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_PYTHON_SCALARS: dict[Any, str] = {python: json for json, python in _JSON_SCALARS.items()}


def _resolve_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """Resolve a Python annotation into a JSON-schema property.

    Returns the property and whether it accepts ``None``.

    Parameterized annotations resolve through ``get_origin``/``get_args``, so
    ``list[str]`` keeps its element type and nested containers survive. A union
    collapses to its branches; a ``None`` member marks the property nullable and
    adds a ``{"type": "null"}`` branch.

    An annotation the mapping does not cover resolves to an unconstrained
    property. That is honest about what the builder knows; asserting ``string``
    — the previous fallback — advertises a type the tool would reject.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}, False
    if annotation is type(None):
        return {"type": _JSON_NULL}, True

    origin = get_origin(annotation)

    if origin in (types.UnionType, Union):
        args = get_args(annotation)
        branches = [arg for arg in args if arg is not type(None)]
        nullable = len(branches) != len(args)
        schemas = [_resolve_schema(branch)[0] for branch in branches]
        if nullable:
            schemas.append({"type": _JSON_NULL})
        if len(schemas) == 1:
            return schemas[0], nullable
        return {"anyOf": schemas}, nullable

    if origin is list or annotation is list:
        args = get_args(annotation)
        prop: dict[str, Any] = {"type": _JSON_ARRAY}
        items = _resolve_schema(args[0])[0] if args else {}
        if items:
            prop["items"] = items
        return prop, False

    if origin is dict or annotation is dict:
        return {"type": _JSON_OBJECT}, False

    scalar = _PYTHON_SCALARS.get(annotation)
    if scalar is not None:
        return {"type": scalar}, False
    return {}, False


def _get_params_schema(tool_def: ToolDefinition) -> dict[str, Any]:
    """Extract JSON schema from a registered tool.

    If the tool function is a ``FunctionTool``, use its ``params_json_schema``.
    Otherwise, introspect the raw function (skip the first ``ctx`` parameter).
    """
    fn = tool_def.function

    # Path 1: FunctionTool from OpenAI Agents SDK
    if hasattr(fn, "params_json_schema"):
        schema: dict[str, Any] = fn.params_json_schema
        return schema

    # Path 2: raw function — build schema from type hints
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "ctx":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        prop, nullable = _resolve_schema(param.annotation)

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        elif not nullable:
            required.append(name)

        properties[name] = prop

    return {"type": "object", "properties": properties, "required": required}


def _resolve_annotation(prop_def: dict[str, Any]) -> tuple[Any, bool]:
    """Resolve a JSON-schema property into a Python annotation.

    Returns the annotation and whether the property accepts ``null``.

    An ``anyOf`` collapses to the union of its non-null branches; a
    ``{"type": "null"}`` branch marks the property nullable. Arrays keep their
    element type (``items`` resolves recursively, so nested containers survive)
    and objects become ``dict[str, Any]``.

    A shape the mapping does not cover resolves to ``Any``. An unconstrained
    parameter is honest about what the builder knows; asserting ``str`` — the
    previous fallback — advertises a type the tool would reject at invocation.
    """
    any_of = prop_def.get("anyOf")
    if any_of:
        branches = [b for b in any_of if b.get("type") != _JSON_NULL]
        nullable = len(branches) != len(any_of)
        if not branches:
            return type(None), True
        annotation, _ = _resolve_annotation(branches[0])
        for branch in branches[1:]:
            annotation = annotation | _resolve_annotation(branch)[0]
        return annotation, nullable

    json_type = str(prop_def.get("type", ""))
    if json_type == _JSON_ARRAY:
        items = prop_def.get("items")
        item_type = _resolve_annotation(items)[0] if items else Any
        return types.GenericAlias(list, (item_type,)), False
    if json_type == _JSON_OBJECT:
        return dict[str, Any], False
    if json_type == _JSON_NULL:
        return type(None), True
    return _JSON_SCALARS.get(json_type, Any), False


def _build_mcp_handler(
    tool_name: str,
    params_schema: dict[str, Any],
    description: str,
    adapter: "MCPToolAdapter",
) -> Any:
    """Build a typed async function suitable for FastMCP registration.

    Creates a wrapper with proper ``__signature__`` and ``__annotations__``
    so FastMCP can introspect it for JSON schema generation.
    """
    properties = params_schema.get("properties", {})
    required_set = set(params_schema.get("required", []))

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    for prop_name, prop_def in properties.items():
        py_type, nullable = _resolve_annotation(prop_def)
        has_default = "default" in prop_def
        is_required = prop_name in required_set and not has_default and not nullable

        if is_required:
            default = inspect.Parameter.empty
        else:
            default = prop_def.get("default", None)
            if nullable or default is None:
                py_type = py_type | None

        parameters.append(
            inspect.Parameter(
                prop_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=py_type,
            )
        )
        annotations[prop_name] = py_type

    annotations["return"] = str

    async def handler(**kwargs: Any) -> str:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return await adapter.invoke(tool_name, **clean)

    handler.__name__ = tool_name
    handler.__qualname__ = tool_name
    handler.__doc__ = description
    handler.__annotations__ = annotations
    handler.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=parameters, return_annotation=str
    )

    return handler


class MCPToolAdapter:
    """Adapts registered tool functions for MCP invocation.

    Usage::

        adapter = MCPToolAdapter(registry, context_factory)
        result = await adapter.invoke("discover", target="overview")
    """

    def __init__(
        self,
        registry: ToolRegistry,
        context_factory: MCPContextFactory,
    ) -> None:
        self._registry = registry
        self._context_factory = context_factory

    async def invoke(self, tool_name: str, **params: Any) -> str:
        """Invoke a registered tool by name with the given parameters.

        Returns the tool's string result.

        Raises:
            KeyError: If the tool is not registered.
            RuntimeError: If the tool invocation fails.
        """
        tool_def = self._registry.get_tool(tool_name)
        if tool_def is None:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        fn = tool_def.function

        # Path 1: tool has a standalone _impl (protocol-agnostic)
        impl = getattr(fn, "_impl", None)
        if impl is not None:
            async with self._context_factory.tool_context() as ctx:
                impl_result = await impl(ctx, **params)
                return impl_result if isinstance(impl_result, str) else str(impl_result)

        # Path 2: FunctionTool — invoke via on_invoke_tool with synthetic context
        if _has_on_invoke_tool(fn):
            return await self._invoke_function_tool(fn, tool_name, params)

        # Path 3: raw async function with ctx as first param
        if inspect.iscoroutinefunction(fn):
            async with self._context_factory.tool_context() as ctx:
                fn_result = await fn(ctx, **params)
                return fn_result if isinstance(fn_result, str) else str(fn_result)

        raise RuntimeError(
            f"Tool '{tool_name}' has no supported invocation method. "
            f"Expected FunctionTool, async function, or _impl attribute."
        )

    async def _invoke_function_tool(
        self,
        fn: Any,
        tool_name: str,
        params: dict[str, Any],
    ) -> str:
        """Invoke a FunctionTool via on_invoke_tool with synthetic ToolContext."""
        from agents.tool_context import ToolContext

        input_json = json.dumps(params)
        async with self._context_factory.tool_context() as app_ctx:
            ctx = ToolContext(
                context=app_ctx,
                tool_name=tool_name,
                tool_call_id=f"mcp-{uuid.uuid4().hex[:12]}",
                tool_arguments=input_json,
            )
            result = await fn.on_invoke_tool(ctx, input_json)
            return result if isinstance(result, str) else str(result)

    def build_mcp_handler(self, tool_name: str) -> Any:
        """Build a typed async handler for FastMCP registration.

        The returned function has proper ``__signature__`` and ``__annotations__``
        so FastMCP can generate the correct JSON schema.
        """
        tool_def = self._registry.get_tool(tool_name)
        if tool_def is None:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        schema = _get_params_schema(tool_def)
        description = tool_def.description or tool_def.name

        return _build_mcp_handler(tool_name, schema, description, self)
