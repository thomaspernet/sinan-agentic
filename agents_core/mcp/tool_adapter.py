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
import uuid
from typing import Any, Optional

from ..registry.tool_registry import ToolDefinition, ToolRegistry
from .context_protocol import MCPContextFactory

logger = logging.getLogger(__name__)


def _has_on_invoke_tool(obj: Any) -> bool:
    """Check if *obj* is a FunctionTool with ``on_invoke_tool``."""
    return hasattr(obj, "on_invoke_tool") and callable(obj.on_invoke_tool)


def _get_params_schema(tool_def: ToolDefinition) -> dict[str, Any]:
    """Extract JSON schema from a registered tool.

    If the tool function is a ``FunctionTool``, use its ``params_json_schema``.
    Otherwise, introspect the raw function (skip the first ``ctx`` parameter).
    """
    fn = tool_def.function

    # Path 1: FunctionTool from OpenAI Agents SDK
    if hasattr(fn, "params_json_schema"):
        return fn.params_json_schema

    # Path 2: raw function — build schema from type hints
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for name, param in sig.parameters.items():
        if name == "ctx":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = param.annotation
        json_type = type_map.get(annotation, "string")
        prop: dict[str, Any] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

        properties[name] = prop

    return {"type": "object", "properties": properties, "required": required}


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

    json_to_python = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    for prop_name, prop_def in properties.items():
        json_type = prop_def.get("type", "string")
        py_type = json_to_python.get(json_type, str)
        has_default = "default" in prop_def
        is_required = prop_name in required_set and not has_default

        if is_required:
            default = inspect.Parameter.empty
        else:
            default = prop_def.get("default", None)
            py_type = Optional[py_type]  # type: ignore[assignment]

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
                return await impl(ctx, **params)

        # Path 2: FunctionTool — invoke via on_invoke_tool with synthetic context
        if _has_on_invoke_tool(fn):
            return await self._invoke_function_tool(fn, tool_name, params)

        # Path 3: raw async function with ctx as first param
        if inspect.iscoroutinefunction(fn):
            async with self._context_factory.tool_context() as ctx:
                return await fn(ctx, **params)

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
