"""MCP tool adapter — invoke registered tools from MCP context.

Bridges the gap between MCP (no RunContextWrapper) and the existing tool
functions (which expect RunContextWrapper/ToolContext as first argument).

Invocation paths, in the order :meth:`MCPToolAdapter.invoke` tries them:

1. **Direct call** — invoke the tool's own callable, reached either through the
   SDK's ``FunctionTool.__wrapped__`` (every ``@function_tool``-decorated tool
   exposes it, and it survives the ``dataclasses.replace`` the registries hand
   records out through) or through an ``_impl`` attribute the app attached
   itself. Skips the argument round-trip path 2 pays for. A callable is only
   reached this way when every parameter can be passed by name, since the
   arguments go in as ``**params``; the SDK owns the mapping for every other
   parameter kind, so those tools stay on path 2.

2. **``FunctionTool.on_invoke_tool``** — build a synthetic ``ToolContext`` from
   the app's ``MCPContextFactory`` and let the SDK parse the arguments back out
   of JSON. Works with any FunctionTool, including one built by hand or wrapping
   an agent, neither of which exposes a callable.

3. **Raw async function** — call it with the app context as first argument.

A direct call reaches the implementation *below* the SDK's function-tool
pipeline, so JSON-schema validation, tool-input guardrails, failure handling and
tracing do not run. Path 1 is therefore gated on the tool carrying no tool-input
guardrails: a guarded tool keeps the invocation path it has always taken here
rather than having the bypass widened to it. The gate reads the tool as the
registry holds it — an agent-level guardrail is attached to a per-agent copy
(:func:`~sinan_agentic_core.registry.guardrail_registry.attach_tool_input_guardrails`)
that MCP never sees, and no path here runs tool-input guardrails at all, since
the SDK runs them from the runner rather than from ``on_invoke_tool``.
"""

import inspect
import json
import logging
import types
import uuid
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin, get_type_hints

from agents import FunctionTool, RunContextWrapper
from agents.tool_context import ToolContext

from ..registry.guardrail_registry import has_tool_input_guardrails
from ..registry.tool_registry import ToolDefinition, ToolRegistry
from .context_protocol import MCPContextFactory

logger = logging.getLogger(__name__)


def _has_on_invoke_tool(obj: Any) -> bool:
    """Check if *obj* is a FunctionTool with ``on_invoke_tool``."""
    return hasattr(obj, "on_invoke_tool") and callable(obj.on_invoke_tool)


def _as_str(result: Any) -> str:
    """Normalize a tool result to the string MCP returns."""
    return result if isinstance(result, str) else str(result)


def _takes_context(fn: Callable[..., Any]) -> bool:
    """Whether *fn* expects the run context as its first argument.

    Applies the rule ``agents.function_schema`` uses when it turns a callable
    into a tool: a first parameter annotated ``RunContextWrapper`` or
    ``ToolContext`` receives the context and every remaining parameter is a tool
    argument. Reading the signature keeps the probe to a lookup — asking the SDK
    would rebuild a Pydantic model for the tool on every invocation.

    A parameter whose annotation cannot be resolved (an unimportable forward
    reference) is read as a tool argument, the same shape an unannotated one has.
    """
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        return False

    first = params[0]
    try:
        annotation = get_type_hints(fn).get(first.name, first.annotation)
    except NameError:
        annotation = first.annotation

    if annotation is inspect.Parameter.empty:
        return False
    return (get_origin(annotation) or annotation) in (RunContextWrapper, ToolContext)


# The parameter kinds ``**params`` reaches. A positional-only parameter,
# ``*args`` or ``**kwargs`` is mapped by kind in
# ``agents.function_schema.FunctionSchema.to_call_args``; a tool declaring one
# keeps that invoker rather than having the mapping re-implemented here.
_KEYWORD_PASSABLE_KINDS = frozenset(
    {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
)


def _wrapped_callable(fn: Any) -> Callable[..., Any] | None:
    """The callable behind *fn* when it can be called with the MCP arguments directly.

    ``__wrapped__`` is read from a :class:`~agents.FunctionTool` only. The SDK
    defines it there as a read-only descriptor that raises ``AttributeError``
    when the tool wraps no callable — a hand-built tool, an agent-as-tool — so
    the probe falls through rather than mistaking a missing callable for a
    usable one. Every other object carrying the attribute means something else:
    ``functools.wraps`` sets it on any wrapper it builds, and a raw async tool
    behind an ordinary decorator belongs on path 3 with its app context.

    Returns ``None`` when there is no callable to reach or when its signature
    needs the SDK's parameter-kind mapping.
    """
    if not isinstance(fn, FunctionTool):
        return None

    wrapped: Callable[..., Any] | None = getattr(fn, "__wrapped__", None)
    if wrapped is None:
        return None

    kinds = (param.kind for param in inspect.signature(wrapped).parameters.values())
    if not all(kind in _KEYWORD_PASSABLE_KINDS for kind in kinds):
        return None
    return wrapped


_JSON_NULL = "null"
_JSON_ARRAY = "array"
_JSON_OBJECT = "object"

# The single authoritative scalar mapping. Both translation directions read it:
# ``_resolve_annotation`` (schema to annotation) forwards, ``_resolve_schema``
# (annotation to schema) through the inverse below. A scalar added here reaches
# both directions at once, so the two cannot drift.
_JSON_SCALARS: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_PYTHON_SCALARS: dict[type, str] = {
    python_type: json_type for json_type, python_type in _JSON_SCALARS.items()
}


def _resolve_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """Resolve a Python annotation into a JSON-schema property.

    Returns the property and whether it accepts ``None``.

    Parameterized annotations resolve through ``get_origin``/``get_args``, so
    ``list[str]`` keeps its element type and nested containers survive. A union
    always carries at least two branches — ``Union[X]`` collapses to ``X`` — so it
    always resolves to an ``anyOf``; a ``None`` member marks the property nullable
    and contributes a ``{"type": "null"}`` branch.

    An annotation the mapping does not cover resolves to an unconstrained
    property. That is honest about what the builder knows; asserting ``string``
    — the previous fallback — advertises a type the tool would reject.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}, False
    # Both spellings of the null annotation reach here. ``inspect.signature``
    # reports ``x: None`` as the literal ``None`` and ``get_args(list[None])``
    # yields ``None`` too, so ``NoneType`` alone would miss every declaration
    # short of an explicit ``types.NoneType``.
    if annotation is None or annotation is type(None):
        return {"type": _JSON_NULL}, True

    origin = get_origin(annotation)

    if origin in (types.UnionType, Union):
        args = get_args(annotation)
        branches = [arg for arg in args if arg is not type(None)]
        nullable = len(branches) != len(args)
        schemas = [_resolve_schema(branch)[0] for branch in branches]
        if nullable:
            schemas.append({"type": _JSON_NULL})
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

        # Path 1: call the tool's own implementation. Both routes to it skip the
        # SDK's function-tool pipeline, so a guarded tool is held back to path 2.
        if not has_tool_input_guardrails(fn):
            impl = getattr(fn, "_impl", None)
            if impl is not None:
                return await self._invoke_impl(impl, params)

            wrapped = _wrapped_callable(fn)
            if wrapped is not None:
                return await self._invoke_wrapped(wrapped, tool_name, params)

        # Path 2: FunctionTool — invoke via on_invoke_tool with synthetic context
        if _has_on_invoke_tool(fn):
            return await self._invoke_function_tool(fn, tool_name, params)

        # Path 3: raw async function with ctx as first param
        if inspect.iscoroutinefunction(fn):
            return await self._invoke_impl(fn, params)

        raise RuntimeError(
            f"Tool '{tool_name}' has no supported invocation method. "
            f"Expected a FunctionTool, an async function taking a context, "
            f"or an _impl attribute."
        )

    async def _invoke_impl(
        self,
        impl: Callable[..., Any],
        params: dict[str, Any],
    ) -> str:
        """Call a standalone implementation with the app context as first argument.

        The app owns both sides of this contract — it writes the implementation
        and attaches it — so the context goes in raw, without the SDK wrapper a
        decorated tool's callable expects.
        """
        async with self._context_factory.tool_context() as app_ctx:
            return _as_str(await impl(app_ctx, **params))

    async def _invoke_wrapped(
        self,
        wrapped: Callable[..., Any],
        tool_name: str,
        params: dict[str, Any],
    ) -> str:
        """Call the callable behind a ``@function_tool``-decorated tool directly.

        The arguments go in as they arrived, so the JSON the SDK would parse back
        into them is never built for the call itself. Spreading them as keywords
        is the whole of the calling convention here, which is why
        :func:`_wrapped_callable` only offers up a callable whose parameters all
        accept one. A callable that declares a context parameter still gets the
        same synthetic ``ToolContext`` path 2 builds; one that declares none is
        called with the tool arguments alone. Decorating a sync function is
        supported by the SDK, so the result is awaited only when there is
        something to await.
        """
        async with self._context_factory.tool_context() as app_ctx:
            args = (
                (self._tool_context(app_ctx, tool_name, params),) if _takes_context(wrapped) else ()
            )
            result = wrapped(*args, **params)
            if inspect.isawaitable(result):
                result = await result
            return _as_str(result)

    async def _invoke_function_tool(
        self,
        fn: Any,
        tool_name: str,
        params: dict[str, Any],
    ) -> str:
        """Invoke a FunctionTool via on_invoke_tool with synthetic ToolContext."""
        input_json = json.dumps(params)
        async with self._context_factory.tool_context() as app_ctx:
            ctx = self._tool_context(app_ctx, tool_name, params, input_json)
            return _as_str(await fn.on_invoke_tool(ctx, input_json))

    @staticmethod
    def _tool_context(
        app_ctx: Any,
        tool_name: str,
        params: dict[str, Any],
        input_json: str | None = None,
    ) -> ToolContext[Any]:
        """Build the synthetic ToolContext an MCP call runs under.

        ``tool_arguments`` is mandatory on the SDK dataclass and is the raw
        argument string a tool reads when it wants the unparsed call, so it is
        serialized here when the caller has not already done so.
        """
        return ToolContext(
            context=app_ctx,
            tool_name=tool_name,
            tool_call_id=f"mcp-{uuid.uuid4().hex[:12]}",
            tool_arguments=input_json if input_json is not None else json.dumps(params),
        )

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
