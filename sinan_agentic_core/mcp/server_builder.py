"""MCP server builder — generate a FastMCP server from the tool registry.

Reads tool functions from ``ToolRegistry``, metadata from ``ToolCatalog``
(tools.yaml), and server config from ``AgentCatalog`` (agents.yaml
``mcp_servers`` section). Produces a ready-to-run ``FastMCP`` server.

Usage::

    from sinan_agentic_core.mcp import build_mcp_server

    server = build_mcp_server(
        server_name="My App",
        tool_registry=get_tool_registry(),
        tool_catalog=tool_catalog,
        mcp_config=mcp_config,
        context_factory=MyContextFactory(),
    )
    server.run(transport="stdio")
"""

import logging
from typing import Any

from ..registry.tool_catalog import ToolCatalog
from ..registry.tool_registry import ToolRegistry
from .context_protocol import MCPContextFactory
from .tool_adapter import MCPToolAdapter
from .yaml_schema import MCPServerConfig, MCPToolConfig

logger = logging.getLogger(__name__)


def _import_fastmcp() -> Any:
    """Import FastMCP with a clear error if not installed."""
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError:
        raise ImportError(
            "FastMCP is required for MCP server support. "
            "Install it with: pip install 'sinan_agentic_core[mcp]' "
            "or: pip install mcp"
        )


def _import_tool_annotations() -> Any:
    """Import ToolAnnotations from mcp.types."""
    from mcp.types import ToolAnnotations

    return ToolAnnotations


class MCPServerBuilder:
    """Build a FastMCP server from the existing tool registry.

    Reads tool functions from ``ToolRegistry``, descriptions and annotations
    from ``ToolCatalog`` (tools.yaml), and which tools to expose from
    ``MCPServerConfig`` (agents.yaml ``mcp_servers`` section).

    No tool descriptions are duplicated — everything comes from YAML.
    """

    def __init__(
        self,
        server_name: str,
        tool_registry: ToolRegistry,
        tool_catalog: ToolCatalog,
        mcp_config: MCPServerConfig,
        context_factory: MCPContextFactory,
        **fastmcp_kwargs: Any,
    ) -> None:
        self._server_name = server_name
        self._registry = tool_registry
        self._catalog = tool_catalog
        self._config = mcp_config
        self._context_factory = context_factory
        self._adapter = MCPToolAdapter(tool_registry, context_factory)
        self._fastmcp_kwargs = fastmcp_kwargs

    def build(self, include_write_tools: bool = False) -> Any:
        """Generate a fully configured FastMCP server.

        Args:
            include_write_tools: If True, also register tools from
                ``mcp_config.write_tools``. Defaults to False (read-only).

        Returns:
            A ``FastMCP`` instance ready to run.
        """
        FastMCP = _import_fastmcp()
        ToolAnnotations = _import_tool_annotations()

        mcp = FastMCP(self._server_name, **self._fastmcp_kwargs)

        # Collect tool names
        tool_names = list(self._config.tools)
        if include_write_tools:
            tool_names.extend(self._config.write_tools)

        # Register each tool
        registered = 0
        for tool_name in tool_names:
            tool_def = self._registry.get_tool(tool_name)
            if tool_def is None:
                logger.warning(
                    "MCP: tool '%s' listed in config but not registered — skipping",
                    tool_name,
                )
                continue

            # Get description from catalog (YAML) — single source of truth
            description = tool_def.description  # already enriched by catalog
            if not description:
                description = tool_name

            # Get MCP annotations from catalog (stored as dict in YAML)
            annotations = None
            mcp_tool_config = self._get_tool_mcp_config(tool_name)
            if mcp_tool_config and mcp_tool_config.annotations:
                ann = mcp_tool_config.annotations
                annotations = ToolAnnotations(
                    readOnlyHint=ann.get("readOnlyHint"),
                    openWorldHint=ann.get("openWorldHint"),
                    destructiveHint=ann.get("destructiveHint"),
                    idempotentHint=ann.get("idempotentHint"),
                )

            # Build typed handler function
            handler = self._adapter.build_mcp_handler(tool_name)

            # Register with FastMCP
            mcp.add_tool(
                handler,
                name=tool_name,
                description=description,
                annotations=annotations,
            )
            registered += 1
            logger.debug("MCP: registered tool '%s'", tool_name)

        logger.info(
            "MCP server '%s' built with %d tools (%d read, %d write)",
            self._server_name,
            registered,
            len(self._config.tools),
            len(self._config.write_tools) if include_write_tools else 0,
        )

        # Register resources
        self._register_resources(mcp)

        # Register prompts
        self._register_prompts(mcp)

        return mcp

    def _get_tool_mcp_config(self, tool_name: str) -> MCPToolConfig | None:
        """Get MCP-specific config for a tool from the catalog."""
        try:
            entry = self._catalog.get(tool_name)
        except KeyError:
            return None

        mcp_raw = getattr(entry, "mcp", None)
        if mcp_raw is None:
            return None
        return mcp_raw

    def _register_resources(self, mcp: Any) -> None:
        """Register MCP resources from the context factory."""
        handlers = self._context_factory.get_resource_handlers()
        if not handlers:
            return

        for uri_pattern, handler in handlers.items():
            # Find matching config for description
            description = ""
            for res_cfg in self._config.resources:
                if res_cfg.uri == uri_pattern:
                    description = res_cfg.description
                    break

            mcp.resource(uri_pattern, description=description)(handler)
            logger.debug("MCP: registered resource '%s'", uri_pattern)

    def _register_prompts(self, mcp: Any) -> None:
        """Register MCP prompts from the context factory."""
        handlers = self._context_factory.get_prompt_handlers()
        if not handlers:
            return

        for name, handler in handlers.items():
            # Find matching config for description
            description = ""
            for prompt_cfg in self._config.prompts:
                if prompt_cfg.name == name:
                    description = prompt_cfg.description
                    break

            mcp.prompt(name=name, description=description)(handler)
            logger.debug("MCP: registered prompt '%s'", name)


def build_mcp_server(
    server_name: str,
    tool_registry: ToolRegistry,
    tool_catalog: ToolCatalog,
    mcp_config: MCPServerConfig,
    context_factory: MCPContextFactory,
    include_write_tools: bool = False,
    **fastmcp_kwargs: Any,
) -> Any:
    """Build a FastMCP server from the tool registry — convenience function.

    Args:
        server_name: Name for the MCP server (shown to clients).
        tool_registry: The global tool registry with registered functions.
        tool_catalog: Tool catalog loaded from tools.yaml.
        mcp_config: MCP server config (from agents.yaml ``mcp_servers``).
        context_factory: App-provided factory for tool execution context.
        include_write_tools: Enable write tools (default: read-only).

    Returns:
        A ``FastMCP`` instance ready to ``run(transport="stdio")`` or mount.

    Example::

        from sinan_agentic_core.mcp import build_mcp_server, MCPContextFactory
        from sinan_agentic_core import get_tool_registry

        class MyFactory(MCPContextFactory):
            async def create_context(self):
                return MyContext(db=await connect())
            async def cleanup(self, ctx):
                await ctx.db.close()

        server = build_mcp_server(
            server_name="My App",
            tool_registry=get_tool_registry(),
            tool_catalog=catalog,
            mcp_config=config,
            context_factory=MyFactory(),
        )
        server.run(transport="stdio")
    """
    builder = MCPServerBuilder(
        server_name=server_name,
        tool_registry=tool_registry,
        tool_catalog=tool_catalog,
        mcp_config=mcp_config,
        context_factory=context_factory,
        **fastmcp_kwargs,
    )
    return builder.build(include_write_tools=include_write_tools)
