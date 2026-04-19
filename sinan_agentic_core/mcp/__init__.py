"""MCP server capabilities for sinan_agentic_core.

Build an MCP server from the existing tool registry with zero duplication —
descriptions, schemas, and annotations all come from tools.yaml.

Requires the ``mcp`` extra::

    pip install 'sinan_agentic_core[mcp]'

Quick start::

    from sinan_agentic_core.mcp import build_mcp_server, MCPContextFactory

    class MyFactory(MCPContextFactory):
        async def create_context(self):
            return MyContext(db=await connect_db())
        async def cleanup(self, ctx):
            await ctx.db.close()

    server = build_mcp_server(
        server_name="My App",
        tool_registry=get_tool_registry(),
        tool_catalog=tool_catalog,
        mcp_config=mcp_config,
        context_factory=MyFactory(),
    )
    server.run(transport="stdio")
"""

from .context_protocol import MCPContextFactory
from .server_builder import MCPServerBuilder, build_mcp_server
from .yaml_schema import (
    MCPAnnotationsConfig,
    MCPPromptConfig,
    MCPResourceConfig,
    MCPServerConfig,
    MCPToolConfig,
)

__all__ = [
    "MCPContextFactory",
    "MCPServerBuilder",
    "build_mcp_server",
    "MCPAnnotationsConfig",
    "MCPPromptConfig",
    "MCPResourceConfig",
    "MCPServerConfig",
    "MCPToolConfig",
]
