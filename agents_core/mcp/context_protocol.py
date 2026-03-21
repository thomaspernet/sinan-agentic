"""MCP context factory protocol.

Apps implement ``MCPContextFactory`` to provide the runtime dependencies
(database connector, filters, auth) that MCP tool calls need.

Example::

    class MyContextFactory(MCPContextFactory):
        async def create_context(self):
            db = await connect_to_database()
            return MyAppContext(database_connector=db)

        async def cleanup(self, context):
            await context.database_connector.close()
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable


class MCPContextFactory(ABC):
    """Abstract factory for creating tool execution contexts.

    Each MCP tool call gets its own context (one db_connector per call).
    The factory creates it before the call and cleans up after.
    """

    @abstractmethod
    async def create_context(self) -> Any:
        """Create a context object for a single MCP tool call.

        Returns whatever the app's tools expect as their context — typically
        a dataclass with ``database_connector``, ``filters``, etc.
        """
        ...

    async def cleanup(self, context: Any) -> None:
        """Clean up after a tool call completes.

        Override to close database connections, release resources, etc.
        Default does nothing.
        """

    @asynccontextmanager
    async def tool_context(self) -> AsyncIterator[Any]:
        """Context manager wrapping create_context + cleanup."""
        ctx = await self.create_context()
        try:
            yield ctx
        finally:
            await self.cleanup(ctx)

    def get_resource_handlers(self) -> dict[str, Callable]:
        """Return MCP resource handlers: ``{uri_pattern: async_handler}``.

        Override to expose app-specific resources (documents, schemas, etc.).
        Default returns empty dict (no resources).
        """
        return {}

    def get_prompt_handlers(self) -> dict[str, Callable]:
        """Return MCP prompt handlers: ``{name: handler_fn}``.

        Override to expose app-specific prompt templates.
        Default returns empty dict (no prompts).
        """
        return {}
