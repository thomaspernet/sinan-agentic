"""RunHooks for tracking tool calls during agent execution.

Provides ``StreamingRunHooks``, a ``RunHooks`` subclass that pushes events
to an ``asyncio.Queue`` whenever a tool starts, finishes, or an agent begins
processing.  Pair it with ``chat_with_hooks()`` or use directly with
``Runner.run()``.

Usage:
    import asyncio
    from agents import Runner
    from sinan_agentic_core.services.hooks import StreamingRunHooks

    queue = asyncio.Queue()
    hooks = StreamingRunHooks(queue, tool_friendly_names={"get_weather": "Checking weather"})

    result = await Runner.run(agent, input=history, hooks=hooks)
    print(hooks.tools_called)  # ["get_weather"]
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agents import RunHooks

logger = logging.getLogger(__name__)


class StreamingRunHooks(RunHooks):
    """Track tool calls in real time via an asyncio.Queue.

    Events pushed to the queue are plain dicts::

        {"event": "tool_start", "data": {"tool": "...", "friendly_name": "...", "message": "..."}}
        {"event": "tool_end",   "data": {"tool": "...", "friendly_name": "...", "message": "..."}}
        {"event": "thinking",   "data": {"message": "Analyzing your question..."}}

    After the run completes, ``self.tools_called`` contains the names of
    every tool that was invoked (in order).
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        tool_friendly_names: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            event_queue: Queue that receives event dicts.
            tool_friendly_names: Optional ``tool_name → display name`` map.
                Falls back to replacing underscores with spaces.
        """
        self.event_queue = event_queue
        self.tool_friendly_names = tool_friendly_names or {}
        self.tools_called: List[str] = []

    def _friendly_name(self, tool_name: str) -> str:
        return self.tool_friendly_names.get(tool_name, tool_name.replace("_", " "))

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        tool_name = getattr(tool, "name", str(tool))
        friendly = self._friendly_name(tool_name)
        self.tools_called.append(tool_name)

        await self.event_queue.put({
            "event": "tool_start",
            "data": {
                "tool": tool_name,
                "friendly_name": friendly,
                "message": f"Fetching {friendly}...",
            },
        })
        logger.debug("Tool started: %s", tool_name)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        tool_name = getattr(tool, "name", str(tool))
        friendly = self._friendly_name(tool_name)

        await self.event_queue.put({
            "event": "tool_end",
            "data": {
                "tool": tool_name,
                "friendly_name": friendly,
                "message": f"Completed {friendly}",
            },
        })
        logger.debug("Tool ended: %s", tool_name)

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        await self.event_queue.put({
            "event": "thinking",
            "data": {"message": "Analyzing your question..."},
        })
        logger.debug("Agent started: %s", getattr(agent, "name", agent))
