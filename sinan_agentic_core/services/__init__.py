"""Services for agent execution and streaming.

Public API::

    # Chat functions
    from sinan_agentic_core.services import chat, chat_with_hooks, chat_streamed

    # RunHooks for tool tracking
    from sinan_agentic_core.services import StreamingRunHooks

    # Event types + emitter (used by orchestrator)
    from sinan_agentic_core.services import StreamingHelper, AgentStartEvent, ...
"""

from .events import (
    StreamingHelper,
    BaseEvent,
    AgentStartEvent,
    AgentCompleteEvent,
    ThinkingEvent,
    ToolCallEvent,
    StreamingTextEvent,
    AnswerEvent,
    ErrorEvent,
)
from .hooks import StreamingRunHooks
from .chat import chat, chat_with_hooks, chat_streamed

__all__ = [
    # Chat
    "chat",
    "chat_with_hooks",
    "chat_streamed",
    # Hooks
    "StreamingRunHooks",
    # Events
    "StreamingHelper",
    "BaseEvent",
    "AgentStartEvent",
    "AgentCompleteEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "StreamingTextEvent",
    "AnswerEvent",
    "ErrorEvent",
]
