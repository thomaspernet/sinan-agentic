"""Streaming events and event emitter.

Event dataclasses for real-time agent workflow notifications.
Use ``StreamingHelper`` to emit events through a callback, or use
the dataclasses directly with your own transport (SSE, WebSocket, etc.).

Usage:
    from sinan_agentic_core.services.events import StreamingHelper, AgentStartEvent

    helper = StreamingHelper(event_callback=my_callback)
    helper.emit_agent_start("analyzer", iteration=1)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# -- Event dataclasses -------------------------------------------------------


@dataclass
class BaseEvent:
    """Base for all streaming events."""

    event_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {"event_type": self.event_type}


@dataclass
class AgentStartEvent(BaseEvent):
    """Emitted when an agent starts processing."""

    event_type: str = field(default="agent_start", init=False)
    agent_name: str = ""
    iteration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "iteration": self.iteration,
        }


@dataclass
class AgentCompleteEvent(BaseEvent):
    """Emitted when an agent finishes processing."""

    event_type: str = field(default="agent_complete", init=False)
    agent_name: str = ""
    iteration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "iteration": self.iteration,
        }


@dataclass
class ThinkingEvent(BaseEvent):
    """Emitted when an agent is thinking."""

    event_type: str = field(default="thinking", init=False)
    message: str = ""
    agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "agent_name": self.agent_name,
        }


@dataclass
class ToolCallEvent(BaseEvent):
    """Emitted when an agent invokes a tool."""

    event_type: str = field(default="tool_call", init=False)
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "agent_name": self.agent_name,
        }


@dataclass
class StreamingTextEvent(BaseEvent):
    """Emitted for each token / text chunk."""

    event_type: str = field(default="text_delta", init=False)
    text: str = ""
    agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "text": self.text,
            "agent_name": self.agent_name,
        }


@dataclass
class AnswerEvent(BaseEvent):
    """Emitted when the final answer is ready."""

    event_type: str = field(default="answer", init=False)
    answer: str = ""
    sources: List[Any] = field(default_factory=list)
    followup_question: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "answer": self.answer,
            "sources": self.sources,
            "followup_question": self.followup_question,
            "confidence": self.confidence,
        }


@dataclass
class ErrorEvent(BaseEvent):
    """Emitted on error."""

    event_type: str = field(default="error", init=False)
    error: str = ""
    agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "error": self.error,
            "agent_name": self.agent_name,
        }


# -- Event emitter -----------------------------------------------------------


class StreamingHelper:
    """Emit streaming events through a callback.

    This is a thin convenience layer used by the orchestrator.
    Pass any callable (sync) as ``event_callback`` — it receives a single
    event dataclass instance.

    Usage:
        def on_event(event):
            print(event.to_dict())

        helper = StreamingHelper(event_callback=on_event)
        helper.emit_agent_start("analyzer", iteration=1)
        helper.emit_answer("42", sources=["data.csv"])
    """

    def __init__(self, event_callback: Optional[Callable] = None):
        self.event_callback = event_callback

    def emit_agent_start(self, agent_name: str, iteration: int = 1) -> None:
        if self.event_callback:
            self.event_callback(AgentStartEvent(agent_name=agent_name, iteration=iteration))

    def emit_agent_complete(self, agent_name: str, iteration: int = 1) -> None:
        if self.event_callback:
            self.event_callback(AgentCompleteEvent(agent_name=agent_name, iteration=iteration))

    def emit_answer(
        self,
        answer: str,
        sources: Optional[List[Any]] = None,
        followup_question: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        if self.event_callback:
            self.event_callback(
                AnswerEvent(
                    answer=answer,
                    sources=sources or [],
                    followup_question=followup_question,
                    confidence=confidence,
                )
            )

    def emit_error(self, error_msg: str) -> None:
        if self.event_callback:
            self.event_callback(ErrorEvent(error=error_msg))
