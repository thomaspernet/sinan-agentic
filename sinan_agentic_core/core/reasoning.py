"""What a model says about its own thinking, as stream events.

A reasoning model narrates its work only when the caller asks for it —
``Reasoning(summary=...)`` in the run's ``model_settings``. When it does, the
narration arrives in parts, one thought per part.

Both streaming paths — ``BaseAgentRunner._execute_streamed`` and
``services.chat.chat_streamed`` — report it through the builders here, so the
event names and payload shape are decided once rather than once per path.
"""

from typing import Any

from openai.types.responses import (
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)

__all__ = [
    "reasoning_event",
    "reasoning_stream_event",
    "reasoning_summary_texts",
    "run_reasoning_texts",
]


def reasoning_stream_event(data: Any) -> dict[str, Any] | None:
    """The event one raw SDK response event carries, or ``None`` for nothing.

    ``None`` covers both an event of no interest here and an empty text delta —
    the same bar the answer text is held to, since an empty chunk says nothing.
    """
    if isinstance(data, ResponseReasoningSummaryTextDeltaEvent):
        if not data.delta:
            return None
        return {
            "event": "reasoning_delta",
            "data": {"delta": data.delta, "index": data.summary_index},
        }
    if isinstance(data, ResponseReasoningSummaryPartAddedEvent):
        return {"event": "reasoning_part_added", "data": {"index": data.summary_index}}
    if isinstance(data, ResponseReasoningSummaryPartDoneEvent):
        return {
            "event": "reasoning_part_done",
            "data": {"index": data.summary_index, "text": data.part.text},
        }
    return None


def reasoning_event(summary: list[str]) -> dict[str, Any]:
    """The terminal event repeating every finished part of one summary.

    Not redundant with the deltas: it is the same guarantee ``answer`` gives for
    the response text, so a consumer that dropped a fragment still ends up with
    exactly what was said.
    """
    return {"event": "reasoning", "data": {"summary": summary}}


def reasoning_summary_texts(item: Any) -> list[str]:
    """The finished summary parts of one reasoning item, in order.

    A reasoning model emits a reasoning item on every turn whether or not a
    summary was asked for, and the parts carry text only when it was. An empty
    list therefore means the model said nothing out loud, not that something was
    lost — callers skip the event rather than delivering an empty one.
    """
    raw = getattr(item, "raw_item", None)
    parts = getattr(raw, "summary", None) or []
    return [text for part in parts if (text := getattr(part, "text", ""))]


def run_reasoning_texts(result: Any) -> list[str]:
    """Every reasoning summary a finished run produced, in order."""
    return [
        text
        for item in result.new_items
        if getattr(item, "type", None) == "reasoning_item"
        for text in reasoning_summary_texts(item)
    ]
