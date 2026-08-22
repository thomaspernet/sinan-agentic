"""Shared helpers for the ``sinan_agentic_core.core`` tests.

Several modules here script a model, or fake what one returned, and every one of
them needs the same thing: a completed assistant message carrying a given text.
The construction is the SDK's, not this package's, so it lives once and is built
from one place rather than re-derived per module.
"""

from __future__ import annotations

from openai.types.responses import ResponseOutputMessage, ResponseOutputText

ASSISTANT_ROLE = "assistant"


def assistant_message(text: str, *, message_id: str) -> ResponseOutputMessage:
    """Build the completed assistant message an SDK response carries *text* in."""
    return ResponseOutputMessage(
        id=message_id,
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role=ASSISTANT_ROLE,
        status="completed",
        type="message",
    )
