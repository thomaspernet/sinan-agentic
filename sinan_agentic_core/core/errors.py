"""Structured error handling for agent-as-tool calls.

When a sub-agent fails (exception, max_turns, validation error), the parent
agent receives a tool result. By default, the OpenAI SDK returns a generic
string like "An error occurred...". This module provides a structured JSON
error function so the parent agent can understand what went wrong and retry
with corrected input.
"""

import json
import logging
from typing import Any

from .run_errors import RunErrorKind, classify_run_error

logger = logging.getLogger(__name__)

# What the parent agent should do about each kind of run failure. Keyed by type
# rather than by message so a reworded upstream error keeps its hint.
_KIND_RETRY_HINTS: dict[RunErrorKind, str] = {
    RunErrorKind.MAX_TURNS: (
        "The sub-agent ran out of turns. Simplify the request or break it into smaller steps."
    ),
    RunErrorKind.CONTEXT_OVERFLOW: (
        "The sub-agent's context overflowed. Narrow the request so it "
        "gathers less data, or split it across several calls."
    ),
    RunErrorKind.MODEL_REFUSAL: (
        "The sub-agent refused to answer. Do not re-send the same request "
        "-- restate what you need or handle the task another way."
    ),
    RunErrorKind.MODEL_BEHAVIOR: (
        "The sub-agent's response did not match its output schema. Retry "
        "once with a simpler request."
    ),
}

# Hints for the ValueErrors this framework's own tools raise. These have no
# exception class to key off, but the wording is the framework's own -- not
# upstream's -- so matching it does not drift with an SDK release.
_TOOL_MESSAGE_HINTS: tuple[tuple[str, str], ...] = (
    (
        "not found",
        "A referenced item was not found. Verify the UUID exists "
        "in your context before retrying.",
    ),
    (
        "required",
        "A required parameter is missing. Check your context for "
        "available UUIDs and provide all required fields.",
    ),
)

_DEFAULT_RETRY_HINT = "Review the error message and retry with corrected input."


def structured_tool_error(ctx: Any, error: Exception) -> str:
    """Return a structured JSON error for agent-as-tool failures.

    The parent agent sees this as the tool result and can parse the
    structured fields to decide whether and how to retry.

    Args:
        ctx: Tool context (from OpenAI SDK)
        error: The exception that occurred

    Returns:
        JSON string with status, error_type, message, and retry_hint
    """
    error_type = type(error).__name__
    message = str(error)

    result = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "retry_hint": _retry_hint(error, message),
    }

    logger.warning("Agent-as-tool error: %s: %s", error_type, message)
    return json.dumps(result)


def _retry_hint(error: Exception, message: str) -> str:
    """Pick the retry hint for *error*, preferring its classified kind."""
    kind_hint = _KIND_RETRY_HINTS.get(classify_run_error(error))
    if kind_hint is not None:
        return kind_hint

    lowered = message.lower()
    for needle, hint in _TOOL_MESSAGE_HINTS:
        if needle in lowered:
            return hint

    return _DEFAULT_RETRY_HINT
