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

logger = logging.getLogger(__name__)


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

    # Provide actionable retry hints based on error type
    if "Max turns" in message:
        retry_hint = (
            "The sub-agent ran out of turns. Simplify the request "
            "or break it into smaller steps."
        )
    elif "not found" in message.lower():
        retry_hint = (
            "A referenced item was not found. Verify the UUID exists "
            "in your context before retrying."
        )
    elif "required" in message.lower():
        retry_hint = (
            "A required parameter is missing. Check your context for "
            "available UUIDs and provide all required fields."
        )
    else:
        retry_hint = "Review the error message and retry with corrected input."

    result = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "retry_hint": retry_hint,
    }

    logger.warning("Agent-as-tool error: %s: %s", error_type, message)
    return json.dumps(result)
