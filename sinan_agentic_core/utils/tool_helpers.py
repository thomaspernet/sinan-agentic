"""Generic tool helper functions for agent tools.

These utilities eliminate boilerplate that every agent tool needs:
- Context unwrapping (OpenAI Agents SDK wraps context in RunContextWrapper)
- JSON response formatting for LLM consumption
- JSON error formatting for LLM consumption
"""

import json
from typing import Any


def unwrap_context(ctx: Any) -> Any:
    """Extract the actual AgentContext from a RunContextWrapper.

    The OpenAI Agents SDK wraps the context in a RunContextWrapper.
    Every tool function receives this wrapper instead of the raw context.
    This helper unwraps it safely.
    """
    return ctx.context if hasattr(ctx, "context") else ctx


def tool_response(data: Any) -> str:
    """Serialize data to JSON for tool output.

    Uses default=str to handle non-serializable types (datetime, UUID, etc.).
    """
    return json.dumps(data, default=str)


def tool_error(message: str, **extra: Any) -> str:
    """Return a JSON error string for tool output.

    Args:
        message: Human-readable error description.
        **extra: Additional key-value pairs to include in the error object.
    """
    payload: dict[str, Any] = {"error": message}
    if extra:
        payload.update(extra)
    return json.dumps(payload, default=str)
