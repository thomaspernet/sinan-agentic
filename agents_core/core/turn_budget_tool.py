"""Turn budget extension tool for agent self-approval.

This tool is automatically injected into agents that use a TurnBudget.
The agent calls it when running low on turns and needs more time.
"""

from agents import function_tool

from ..utils.tool_helpers import unwrap_context, tool_response, tool_error


@function_tool(name_override="request_extension")
async def request_extension_tool(ctx, reason: str) -> str:
    """Request additional turns when the current budget is insufficient.

    Call this when you need more turns to complete the task. You must provide
    a reason explaining why the extension is needed. Extensions are self-approved
    but limited in number.

    Args:
        reason: Why additional turns are needed (e.g., "Need to process 3 more documents").
    """
    context = unwrap_context(ctx)
    budget = getattr(context, "_turn_budget", None)
    if budget is None:
        return tool_error("No turn budget is configured for this agent.")

    success, message = budget.request_extension(reason)
    if success:
        return tool_response({"status": "approved", "message": message, "remaining": budget.remaining})
    return tool_error(message)
