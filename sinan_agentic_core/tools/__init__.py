"""Tools package.

Define your agent tools here. Tools are functions that agents can call.

Example:
    from agents import function_tool
    from sinan_agentic_core.registry import register_tool
    
    @register_tool(
        name="my_tool",
        description="Does something useful",
        category="utility",
        parameters_description="param (str): Description",
        returns_description="Dict with results"
    )
    @function_tool
    async def my_tool(ctx, param: str) -> dict:
        # ctx.context contains your AgentContext
        # ctx.session contains your AgentSession
        return {"success": True, "data": param}
"""

__all__ = []
