"""Example: Writing a custom Capability.

A Capability is a pluggable, stateful behavior attached to an agent. The
runtime invokes its lifecycle hooks at the right moments and merges any
instruction fragments into the system prompt before each LLM call.

This example defines `ToolCallLogger`, a capability that prints every tool
call as it happens, and runs an agent end-to-end with the capability
attached via `AgentDefinition.capabilities`.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/custom_capability.py
"""

import asyncio
import os
from datetime import datetime

from agents import RunContextWrapper, Tool, function_tool

from sinan_agentic_core import (
    AgentContext,
    AgentDefinition,
    AgentSession,
    BaseAgentRunner,
    Capability,
    register_agent,
    register_tool,
)


# =============================================================================
# 1. A custom Capability: log every tool call
# =============================================================================


class ToolCallLogger(Capability):
    """Print every tool call - args in, result out.

    Demonstrates the two most common Capability hooks: `on_tool_start` and
    `on_tool_end`. Stores the call trace on `self.calls` so callers can
    inspect it after the run.
    """

    def __init__(self, prefix: str = "[tool]") -> None:
        self.prefix = prefix
        self.calls: list[tuple[str, str]] = []

    def on_tool_start(
        self,
        ctx: RunContextWrapper,
        tool: Tool,
        args: str,
    ) -> None:
        print(f"{self.prefix} -> {tool.name}({args})")

    def on_tool_end(
        self,
        ctx: RunContextWrapper,
        tool: Tool,
        result: str,
    ) -> None:
        snippet = (result[:120] + "...") if len(result) > 120 else result
        print(f"{self.prefix} <- {tool.name}: {snippet}")
        self.calls.append((tool.name, snippet))

    def reset(self) -> None:
        self.calls.clear()


# =============================================================================
# 2. A trivial tool for the agent to call
# =============================================================================


@register_tool(
    name="get_current_time",
    description="Get the current date and time",
    category="utility",
)
@function_tool
async def get_current_time(ctx) -> dict:
    """Return the current time in ISO format."""
    return {"current_time": datetime.now().isoformat(), "timezone": "local"}


# =============================================================================
# 3. Register an agent with the capability attached declaratively
# =============================================================================


# Templates declared on the AgentDefinition. The runner calls clone() on
# each one before every execute() call so concurrent runs stay isolated.
logger_capability = ToolCallLogger(prefix="[trace]")

register_agent(AgentDefinition(
    name="time_assistant",
    description="Tells the user the current time",
    instructions=(
        "You are a helpful assistant. When the user asks for the time, "
        "call the get_current_time tool and report the result."
    ),
    tools=["get_current_time"],
    model="gpt-4o-mini",
    capabilities=[logger_capability],
))


# =============================================================================
# 4. Run the agent
# =============================================================================


async def run() -> None:
    runner = BaseAgentRunner()
    context = AgentContext()
    session = AgentSession(session_id="custom-cap-demo")

    print("Asking the agent for the time...\n")
    output = await runner.execute(
        agent_name="time_assistant",
        context=context,
        session=session,
        input_text="What time is it right now?",
    )

    print(f"\nAgent response: {output}")

    # The template is untouched - the runner ran on a clone. Either inspect
    # the clone via streaming events, or pass a fresh ToolCallLogger as a
    # runtime kwarg if you need to read its trace after the run.
    print(f"\nTemplate ToolCallLogger.calls (still empty - clones are isolated):"
          f" {logger_capability.calls}")


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - skipping live agent run.")
        print("Export an API key and re-run to see the capability in action:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python examples/custom_capability.py")
        return

    await run()


if __name__ == "__main__":
    asyncio.run(main())
