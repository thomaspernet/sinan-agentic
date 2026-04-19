"""Example: Simple Chat Agent

This is a minimal example showing how to build and run an agent using sinan_agentic_core.
Fork this file as a starting point for your own agents.

Usage:
    python examples/simple_chat_agent.py
"""

import asyncio
import os
from datetime import datetime

# Ensure OpenAI API key is set
# export OPENAI_API_KEY="your-key"

from agents import function_tool, Agent, Runner

from sinan_agentic_core import (
    AgentDefinition,
    AgentSession,
    AgentContext,
    register_agent,
    register_tool,
    get_agent_registry,
    get_tool_registry,
)


# =============================================================================
# Step 1: Define Tools
# =============================================================================

@register_tool(
    name="get_current_time",
    description="Get the current date and time",
    category="utility",
)
@function_tool
async def get_current_time(ctx) -> dict:
    """Get the current time in ISO format."""
    return {
        "success": True,
        "current_time": datetime.now().isoformat(),
        "timezone": "local"
    }


@register_tool(
    name="calculate",
    description="Perform a mathematical calculation",
    category="utility",
)
@function_tool
async def calculate(ctx, expression: str) -> dict:
    """Safely evaluate a math expression.
    
    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"
    """
    try:
        # Safe eval for basic math
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression"}
        
        result = eval(expression)  # Only safe because we validated input
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Step 2: Define Agent
# =============================================================================

assistant_agent = AgentDefinition(
    name="assistant",
    description="A helpful AI assistant that can tell time and do math",
    instructions="""You are a friendly and helpful AI assistant.

You have access to tools that let you:
- Get the current time
- Perform mathematical calculations

When the user asks questions:
1. If they ask about time, use the get_current_time tool
2. If they ask for calculations, use the calculate tool
3. For general questions, answer directly from your knowledge

Always be helpful, concise, and friendly!
""",
    tools=["get_current_time", "calculate"],
    model="gpt-4o-mini",
)

register_agent(assistant_agent)


# =============================================================================
# Step 3: Run the Agent
# =============================================================================

async def run_chat():
    """Run an interactive chat session with the agent."""
    
    # Get registries
    agent_registry = get_agent_registry()
    tool_registry = get_tool_registry()
    
    # Get agent definition
    agent_def = agent_registry.get("assistant")
    if not agent_def:
        raise ValueError("Agent 'assistant' not found in registry")
    
    # Build list of actual tool functions
    tools = tool_registry.get_tool_functions(agent_def.tools)
    
    # Create the SDK Agent
    agent = Agent(
        name=agent_def.name,
        instructions=agent_def.instructions,
        model=agent_def.model,
        tools=tools,
    )
    
    # Create session for conversation history
    session = AgentSession(session_id="chat-session-1")
    
    print("=" * 60)
    print("🤖 Simple Chat Agent")
    print("=" * 60)
    print("Type 'quit' to exit, 'history' to see conversation history")
    print()
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye! 👋")
            break
        
        if user_input.lower() == "history":
            history = await session.get_items()
            print(f"\n📜 Conversation History ({len(history)} messages):")
            for i, msg in enumerate(history, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                print(f"  {i}. [{role}] {content}")
            print()
            continue
        
        # Add user message to session
        await session.add_items([{
            "role": "user",
            "content": user_input
        }])
        
        # Run the agent
        try:
            # Get conversation history for context
            history = await session.get_items()
            
            # Run agent with Runner
            result = await Runner.run(
                starting_agent=agent,
                input=history,  # Pass full conversation history
            )
            
            # Extract response
            response = result.final_output
            
            # Add assistant response to session
            await session.add_items([{
                "role": "assistant", 
                "content": response
            }])
            
            print(f"\n🤖 Assistant: {response}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def main():
    """Main entry point."""
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='your-key'")
        return
    
    await run_chat()


if __name__ == "__main__":
    asyncio.run(main())
