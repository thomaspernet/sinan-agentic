"""Example: Context and Session Management

This example demonstrates how to:
1. Store data in AgentContext for workflow state
2. Use dynamic instructions that read from context
3. Update context from tools during execution
4. Manage conversation history with AgentSession

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/context_and_session.py
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agents import Agent, Runner, RunContextWrapper, function_tool

from sinan_agentic_core import AgentSession


# =============================================================================
# 1. Define Custom Context
# =============================================================================

@dataclass
class ConversationContext:
    """Context that carries data through the conversation.
    
    This data is:
    - Accessible in tools via ctx.context
    - Accessible in dynamic instructions
    - Persists across multiple agent runs
    """
    # User information (set at start)
    user_id: str
    user_name: str
    role: str = "user"
    
    # User preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Data accumulated during conversation
    facts_learned: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)
    
    # Summary of previous conversation (for long conversations)
    conversation_summary: str = ""


# =============================================================================
# 2. Dynamic Instructions Function
# =============================================================================

def build_instructions(ctx: RunContextWrapper[ConversationContext], agent: Agent) -> str:
    """Build personalized instructions based on context.
    
    This function is called each time the agent runs, allowing
    the instructions to reflect the current state of the context.
    """
    c = ctx.context
    
    instructions = f"""You are a helpful assistant for {c.user_name}.
User ID: {c.user_id}
User role: {c.role}

Your capabilities:
- Answer questions and have conversations
- Remember facts about the user (use the remember_fact tool)
- Track topics discussed (use the note_topic tool)
- Recall what you've learned about the user

Be friendly and personalized. Reference facts you've learned when relevant.
"""
    
    # Add preferences if available
    if c.preferences:
        prefs_str = ", ".join(f"{k}: {v}" for k, v in c.preferences.items())
        instructions += f"\nUser preferences: {prefs_str}"
    
    # Add learned facts
    if c.facts_learned:
        instructions += "\n\nFacts you've learned about this user:"
        for fact in c.facts_learned:
            instructions += f"\n- {fact}"
    
    # Add topics discussed
    if c.topics_discussed:
        instructions += f"\n\nTopics discussed so far: {', '.join(c.topics_discussed)}"
    
    # Add conversation summary if available
    if c.conversation_summary:
        instructions += f"\n\nPrevious conversation summary:\n{c.conversation_summary}"
    
    return instructions


# =============================================================================
# 3. Tools That Update Context
# =============================================================================

@function_tool
def remember_fact(ctx: RunContextWrapper[ConversationContext], fact: str) -> str:
    """Store a fact about the user for future reference.
    
    Use this when the user shares information about themselves,
    their preferences, or anything worth remembering.
    
    Args:
        fact: A fact to remember about the user
    """
    # Update the context - this persists across runs
    ctx.context.facts_learned.append(fact)
    return f"I'll remember that: {fact}"


@function_tool
def note_topic(ctx: RunContextWrapper[ConversationContext], topic: str) -> str:
    """Note a topic that was discussed.
    
    Args:
        topic: The topic being discussed
    """
    if topic not in ctx.context.topics_discussed:
        ctx.context.topics_discussed.append(topic)
    return f"Noted topic: {topic}"


@function_tool
def get_user_profile(ctx: RunContextWrapper[ConversationContext]) -> str:
    """Get the current user's profile from context."""
    c = ctx.context
    
    profile = f"""User Profile:
- Name: {c.user_name}
- ID: {c.user_id}
- Role: {c.role}
- Preferences: {c.preferences}
- Facts learned: {c.facts_learned}
- Topics discussed: {c.topics_discussed}"""
    
    return profile


# =============================================================================
# 4. Create Agent with Dynamic Instructions
# =============================================================================

agent = Agent[ConversationContext](
    name="memory_assistant",
    instructions=build_instructions,  # Pass function, not string!
    tools=[remember_fact, note_topic, get_user_profile],
    model="gpt-4o-mini",
)


# =============================================================================
# 5. Main Conversation Loop
# =============================================================================

async def run_conversation():
    """Run an interactive conversation with context and session management."""
    
    print("=" * 60)
    print("Context and Session Example")
    print("=" * 60)
    
    # Get user info
    user_name = input("What's your name? ").strip() or "User"
    role = input("What's your role? (user/developer/admin) ").strip() or "user"
    
    # Initialize context with user data
    context = ConversationContext(
        user_id=f"user-{hash(user_name) % 10000}",
        user_name=user_name,
        role=role,
        preferences={
            "style": "friendly",
            "detail_level": "moderate",
        },
    )
    
    # Initialize session for conversation history
    session = AgentSession(
        session_id=f"session-{context.user_id}",
        max_items=50,
    )
    
    print(f"\nHello {user_name}! I'm your assistant with memory.")
    print("I can remember facts about you and recall them later.")
    print("Try telling me about yourself, then ask what I know about you.")
    print("Type 'quit' to exit, 'status' to see context state.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "status":
            print("\n--- Context State ---")
            print(f"User: {context.user_name} ({context.user_id})")
            print(f"Role: {context.role}")
            print(f"Preferences: {context.preferences}")
            print(f"Facts learned: {context.facts_learned}")
            print(f"Topics discussed: {context.topics_discussed}")
            print(f"Session messages: {session.get_message_count()}")
            print("-------------------\n")
            continue
        
        # Add user message to session history
        await session.add_items([{"role": "user", "content": user_input}])
        
        try:
            # Get conversation history
            history = await session.get_items()
            
            # Run agent with both context and history
            result = await Runner.run(
                agent,
                input=history,      # Conversation history
                context=context,    # Workflow state
            )
            
            response = result.final_output
            
            # Add assistant response to session
            await session.add_items([{"role": "assistant", "content": response}])
            
            print(f"\nAssistant: {response}\n")
            
            # Show what was learned (for demo purposes)
            if context.facts_learned:
                print(f"  [Facts in memory: {len(context.facts_learned)}]")
            
        except Exception as e:
            print(f"\nError: {e}\n")


async def main():
    """Main entry point."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY='your-key'")
        return
    
    await run_conversation()


if __name__ == "__main__":
    asyncio.run(main())
