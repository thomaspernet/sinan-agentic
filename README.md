# Agents Core

A framework for building AI agents using the OpenAI Agents SDK. Fork this repository to quickly create agent-based applications.

## Features

- **Declarative Agents** - Define agents as data, not code
- **Registry Pattern** - Central registries for agents, tools, and guardrails
- **Agent Factory** - `create_agent_from_registry()` builds an `Agent` in one call
- **Chat Service** - `chat()`, `chat_with_hooks()`, `chat_streamed()` for API endpoints
- **Token Usage Tracking** - Every chat/run returns token counts (input, output, cached, reasoning)
- **RunHooks** - Track tool calls in real time via `StreamingRunHooks`
- **Session** - In-memory or SQLite-backed conversation history
- **Dynamic Context** - Pass runtime data to instructions and tools
- **InstructionBuilder** - Base class for section-based instruction assembly (persona, context, steps, rules, output)
- **Output Models** - `ToolOutput` and `ChatResponse` dataclasses

## Installation

```bash
pip install git+https://github.com/thomaspernet/package_agentic.git
export OPENAI_API_KEY="your-key"
```

## Usage

### 1. Register a tool and an agent

```python
from agents import function_tool
from agents_core import register_tool, AgentDefinition, register_agent

@register_tool(name="get_weather", description="Get weather for a city",
               category="api", parameters_description="city (str)", returns_description="dict")
@function_tool
async def get_weather(ctx, city: str) -> dict:
    return {"temperature": 72, "conditions": "sunny"}

register_agent(AgentDefinition(
    name="weather_assistant",
    description="Helps with weather queries",
    instructions="You help users get weather information.",
    tools=["get_weather"],
    model="gpt-4o-mini",
))
```

### 2. Run the agent

```python
from agents import Runner
from agents_core import create_agent_from_registry

agent = create_agent_from_registry("weather_assistant")
result = await Runner.run(agent, "What's the weather in Paris?")
print(result.final_output)
```

## Chat Service

Ready-to-use functions for API endpoints. All accept an optional `context=` parameter that gets forwarded to `Runner.run()`, making it available to dynamic instructions and tools via `RunContextWrapper`.

```python
from agents_core import chat, chat_with_hooks, chat_streamed, AgentSession

session = AgentSession(session_id="user-123")

# --- Non-streaming ---
result = await chat("What's the weather?", "weather_assistant", session)
# {"success": True, "response": "...", "session_id": "...", "tools_called": [...], "usage": {...}}

# --- Streaming with tool notifications (RunHooks) ---
async for event in chat_with_hooks("What's the weather?", "weather_assistant", session,
                                    tool_friendly_names={"get_weather": "Checking weather"}):
    if event["event"] == "tool_start":
        print(f"Tool: {event['data']['friendly_name']}")
    elif event["event"] == "answer":
        print(event["data"]["response"])

# --- Token-level streaming ---
async for event in chat_streamed("What's the weather?", "weather_assistant", session):
    if event["event"] == "text_delta":
        print(event["data"]["delta"], end="", flush=True)
    elif event["event"] == "answer":
        print(f"\nTools used: {event['data']['tools_called']}")
```

## Token Usage Tracking

All three chat functions and `BaseAgentRunner.run_agent()` return token usage automatically.

```python
result = await chat("What's the weather?", "weather_assistant", session)
print(result["usage"])
# {
#     "requests": 2,
#     "input_tokens": 1500,
#     "output_tokens": 350,
#     "total_tokens": 1850,
#     "input_tokens_details": {"cached_tokens": 200},
#     "output_tokens_details": {"reasoning_tokens": 0},
# }

# Streaming functions include usage in the final "answer" event:
async for event in chat_streamed("Hello", "my_agent", session):
    if event["event"] == "answer":
        print(f"Tokens used: {event['data']['usage']['total_tokens']}")
```

## Dynamic Context

Pass runtime data to agent instructions and tools via `context=`.

```python
from dataclasses import dataclass
from agents import Agent, Runner, RunContextWrapper

@dataclass
class UserContext:
    user_name: str
    language: str = "English"

def dynamic_instructions(ctx: RunContextWrapper[UserContext], agent: Agent) -> str:
    return f"You are a helpful assistant for {ctx.context.user_name}. Respond in {ctx.context.language}."

agent = Agent[UserContext](name="assistant", instructions=dynamic_instructions, model="gpt-4o-mini")

result = await Runner.run(agent, "Hello!", context=UserContext(user_name="Thomas"))

# Works the same with chat functions:
result = await chat("Hello!", "my_agent", session, context=UserContext(user_name="Thomas"))
```

## InstructionBuilder

Build agent system instructions with a consistent section pattern instead of ad-hoc string concatenation. Subclass `InstructionBuilder`, override the sections you need, and skip the rest.

```python
from agents_core import InstructionBuilder, AgentDefinition, register_agent

class MyAgentBuilder(InstructionBuilder):
    def __init__(self, context, agent_def):
        super().__init__(context, agent_def)
        self._config = self._ctx_attr("my_config", {})

    def persona(self):
        return self.format_persona("data analyst", self._config.get("persona"))

    def steps(self):
        return self.format_steps(["Load the dataset.", "Analyze trends.", "Produce output."])

    def rules(self):
        return self.format_rules(["Do not fabricate data.", "Cite sources."])

    def output_format(self):
        return 'Output: {"analysis": "...", "confidence": 0.95}'

register_agent(AgentDefinition(
    name="analyst",
    description="Analyzes data",
    instructions=MyAgentBuilder.callable(),  # (context, agent_def) -> str
    tools=["read_data"],
))
```

`build()` assembles sections in order (persona → context → steps → rules → output), skips any that return `None`, and joins with double newlines. Override `sections()` to reorder, or `extra_sections()` to append additional `(header, body)` blocks.

For agents with fundamentally different instruction paths (e.g., surface vs deep extraction), use a shared private base with concrete subclasses and a dispatcher function — see `agents_core/instructions/builder.py` docstring for details.

## Session Persistence

```python
from agents_core import AgentSession, SQLiteSessionStore

# In-memory (default)
session = AgentSession(session_id="user-123")
await session.add_items([{"role": "user", "content": "Hello!"}])
history = await session.get_items()

# SQLite (persistent)
store = SQLiteSessionStore("data/conversations.db")
store.add_message("session-123", "user", "What's the weather?")
store.add_message("session-123", "assistant", "It's sunny.")
history = store.get_conversation_history("session-123")  # [{"role": ..., "content": ...}]
store.archive_session("session-123")   # Archive (keeps data)
store.clear_session("session-123")     # Delete permanently
```

## Output Models

```python
from agents_core import ToolOutput, ChatResponse

output = ToolOutput(success=True, data={"temp": 72}, metadata={"source": "api"})
response = ChatResponse(success=True, response="It's sunny.", session_id="u-123", tools_called=["get_weather"])

output.to_dict()    # {"success": True, "data": {...}, "metadata": {...}}
response.to_dict()  # {"success": True, "response": "...", ...}
```

## Project Structure

```
agents_core/
├── __init__.py              # Main exports
├── orchestrator.py          # Multi-agent orchestration
├── instructions/
│   └── builder.py           # InstructionBuilder base class
├── registry/
│   ├── agent_registry.py    # AgentDefinition + registry
│   ├── agent_factory.py     # create_agent_from_registry()
│   ├── tool_registry.py     # ToolDefinition + registry
│   └── guardrail_registry.py
├── services/
│   ├── chat.py              # chat(), chat_with_hooks(), chat_streamed()
│   ├── hooks.py             # StreamingRunHooks (tool call tracking)
│   └── events.py            # Event dataclasses + StreamingHelper
├── session/
│   ├── agent_session.py     # In-memory session
│   └── sqlite_store.py      # SQLite persistence
├── models/
│   ├── context.py           # AgentContext
│   └── outputs/             # ToolOutput, ChatResponse
├── agents/                  # Your agent definitions
├── tools/                   # Your tool implementations
└── guardrails/              # Input validation
```

## License

MIT License
