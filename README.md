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
- **Agent Catalog** - Load agent definitions (model, description, tools) from a YAML file
- **Knowledge Store** - Inject domain knowledge from YAML files into agent system prompts
- **Structured Agent-as-Tool** - Typed input schemas and structured error handling for sub-agent calls

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

`build()` assembles sections in order (persona → domain_knowledge → context → steps → rules → output), skips any that return `None`, and joins with double newlines. Override `sections()` to reorder, or `extra_sections()` to append additional `(header, body)` blocks.

For agents with fundamentally different instruction paths (e.g., surface vs deep extraction), use a shared private base with concrete subclasses and a dispatcher function — see `agents_core/instructions/builder.py` docstring for details.

## Agent Catalog (YAML-driven agent config)

Keep static agent config (model, description, tools) in a YAML file instead of scattered across Python files. Dynamic parts (instructions, output_dataclass, hosted_tools) stay in Python.

### Basic usage

```yaml
# agents.yaml
agents:
  weather_assistant:
    model: gpt-4o-mini
    description: Helps with weather queries
    tools:
      - get_weather
      - get_forecast
```

```python
from agents_core import load_agent_catalog, AgentDefinition, register_agent

catalog = load_agent_catalog("agents.yaml")
cfg = catalog.get("weather_assistant")
# cfg.model -> "gpt-4o-mini"
# cfg.tools -> ["get_weather", "get_forecast"]

register_agent(AgentDefinition(
    name="weather_assistant",
    model=cfg.model,
    description=cfg.description,
    tools=cfg.tools,
    instructions=build_weather_instructions,  # dynamic — stays in Python
))
```

### Tool groups

Define reusable tool sets once, reference them with `group:`.

```yaml
tool_groups:
  graph_navigation:
    - discover
    - search
    - explore
    - read
  project_knowledge:
    - get_rules
    - get_skills

agents:
  research_agent:
    model: gpt-4o
    description: Deep research agent
    tools:
      - think
      - group: graph_navigation      # expands to 4 tools
      - group: project_knowledge     # expands to 2 tools
```

### Conditional tools

Include a tool only when a config flag is truthy. Pass your config object to `catalog.get()` -- the `when:` path is resolved via `getattr()`.

```yaml
agents:
  chatbot:
    model: gpt-4o
    description: Main assistant
    tools:
      - think
      - search
      - tool: web_search
        when: features.web_search_enabled   # dot-path into your config
```

```python
cfg = catalog.get("chatbot", config=my_config)
# If my_config.features.web_search_enabled is True:
#   cfg.tools -> ["think", "search", "web_search"]
# If False or missing:
#   cfg.tools -> ["think", "search"]
```

### Agent-level conditions

Control entire agent registration with a `when:` clause. Use `catalog.is_enabled()` to check.

```yaml
agents:
  web_search_agent:
    model: gpt-4o-mini
    when: features.web_search_enabled
    description: Search the internet
    tools: []
```

```python
if catalog.is_enabled("web_search_agent", config=my_config):
    cfg = catalog.get("web_search_agent", config=my_config)
    register_agent(AgentDefinition(
        name="web_search_agent",
        model=cfg.model,
        description=cfg.description,
        tools=cfg.tools,
        hosted_tools=[get_web_search_tool],  # SDK-specific — stays in Python
    ))
```

### API reference

| Method | Returns | Description |
|--------|---------|-------------|
| `catalog.get(name, config=None)` | `AgentYamlEntry` | Resolved entry (groups expanded, conditions evaluated) |
| `catalog.is_enabled(name, config=None)` | `bool` | Check agent-level `when` condition |
| `catalog.list_agents()` | `list[str]` | All agent names in the catalog |

## Knowledge Store

Inject domain knowledge into agent system prompts from separate YAML files. Knowledge teaches agents "what things are" (domain model), while tool descriptions handle routing. Inspired by the CLAUDE.md pattern — static context that shapes agent behavior.

### Setup

Create a `knowledge/` directory with one YAML file per scope:

```yaml
# knowledge/global.yaml
content: |
  The workspace is a tree of items. Pages are containers that can nest
  other pages or hold documents. Documents are imported content.

# knowledge/extraction.yaml
content: |
  Named entities are shared and deduplicated across documents.
  Insights are per-document observations with supporting evidence.
```

Reference scopes in your `agents.yaml`:

```yaml
agents:
  extraction_agent:
    model: gpt-4o
    description: Extracts structured data
    knowledge:
      - global        # -> knowledge/global.yaml
      - extraction    # -> knowledge/extraction.yaml
    tools:
      - extract_entities
```

### Loading

Pass `knowledge_dir` to `load_agent_catalog()`:

```python
from agents_core import load_agent_catalog

catalog = load_agent_catalog("agents.yaml", knowledge_dir="knowledge/")
cfg = catalog.get("extraction_agent")
cfg.knowledge_text  # global + extraction content, concatenated with \n\n
```

Then wire it into your `AgentDefinition`:

```python
register_agent(AgentDefinition(
    name="extraction_agent",
    model=cfg.model,
    description=cfg.description,
    tools=cfg.tools,
    knowledge_text=cfg.knowledge_text,  # injected into system prompt
    instructions=build_extraction_instructions,
))
```

### How it reaches the system prompt

`InstructionBuilder.domain_knowledge()` reads `agent_def.knowledge_text` and places it between persona and context sections. The section order is:

1. **persona** — identity statement
2. **domain_knowledge** — static domain knowledge from YAML files
3. **context_section** — runtime environment info
4. **steps** — task instructions
5. **rules** — constraints
6. **output_format** — expected output structure

Override `domain_knowledge()` in your builder subclass for custom behavior.

## Structured Agent-as-Tool

When using sub-agents via the SDK's `as_tool()` pattern, you can enforce structured input and get structured error responses instead of freeform text.

### Structured input parameters

Define a dataclass for the sub-agent's input schema. The parent LLM is forced to fill in the required fields before calling the sub-agent.

```python
from dataclasses import dataclass
from agents_core import AgentDefinition, register_agent

@dataclass
class WriterRequest:
    operation: str      # exact tool name to execute
    target_id: str      # primary target identifier
    payload: str = ""   # additional params as JSON

register_agent(AgentDefinition(
    name="writer_agent",
    description="Execute write operations on the database.",
    instructions="You are a write-operation executor.",
    tools=["create_record", "update_record", "delete_record"],
    as_tool_parameters=WriterRequest,  # enforced input schema
))
```

When the parent agent calls `writer_agent`, the LLM must provide `operation`, `target_id`, and optionally `payload` — no more ambiguous freeform text.

### Structured error handling

Sub-agent failures automatically return structured JSON instead of generic error strings:

```json
{
    "status": "error",
    "error_type": "ValueError",
    "message": "target_id is required",
    "retry_hint": "A required parameter is missing. Check your context for available IDs and provide all required fields."
}
```

This is handled by `structured_tool_error()`, which is automatically wired into all agent-as-tool calls via `BaseAgentRunner`. Retry hints are generated based on error type:

| Error pattern | Retry hint |
|---------------|------------|
| "Max turns" in message | Simplify the request or break into smaller steps |
| "not found" in message | Verify the ID exists in your context |
| "required" in message | Check context for available IDs and provide all required fields |
| Other | Review the error message and retry with corrected input |

### How it works

`BaseAgentRunner._build_tools()` automatically:
1. Passes `as_tool_parameters` to `agent.as_tool(parameters=...)` when defined
2. Passes `structured_tool_error` as `failure_error_function` for all agent-as-tool calls
3. If `as_tool_turn_budget` is set, wires up `TurnBudgetHooks` and dynamic instructions (see [Turn budget for sub-agents](#turn-budget-for-sub-agents-agent-as-tool))

No manual wiring needed — just set `as_tool_parameters` on your `AgentDefinition`.

## Tool Error Recovery

When a tool returns an error, agents often retry with identical parameters, wasting turns in a loop. `ToolErrorRecovery` solves this by tracking tool errors and injecting progressive recovery guidance into the agent's instructions — the same dynamic-instructions pattern used by `TurnBudget`.

### How it works

1. **`on_tool_end` hook** tracks tool results. When a result contains `{"error": "..."}`, the tool name, arguments, and error are recorded.
2. **Dynamic instructions** inject a `## Tool Error Recovery` section before each LLM call, telling the agent what failed and how to recover.
3. **Progressive escalation** — guidance gets more directive with each repeated failure:
   - **1st failure**: Show the error and recovery hint
   - **2nd failure (same args)**: Warn that the agent is repeating the same failing call
   - **3rd failure (same args)**: Tell the agent to stop retrying and move on

### Registering recovery hints

Add a `recovery_hint` to your tool registration. This static hint is shown to the agent whenever the tool errors — no need to handle each error case individually:

```python
@register_tool(
    name="search_database",
    description="Search the database by query",
    category="search",
    parameters_description="query (str): Search query. scope (str): 'local' or 'global'.",
    returns_description="JSON with results",
    recovery_hint="Requires a non-empty query. Use scope='local' for fast search, 'global' for external APIs.",
)
@function_tool
async def search_database(ctx, query: str, scope: str = "local") -> str:
    ...
```

For MCP tools (where you don't control the code), pass hints via config:

```python
recovery = ToolErrorRecovery(
    tool_registry=registry,
    mcp_hints={
        "mcp_arxiv_search": "Query must be at least 2 characters.",
        "mcp_slack_post": "Channel ID is required, not channel name.",
    },
)
```

### Usage with BaseAgentRunner

Pass `error_recovery` to `execute()`:

```python
from agents_core import BaseAgentRunner, ToolErrorRecovery

runner = BaseAgentRunner()
recovery = ToolErrorRecovery(tool_registry=runner.tool_registry)

output = await runner.execute(
    agent_name="my_agent",
    context=context,
    session=session,
    input_text="Find recent papers on transformers",
    error_recovery=recovery,
)

# After execution, inspect error state:
if recovery.has_errors:
    print(recovery.get_error_summary())
```

### Combining with Turn Budget

Both features compose automatically. When both are active, the agent sees both sections in its instructions:

```python
from agents_core import BaseAgentRunner, TurnBudget, ToolErrorRecovery

runner = BaseAgentRunner()
budget = TurnBudget(default_turns=15)
recovery = ToolErrorRecovery(tool_registry=runner.tool_registry)

output = await runner.execute(
    agent_name="research_agent",
    context=context,
    session=session,
    input_text="Analyze recent publications",
    turn_budget=budget,
    error_recovery=recovery,
)
```

The runner composes both hook sets into a single `_CompositeHooks` and chains both dynamic instruction sections.

### What the agent sees

After a tool error, the agent's system prompt includes:

```text
## Tool Error Recovery
- search_database returned error: "No results found" (called with: query=transformrs, scope=local)
  Recovery hint: Requires a non-empty query. Use scope='local' for fast search, 'global' for external APIs.

General rule: Never retry a tool call with identical parameters. If a tool fails, read the error and choose a different approach.
```

After repeated identical failures:

```text
## Tool Error Recovery
- STOP: search_database has failed 3 times with the same arguments. Do NOT call this tool again.
  Error was: "Connection timeout"
  Move on to your next task, try a completely different approach, or return your partial results.
```

### Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tool_registry` | None | ToolRegistry for looking up `recovery_hint` |
| `mcp_hints` | `{}` | Dict mapping MCP tool names to hint strings |
| `max_identical_before_stop` | 3 | Stop threshold for identical-argument retries |

## Turn Budget (Soft Turn Management)

The OpenAI Agents SDK enforces `max_turns` as a hard cutoff -- when hit, everything stops with no graceful handling. `TurnBudget` adds a soft budget layer where the agent self-manages its turns:

- **Default budget** -- the agent perceives a soft turn limit (e.g., 10 turns)
- **Hard ceiling** -- the SDK's `max_turns` is set to a higher absolute maximum (safety net)
- **Warnings** -- the agent gets notified when running low on turns
- **Self-extension** -- the agent can call `request_extension()` to approve more turns for itself

### Concept

```text
TurnBudget(default_turns=10)
  |
  |-- Agent perceives 10 turns (soft limit)
  |-- SDK gets max_turns=25 (hard ceiling, invisible to agent)
  |-- At turn 8: "2 turns remaining, wrap up or request_extension()"
  |-- Agent calls request_extension("processing remaining docs") -> budget extends to 15
  |-- If truly exhausted: error handler provides graceful fallback
```

### YAML configuration

Define turn budgets per agent in `agents.yaml`. The `max_turns` field becomes the hard ceiling (`absolute_max`), and `turn_budget` controls the soft budget:

```yaml
agents:
  research_agent:
    model: gpt-4o
    max_turns: 25            # hard ceiling (SDK safety net)
    description: Deep research
    tools: [think, search, read]
    turn_budget:
      default_turns: 10      # soft limit the agent perceives
      reminder_at: 2         # warn when 2 turns remain
      max_extensions: 3      # agent can self-extend up to 3 times
      extension_size: 5      # turns added per extension

  simple_agent:
    model: gpt-4o-mini
    max_turns: 10
    description: Quick tasks
    tools: [think]
    # No turn_budget -- uses plain max_turns cutoff
```

Then use `build_turn_budget()` to create a `TurnBudget` from the catalog entry:

```python
from agents_core import load_agent_catalog, BaseAgentRunner

catalog = load_agent_catalog("agents.yaml")
cfg = catalog.get("research_agent")
budget = cfg.build_turn_budget()  # TurnBudget or None if not configured

runner = BaseAgentRunner()
output = await runner.execute(
    agent_name="research_agent",
    context=context,
    session=session,
    input_text="Analyze these 10 papers",
    max_turns=cfg.max_turns,
    turn_budget=budget,
)
```

### Programmatic usage

You can also create a `TurnBudget` directly:

```python
from agents_core import BaseAgentRunner, TurnBudget

budget = TurnBudget(
    default_turns=10,
    reminder_at=2,
    max_extensions=3,
    extension_size=5,
    absolute_max=25,
)

output = await runner.execute(
    agent_name="research_agent",
    context=context,
    session=session,
    input_text="Analyze these 10 papers",
    turn_budget=budget,
)

# After execution, inspect budget state:
print(budget.turns_used)        # 14
print(budget.extensions_used)   # 1
print(budget.extension_reasons) # ["Need to process 5 remaining papers"]
```

### How instructions are injected

When a `TurnBudget` is provided, the runner:

1. Sets the SDK's `max_turns` to `budget.absolute_max` (hard safety ceiling)
2. Attaches `budget` to the context as `_turn_budget`
3. Adds the `request_extension` tool to the agent
4. Wraps agent instructions as a dynamic callable that appends budget status each turn
5. Uses `TurnBudgetHooks` (a `RunHooks` subclass) to count turns via `on_llm_start`

If using `InstructionBuilder`, the `turn_budget_section()` method is included in the default section order and automatically reads the budget from context.

### Turn budget for sub-agents (agent-as-tool)

Sub-agents running via `.as_tool()` can have their own independent turn budget. Set `as_tool_turn_budget` on the `AgentDefinition` instead of `as_tool_max_turns`:

```yaml
agents:
  writer_agent:
    model: gpt-4o-mini
    max_turns: 10
    turn_budget:
      default_turns: 5
      reminder_at: 1
      max_extensions: 1
      extension_size: 3
```

```python
cfg = catalog.get("writer_agent")

register_agent(AgentDefinition(
    name="writer_agent",
    description="Execute write operations on the database.",
    instructions="You are a write-operation executor.",
    tools=["create_record", "update_record", "delete_record"],
    as_tool_turn_budget=cfg.build_turn_budget(),  # budget instead of as_tool_max_turns
))
```

When `as_tool_turn_budget` is set, `_build_tools()` automatically:
1. Resets the budget for each parent agent creation
2. Makes the sub-agent's instructions dynamic (budget status injected each turn)
3. Adds the `request_extension` tool to the sub-agent
4. Passes `TurnBudgetHooks` to `.as_tool(hooks=...)` for turn tracking
5. Sets `max_turns` to `budget.absolute_max` (overrides `as_tool_max_turns`)

If `as_tool_turn_budget` is not set, the runner falls back to `as_tool_max_turns` (plain hard cap).

### Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_turns` | 10 | Soft budget the agent perceives |
| `reminder_at` | 2 | Warn when this many turns remain |
| `max_extensions` | 3 | Max self-approved extensions |
| `extension_size` | 5 | Turns added per extension |
| `absolute_max` | 25 | Hard ceiling passed to SDK |

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

```text
agents_core/
├── __init__.py              # Main exports
├── orchestrator.py          # Multi-agent orchestration
├── core/
│   ├── base_runner.py       # BaseAgentRunner
│   ├── errors.py            # structured_tool_error for agent-as-tool failures
│   ├── turn_budget.py       # TurnBudget + TurnBudgetHooks (soft turn management)
│   └── turn_budget_tool.py  # request_extension tool (agent self-approval)
├── instructions/
│   └── builder.py           # InstructionBuilder base class
├── registry/
│   ├── agent_catalog.py     # YAML-driven agent catalog (load_agent_catalog)
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
