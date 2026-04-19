# Sinan (司南)

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Model_Si_Nan_of_Han_Dynasty.jpg/250px-Model_Si_Nan_of_Han_Dynasty.jpg" alt="Sinan — a Han dynasty south-pointing spoon on a bronze plate" width="220" align="right" />

> **sinan (司南)** — the earliest known compass. A lodestone carved into a spoon, resting on a bronze plate inscribed with the 24 directions. Han dynasty China, ~2nd century BCE.
>
> Like its namesake, this framework is the instrument that helps you align yourself with our agentic logic — the plate holds the field of tools, knowledge, and rules; the spoon is the agent that always knows which way to turn.

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
- **Tool Catalog** - Load tool metadata (description, category, recovery hints) from a YAML file
- **Knowledge Store** - Inject domain knowledge from YAML files into agent system prompts
- **Structured Agent-as-Tool** - Typed input schemas and structured error handling for sub-agent calls
- **MCP Server** - Expose registered tools as an MCP server (stdio or HTTP) with zero description duplication — all metadata comes from `tools.yaml`

## Installation

```bash
pip install git+https://github.com/thomaspernet/sinan-agentic.git
export OPENAI_API_KEY="your-key"
```

## Usage

### 1. Register a tool and an agent

```python
from agents import function_tool
from sinan_agentic_core import register_tool, AgentDefinition, register_agent

@register_tool(name="get_weather")
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
from sinan_agentic_core import create_agent_from_registry

agent = create_agent_from_registry("weather_assistant")
result = await Runner.run(agent, "What's the weather in Paris?")
print(result.final_output)
```

## Chat Service

Ready-to-use functions for API endpoints. All accept an optional `context=` parameter that gets forwarded to `Runner.run()`, making it available to dynamic instructions and tools via `RunContextWrapper`.

```python
from sinan_agentic_core import chat, chat_with_hooks, chat_streamed, AgentSession

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
from sinan_agentic_core import InstructionBuilder, AgentDefinition, register_agent

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

For agents with fundamentally different instruction paths (e.g., surface vs deep extraction), use a shared private base with concrete subclasses and a dispatcher function — see `sinan_agentic_core/instructions/builder.py` docstring for details.

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
from sinan_agentic_core import load_agent_catalog, AgentDefinition, register_agent

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

## Tool Catalog (YAML-driven tool metadata)

Keep static tool metadata (description, category, parameters, recovery hints) in a YAML file instead of repeating it in every `@register_tool()` decorator. The Python decorator becomes minimal — just a name linking the function to its YAML entry.

### Basic usage

```yaml
# tools.yaml
tools:
  search_database:
    description: >-
      Search the database for records matching a query.
      Supports semantic and keyword search modes.
    category: search
    parameters_description: "query (str): Search text. search_type (str): 'semantic' or 'keyword'."
    returns_description: "JSON with matching records"
    recovery_hint: "Requires a non-empty query. Prefer search_type='semantic' for natural language."

  read_record:
    description: Read a single record by UUID
    category: search
    parameters_description: "uuid (str): Record UUID"
    returns_description: "JSON with record data"
    recovery_hint: "Verify the UUID is valid. Use search_database to find the correct UUID."
```

```python
from agents import function_tool
from sinan_agentic_core import register_tool, load_tool_catalog, get_tool_registry

# Decorator is minimal — just links function to name
@register_tool(name="search_database")
@function_tool
async def search_database(ctx, query: str, search_type: str = "keyword") -> str:
    ...

@register_tool(name="read_record")
@function_tool
async def read_record(ctx, uuid: str) -> str:
    ...

# At startup: load YAML and enrich the registry
catalog = load_tool_catalog("tools.yaml")
catalog.enrich_registry(get_tool_registry())
```

### How the merge works

Registration happens in two phases:

1. **Import time** — `@register_tool(name="search_database")` registers the function with an empty `ToolDefinition` (name + function only)
2. **Startup** — `catalog.enrich_registry(registry)` patches each `ToolDefinition` with metadata from the YAML file

YAML values always win over decorator values. Empty YAML fields do not overwrite existing decorator values. This makes it backward compatible — decorators with full metadata still work without a YAML file.

### Backward compatibility

The decorator still accepts all metadata fields. These two approaches are equivalent:

```python
# Approach 1: YAML-driven (preferred for large projects)
@register_tool(name="search_database")

# Approach 2: Decorator-driven (still works, no YAML needed)
@register_tool(
    name="search_database",
    description="Search the database",
    category="search",
    parameters_description="query (str): Search text",
    returns_description="JSON with results",
)
```

If both are provided, YAML wins for non-empty fields.

### API reference

| Method | Returns | Description |
|--------|---------|-------------|
| `catalog.get(name)` | `ToolYamlEntry` | Resolved entry for a single tool |
| `catalog.list_tools()` | `list[str]` | All tool names in the catalog |
| `catalog.enrich_registry(registry)` | `None` | Patch registry ToolDefinitions with YAML metadata |

## MCP Server (optional)

Expose your registered tools as an [MCP](https://modelcontextprotocol.io/) server so any MCP-compatible client (Claude Desktop, Claude Code, VS Code, Cursor, ChatGPT Desktop) can call them. Tool descriptions, parameter schemas, and annotations all come from `tools.yaml` — no duplication.

Requires the `mcp` extra:

```bash
pip install 'sinan-agentic-core[mcp]'
# or: pip install git+https://github.com/thomaspernet/sinan-agentic.git#egg=sinan-agentic-core[mcp]
```

### YAML configuration

Mark tools for MCP exposure in `tools.yaml`:

```yaml
# tools.yaml
tools:
  search_database:
    description: Search the database for records
    category: search
    parameters_description: "query (str): Search text"
    returns_description: "JSON with results"
    mcp:                          # NEW — MCP-specific metadata
      expose: true
      annotations:
        readOnlyHint: true

  create_record:
    description: Create a new record
    category: write
    mcp:
      expose: true
      annotations:
        readOnlyHint: false
        idempotentHint: false

  internal_tool:
    description: Internal reasoning tool
    category: reasoning
    # No mcp section → not exposed to MCP clients
```

Define which tools the MCP server exposes in `agents.yaml`:

```yaml
# agents.yaml
tool_groups:
  search_tools:
    - search_database
    - read_record

mcp_servers:
  my_server:
    description: "My knowledge base — search and manage records"
    tools:
      - group: search_tools       # reuses existing tool groups
      - list_categories
    write_tools:                  # separate list — opt-in via config
      - create_record
      - update_record
```

### Building the server

Implement a `MCPContextFactory` to provide runtime dependencies (database connections, auth, filters) for each tool call:

```python
from sinan_agentic_core.mcp import MCPContextFactory, build_mcp_server
from sinan_agentic_core import get_tool_registry, load_agent_catalog, load_tool_catalog

class MyContextFactory(MCPContextFactory):
    async def create_context(self):
        db = await connect_to_database()
        return MyAppContext(db_connector=db)

    async def cleanup(self, context):
        await context.db_connector.close()

# Load catalogs
agent_catalog = load_agent_catalog("agents.yaml", knowledge_dir="knowledge/")
tool_catalog = load_tool_catalog("tools.yaml")
tool_catalog.enrich_registry(get_tool_registry())

# Get MCP server config from agents.yaml
mcp_config = agent_catalog.get_mcp_server("my_server")

# Build the server
server = build_mcp_server(
    server_name="My App",
    tool_registry=get_tool_registry(),
    tool_catalog=tool_catalog,
    mcp_config=mcp_config,
    context_factory=MyContextFactory(),
    include_write_tools=False,  # read-only by default
)

# Run in stdio mode (for Claude Desktop / Claude Code)
server.run(transport="stdio")
```

### Transports

**stdio** — for local MCP clients (Claude Desktop, Claude Code). The server launches as a subprocess:

```json
{
  "mcpServers": {
    "my-app": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "my_app.mcp_server"]
    }
  }
}
```

**Streamable HTTP** — for remote clients or when sharing a process with an existing web app (e.g., FastAPI). Mount the MCP server as an ASGI app:

```python
from fastapi import FastAPI

app = FastAPI()
mcp_app = server.streamable_http_app()
app.mount("/mcp", mcp_app)
```

Clients connect via HTTP:

```json
{
  "mcpServers": {
    "my-app": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### How tool invocation works

The `MCPToolAdapter` bridges MCP calls to your registered `@function_tool` functions:

1. MCP client calls a tool (e.g., `search_database(query="test")`)
2. The adapter creates a context via your `MCPContextFactory`
3. It builds a synthetic `ToolContext` and calls `FunctionTool.on_invoke_tool()`
4. The tool function runs with a real database connection, just like when called by an agent
5. The result is returned to the MCP client
6. The context is cleaned up (connections closed)

Each tool call gets its own context — no shared state between calls.

### API reference

| Function / Class | Description |
|---|---|
| `build_mcp_server(...)` | Build a FastMCP server from registry + catalogs |
| `MCPContextFactory` | ABC — implement `create_context()` and `cleanup()` |
| `MCPServerBuilder` | Low-level builder class (use `build_mcp_server()` for convenience) |
| `MCPServerConfig` | Pydantic model for resolved MCP server definition |
| `MCPToolConfig` | Per-tool MCP config (expose flag + annotations) |
| `catalog.get_mcp_tools()` | List tool names with `mcp.expose: true` |
| `catalog.get_mcp_server(name)` | Get resolved `MCPServerConfig` from `agents.yaml` |

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
from sinan_agentic_core import load_agent_catalog

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
from sinan_agentic_core import AgentDefinition, register_agent

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
from sinan_agentic_core import BaseAgentRunner, ToolErrorRecovery

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
from sinan_agentic_core import BaseAgentRunner, TurnBudget, ToolErrorRecovery

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
from sinan_agentic_core import load_agent_catalog, BaseAgentRunner

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
from sinan_agentic_core import BaseAgentRunner, TurnBudget

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
from sinan_agentic_core import AgentSession, SQLiteSessionStore

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
from sinan_agentic_core import ToolOutput, ChatResponse

output = ToolOutput(success=True, data={"temp": 72}, metadata={"source": "api"})
response = ChatResponse(success=True, response="It's sunny.", session_id="u-123", tools_called=["get_weather"])

output.to_dict()    # {"success": True, "data": {...}, "metadata": {...}}
response.to_dict()  # {"success": True, "response": "...", ...}
```

## Project Structure

```text
sinan_agentic_core/
├── __init__.py              # Main exports
├── orchestrator.py          # Multi-agent orchestration
├── core/
│   ├── base_runner.py       # BaseAgentRunner
│   ├── errors.py            # structured_tool_error for agent-as-tool failures
│   ├── turn_budget.py       # TurnBudget + TurnBudgetHooks (soft turn management)
│   └── turn_budget_tool.py  # request_extension tool (agent self-approval)
├── instructions/
│   └── builder.py           # InstructionBuilder base class
├── mcp/                        # MCP server support (optional, requires sinan-agentic-core[mcp])
│   ├── __init__.py             # Public API: build_mcp_server, MCPContextFactory
│   ├── context_protocol.py     # MCPContextFactory ABC
│   ├── server_builder.py       # MCPServerBuilder + build_mcp_server()
│   ├── tool_adapter.py         # Wraps registered tools for MCP invocation
│   └── yaml_schema.py          # Pydantic models for MCP YAML config
├── registry/
│   ├── agent_catalog.py     # YAML-driven agent catalog + MCP server definitions
│   ├── agent_registry.py    # AgentDefinition + registry
│   ├── agent_factory.py     # create_agent_from_registry()
│   ├── tool_catalog.py      # YAML-driven tool catalog + MCP exposure flags
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

## Future Work

### Conversation-Driven Learning (Lessons)

Agents today are stateless across sessions. When an agent makes a mistake, the only fix is manually adding rules to instructions, which causes instruction bloat and degrades attention over time.

The goal: a post-conversation analysis pipeline that automatically extracts lessons from completed conversations and retrieves relevant ones at the start of future sessions.

**Concept:**
- An analysis agent reads completed conversation transcripts
- It detects mistakes, inefficiencies, and user corrections (explicit: "no, I meant X"; implicit: wasted tool calls, repeated searches)
- It extracts structured lessons (trigger context, mistake, correct behavior, confidence)
- Lessons are stored in a `LessonStore` (SQLite default, overridable)
- At conversation start, `InstructionBuilder` retrieves relevant lessons by context similarity and injects them as dynamic context

**Key difference from rules:** rules grow linearly in the prompt; lessons scale in a database and only the 2-3 relevant ones are injected per conversation.

**Open problems:**
- Signal detection: how to reliably identify that a conversation went badly (especially when mistakes are subtle)
- Generalization: turning one specific mistake into a reusable lesson without overfitting
- Retrieval: selecting the right lessons from hundreds, with minimal context at conversation start
- Quality: preventing bad lessons from degrading future performance (confidence scoring, decay, contradiction detection)

**Research references:** ExpeL (cross-task insight extraction), Memento (case-based reasoning), A-Mem (self-organizing memory), Reflexion (verbal self-reflection), DSPy (automatic prompt optimization). See `documentation/04-brainstorms/10-agent-learning-from-conversations.md` in Digital Brain for the full analysis.

### Dynamic System Prompts

The OpenAI SDK re-sends the full system prompt every turn. For long instructions, this wastes tokens and may cause re-deliberation (agent re-reads its full playbook and reconsiders strategy mid-execution).

**Possible approaches:**
- Turn-aware instructions: full prompt on turn 1, condensed on turn 2+ (knowledge already in message history)
- LLM-rewritten instructions: dynamically condense instructions based on conversation state
- ITR-style retrieval: per-turn RAG over instruction fragments and tool subsets (95% token reduction in research, arxiv 2602.17046)

**Status:** Parked. Current instructions work and there's no evidence of real problems. Worth revisiting when instruction size or conversation length becomes a measurable bottleneck.

## License

MIT License
