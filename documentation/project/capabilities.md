# Capabilities

`Capability` is the extension point for cross-cutting agent behavior - turn
budgets, tool-error recovery, audit logging, memory, compaction, anything
that needs to react to lifecycle events or steer the next model call.

Before this primitive existed, every new behavior meant editing
`base_runner.py` and adding another bespoke `RunHooks` adapter. Now adding a
behavior is "write a `Capability` subclass and attach it to an
`AgentDefinition`" - no runner edits required.

The shape mirrors the OpenAI Agents SDK `sandbox.Capability` (clone, bind,
instructions) so future interop is cheap, but the class has no dependency on
sandbox internals.

## Protocol overview

A capability is a stateful object with two roles:

1. **React to lifecycle events** via `on_*` hook methods. Default
   implementations are no-ops, so subclasses override only what they need.
2. **Steer the next model call** via `instructions(ctx)`, which returns a
   fragment appended to the model input before each LLM call. Return `None`
   to contribute nothing this turn.

The full surface lives in `sinan_agentic_core/core/capabilities/base.py`:

| Method | When it fires |
|---|---|
| `instructions(ctx)` | Before each LLM call - return a steering fragment or `None` |
| `on_agent_start(ctx, agent)` | Run begins |
| `on_agent_end(ctx, agent, output)` | Run ends with a final output |
| `on_handoff(ctx, from_agent, to_agent)` | Control hands off between agents |
| `on_tool_start(ctx, tool, args)` | Tool is about to execute |
| `on_tool_end(ctx, tool, result)` | Tool finished |
| `on_llm_start(ctx, agent, system_prompt, input_items)` | Model call begins |
| `on_llm_end(ctx, agent, response)` | Model call returns |
| `on_fallback_start(ctx, prompt, collected_items)` | Summarize-and-extract recovery call begins |
| `on_fallback_end(ctx, response, usage)` | Recovery call returns |
| `to_snapshot()` | Return a JSON-serializable snapshot of mutable state, or `None` to opt out of persistence |
| `from_snapshot(data)` | Rehydrate mutable state from a snapshot; tolerant of missing keys |
| `tools()` | Extra tools the capability exposes to the agent |
| `reset()` | Called at the start of every `execute()` - clear mutable state |
| `clone()` | Per-run copy; default is `copy.deepcopy(self)` |

A capability may also assign `self.on_event = callback` to emit custom
streaming events. The runtime sets this on cloned capabilities when streaming.

## Lifecycle

```text
AgentDefinition(capabilities=[A, B])
   |
   v
runner.execute(...)
   |
   |-- _build_run_capabilities():  for each declared cap -> cap.clone()
   |                               then cap.reset() on every effective cap
   |
   |-- _build_run_config(caps):     installs CapabilitySteering in the run
   |                                config's one call_model_input_filter slot
   |
   |-- _CompositeHooks(caps):       routes SDK RunHooks events to every
   |                                capability's matching on_* method
   |
   v
Runner.run(...)
   |
   |-- on_agent_start  -> A.on_agent_start, B.on_agent_start
   |-- (before each model call) CapabilitySteering appends A.instructions +
   |   B.instructions to the input as one trailing item
   |-- on_llm_start    -> A.on_llm_start, B.on_llm_start
   |-- on_tool_start / on_tool_end -> ...
   |-- on_llm_end
   |-- on_agent_end
   v
final_output (caps still hold post-run state for inspection)
```

The cloning rule is the key isolation guarantee: declarative capabilities on
`AgentDefinition` are templates, the runner clones them per call, so two
concurrent runs never share `turns_used` or error history.

## Writing a custom capability

Override only the hooks you need. Everything else is a no-op.

```python
from agents import RunContextWrapper, Tool
from sinan_agentic_core import Capability


class ToolCallLogger(Capability):
    """Print every tool call - args in, result out.

    Useful as a debugging aid during development. Drop it in via
    AgentDefinition.capabilities and watch the trace as the agent runs.
    """

    def __init__(self, prefix: str = "[tool]") -> None:
        self.prefix = prefix
        self.calls: list[tuple[str, str, str]] = []

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
        self.calls.append((tool.name, "", snippet))

    def reset(self) -> None:
        self.calls.clear()
```

Wire it onto an agent:

```python
from sinan_agentic_core import AgentDefinition, register_agent

register_agent(AgentDefinition(
    name="weather_assistant",
    description="Answers weather questions",
    instructions="You help users get weather information.",
    tools=["get_weather"],
    capabilities=[ToolCallLogger()],
))
```

The runner clones `ToolCallLogger()` on every `execute()` call, so each run
gets its own `calls` list. Inspect the original (template) for aggregate
state, or read `runner.last_*` if you wired the capability through the kwarg
path (see below).

### Adding instructions

A capability that steers the model overrides `instructions`:

```python
class TimeAwareness(Capability):
    """Tell the agent the current wall-clock time each turn."""

    def instructions(self, ctx):
        from datetime import datetime
        return f"Current time: {datetime.now().isoformat()}"
```

The runtime joins fragments from every attached capability with double
newlines and appends them to the model input as one trailing item before each
LLM call. Order matches `AgentDefinition.capabilities`.

The fragments deliberately do not go into the system prompt. A provider
serializes the system prompt first and caches only a byte-identical leading
span, so a fragment that changes between two calls of one run - a budget
counting down, a recovery tracker gaining an entry - would push the whole
conversation behind it out of the cacheable prefix. At the tail the prefix
stays stable, and the steering lands where a model follows it most reliably.
The item is ephemeral: it is built per call and never written to the session.

The same filter warns when the resolved instructions differ between two model
calls of one run, which catches drift a consumer's own instruction callable or
another filter introduces.

### Exposing tools

A capability can hand the agent extra tools by overriding `tools()`. This is
how `TurnBudget` exposes `request_extension_tool` for self-extension - the
runner picks them up automatically in `create_agent()`.

### Streaming events

Set `self.on_event` to push a structured event into the stream:

```python
def on_tool_end(self, ctx, tool, result):
    if self.on_event:
        self.on_event({"event": "tool_logged", "data": {"tool": tool.name}})
```

The runtime wires `on_event` through to clones automatically. Use this for
UI updates, progress indicators, or any custom telemetry.

## Wiring capabilities

There are two paths into the runner. Pick whichever matches the
capability's lifetime.

### Declarative (preferred)

Attach to `AgentDefinition.capabilities`. The capability is part of the
agent's identity; the runner clones it per run for isolation.

```python
register_agent(AgentDefinition(
    name="research_agent",
    description="Deep research",
    instructions="...",
    tools=["search", "read"],
    capabilities=[
        TurnBudget(default_turns=10),
        ToolErrorRecovery(),
        ToolCallLogger(),
    ],
))
```

### Runtime kwargs

For inspection-after-run flows (you want to read `budget.turns_used` after
`execute()` returns), pass the capability through `execute()`:

```python
budget = TurnBudget(default_turns=10)
recovery = ToolErrorRecovery(tool_registry=runner.tool_registry)

await runner.execute(
    agent_name="research_agent",
    context=ctx,
    session=session,
    input_text="Analyze these papers",
    turn_budget=budget,
    error_recovery=recovery,
)

print(budget.turns_used)
print(recovery.has_errors)
```

Runtime kwargs are used in place (not cloned), so post-run state is yours
to inspect. Declarative capabilities are cloned, so the template stays
clean across runs.

The runner merges both sources into a single effective list, calls
`reset()` on each one, then composes them into a single `_CompositeHooks`
for lifecycle events and a single `CapabilitySteering` for fragment delivery.

## Built-in capabilities

| Capability | What it does |
|---|---|
| `TurnBudget` | Soft turn budget with self-extension. Counts turns via `on_llm_start`, reports budget status via `instructions`, exposes `request_extension` via `tools()`. |
| `ToolErrorRecovery` | Tracks tool errors, detects repeated identical calls, reports progressive recovery guidance via `instructions`. |

Both are first-class `Capability` subclasses. The README sections on
[Turn Budget](../../README.md#turn-budget-soft-turn-management) and
[Tool Error Recovery](../../README.md#tool-error-recovery) cover usage.

## Migration notes

**No required changes for direct-Python users.** The `execute()` kwargs
`turn_budget=` and `error_recovery=` continue to work exactly as before.
Internally they are now treated as runtime capabilities, but the public
surface is unchanged - your code keeps running.

**Direct-Python users (recommended path).** New behaviors should land on
`AgentDefinition.capabilities` so the agent definition is self-contained:

```python
# Before: behaviors threaded through execute() kwargs only
await runner.execute(
    agent_name="agent",
    context=ctx,
    session=session,
    input_text="...",
    turn_budget=TurnBudget(default_turns=10),
    error_recovery=ToolErrorRecovery(),
)

# After (equivalent, declarative): wire on the agent definition
register_agent(AgentDefinition(
    name="agent",
    description="...",
    instructions="...",
    tools=[...],
    capabilities=[
        TurnBudget(default_turns=10),
        ToolErrorRecovery(),
    ],
))

await runner.execute(
    agent_name="agent",
    context=ctx,
    session=session,
    input_text="...",
)
```

Both forms produce the same effective capability list. Runtime kwargs still
win when you need to inspect post-run state.

**YAML users.** The current `agents.yaml` schema continues to use the
`turn_budget:` block and the `error_recovery: bool` flag - those keep
working as a convenience for the two built-in capabilities. A general
`capabilities:` YAML block (for arbitrary user-defined capabilities) is the
next planned increment; until it lands, attach custom capabilities in
Python after loading the catalog:

```python
catalog = load_agent_catalog("agents.yaml")
cfg = catalog.get("research_agent")

register_agent(AgentDefinition(
    name="research_agent",
    model=cfg.model,
    description=cfg.description,
    tools=cfg.tools,
    instructions=build_instructions,
    capabilities=[
        cfg.build_turn_budget(),
        cfg.build_error_recovery(),
        ToolCallLogger(),
    ],
))
```

## Reference

- Protocol: `sinan_agentic_core/core/capabilities/base.py`
- Fragment delivery: `sinan_agentic_core/core/capabilities/steering.py`, placed
  in the run config's filter slot by
  `sinan_agentic_core/core/run_config.py`'s `build_model_input_filter`
- Composition: `BaseAgentRunner._build_run_capabilities` and
  `_CompositeHooks` in `sinan_agentic_core/core/base_runner.py`
- Built-ins: `sinan_agentic_core/core/turn_budget.py`,
  `sinan_agentic_core/core/tool_error_recovery.py`
- Runnable example: `examples/custom_capability.py`
