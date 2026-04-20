# Briefing — Capability Pattern Epic for sinan-agentic-core

> Handoff document for the agent who will write the GitHub epic and child issues.
> Assumes zero prior context.

## Task

Write a **GitHub epic + sequenced child issues** for https://github.com/thomaspernet/sinan-agentic to introduce a **`Capability` abstraction** into the `sinan-agentic-core` package, inspired by OpenAI's Agents SDK 0.14 `sandbox.Capability` primitive.

Deliverable: issue titles, descriptions, acceptance criteria, and inter-issue dependencies — ready to be posted via `gh issue create --repo thomaspernet/sinan-agentic ...` or pasted into GitHub's UI. **No code changes in this task.**

## Background context

OpenAI shipped a major update to the Agents SDK on **2026-04-15** (blog: "The next evolution of the Agents SDK"). The headline is a sandbox-native harness, but the architecturally interesting primitive is `agents.sandbox.Capability`:

> Base class for sandbox capabilities; can mutate manifests, add instructions, expose tools. Key methods: `clone()`, `bind(session)`, `instructions(manifest)`.

This is the same pattern sinan-agentic has already built **twice by hand**, in an ad-hoc way:

- `sinan_agentic_core/core/turn_budget.py` — `TurnBudget` class, tracks turn count, injects guidance into agent instructions
- `sinan_agentic_core/core/tool_error_recovery.py` — `ToolErrorRecovery` class, tracks tool errors, injects recovery hints

Both are hardcoded into `_CompositeHooks` in `sinan_agentic_core/core/base_runner.py:1037-1073`. Each new behavior (skills, compaction, memory) would require editing `base_runner.py` again.

**The epic's goal:** extract a `Capability` protocol, refactor the two existing behaviors to implement it, and make capabilities pluggable via `AgentDefinition`.

## Current state of sinan-agentic (audit summary)

Key files the other agent must reference:

| File | Role |
|---|---|
| `sinan_agentic_core/registry/agent_registry.py:8-39` | `AgentDefinition` dataclass — needs new `capabilities` field |
| `sinan_agentic_core/core/base_runner.py:91-151` | `create_agent()` — where capabilities get wired into the agent |
| `sinan_agentic_core/core/base_runner.py:860-926` | `_apply_dynamic_instructions()` — where capability instruction fragments get injected |
| `sinan_agentic_core/core/base_runner.py:1037-1073` | `_CompositeHooks` — must iterate capabilities instead of hardcoded budget+recovery |
| `sinan_agentic_core/core/turn_budget.py` | Existing ad-hoc capability — 197 lines, has `reset()`, `build_instruction_section()` |
| `sinan_agentic_core/core/tool_error_recovery.py` | Existing ad-hoc capability — 300 lines, uses `RunHooks.on_tool_end()` |
| `sinan_agentic_core/registry/agent_catalog.py:54-89` | YAML loader — needs to parse a `capabilities:` block |
| `sinan_agentic_core/session/agent_session.py` | Session abstraction — natural place for capability state persistence (stretch goal) |
| `sinan_agentic_core/session/sqlite_store.py` | Persistence layer — target for snapshot/rehydrate (stretch goal) |

### Other known facts

- Package is `sinan-agentic-core` (recently renamed, see commit `ab0c4a3`)
- Registry has recent additions: `tool_rules` (commit `a81dd97`) and per-agent `effort` field (commit `1afa0b4`)
- Python, uses `uv` for package management, `pytest` for tests
- **No existing workspace/sandbox/filesystem-isolation concept** — the epic does NOT need to introduce one; that's deferred

## Proposed `Capability` protocol

Minimum interface (mirrors OpenAI's):

- `instructions(ctx) -> str | None` — contribute a fragment to the system prompt each turn
- `on_tool_start(ctx, tool, args)` / `on_tool_end(ctx, tool, result)` — lifecycle hooks
- `reset()` — called at start of each `execute()` call
- `clone()` — per-run copy (capabilities are often stateful and shouldn't leak across runs)
- Optional: `tools() -> list[Tool]` — expose tools to the agent
- Optional (stretch): `to_snapshot() / from_snapshot(data)` — for durable execution

## Proposed issue breakdown

The other agent should use these as a starting point but has license to split/merge where sensible.

### 1. Define `Capability` protocol and base class

- New module: `sinan_agentic_core/core/capabilities/base.py`
- Pure addition, no refactor. Includes unit tests of default behavior.
- Acceptance: protocol importable, type-checks against a dummy capability

### 2. Refactor `TurnBudget` to implement `Capability`

- No behavior change, just adopt the protocol
- Keep backward-compatible constructor + public methods
- All existing turn-budget tests must pass unchanged

### 3. Refactor `ToolErrorRecovery` to implement `Capability`

- Same as (2) for the error-recovery class

### 4. Add `capabilities` field to `AgentDefinition` and generalize `_CompositeHooks`

- Dataclass field: `capabilities: list[Capability] = field(default_factory=list)`
- `_CompositeHooks` iterates `agent_def.capabilities` instead of calling `TurnBudget`/`ToolErrorRecovery` directly
- `_apply_dynamic_instructions` merges all `capability.instructions(ctx)` fragments
- Acceptance: behavior identical to today when `TurnBudget` + `ToolErrorRecovery` are passed as capabilities

### 5. Wire capabilities into YAML registry

- `agent_catalog.py` parses `capabilities:` block
- Built-in shorthand: `turn_budget:` and `error_recovery:` YAML keys auto-materialize the capability
- Allow custom capabilities via a `CapabilityRegistry` (parallel to `ToolRegistry`)

### 6. Migration + documentation

- Migration notes for anyone using `TurnBudget`/`ToolErrorRecovery` directly (should be zero-change)
- Example: custom `Capability` that logs every tool call
- Update README "Extending sinan-agentic" section

### 7. (Stretch, separate issue) Capability snapshot/rehydrate

- `to_snapshot() / from_snapshot()` on `Capability`
- Persist alongside `SQLiteSessionStore` rows
- Out of scope if epic gets too large — defer to a follow-up epic

### Dependencies

`1 → 2, 3` (parallel) `→ 4 → 5 → 6`. Issue 7 depends on 4 but otherwise independent.

## Constraints for the other agent to respect

- **No backward-compat shims.** Thomas's global preferences say: "if something is unused, delete it." After the refactor, `TurnBudget`/`ToolErrorRecovery` keep their public APIs but shouldn't keep dead helper paths.
- **Conventional commits** (`fix:`, `feat:`, `refactor:`, `docs:`, `chore:`, `test:`) — each issue should imply one such commit.
- **Test before push.** Acceptance criteria must list concrete tests.
- **Small surface.** Don't introduce speculative features (no Manifest, no Workspace, no sandbox clients — those are explicitly deferred).
- Keep the `Capability` protocol shape **close to OpenAI's** so future interop is cheap — but don't depend on their SDK's internals.

## Expected output from the other agent

A single markdown document (or direct `gh issue create` commands) containing:

- **1 epic issue** with the vision, goals, non-goals, and links to child issues
- **6–7 child issues** with title, description, acceptance criteria, dependencies, and estimated complexity (S/M/L)
- **Labels** suggestion: `epic`, `capability-pattern`, `refactor`, `enhancement`

## References

1. **Target repository** — `thomaspernet/sinan-agentic`: https://github.com/thomaspernet/sinan-agentic
2. OpenAI blog post — **The next evolution of the Agents SDK** (2026-04-15): https://openai.com/index/the-next-evolution-of-the-agents-sdk/
3. OpenAI Agents SDK **sandbox module reference**: https://openai.github.io/openai-agents-python/ref/sandbox/
4. OpenAI Developer Community **announcement thread**: https://community.openai.com/t/the-next-evolution-of-the-agents-sdk/1379072
5. Hacker News **discussion** (item 47782022): https://news.ycombinator.com/item?id=47782022
6. **`openai-agents` Python package** (install: `pip install "openai-agents>=0.14.0"`):
   - PyPI: https://pypi.org/project/openai-agents/
   - GitHub source: https://github.com/openai/openai-agents-python
   - Docs root: https://openai.github.io/openai-agents-python/
7. Example code from the blog post uses: `agents.sandbox.{Manifest, SandboxAgent, SandboxRunConfig}`, `agents.sandbox.entries.LocalDir`, `agents.sandbox.sandboxes.UnixLocalSandboxClient` — reference for API surface shape
8. Sinan-agentic audit — performed in the originating conversation; key files listed in the table above
