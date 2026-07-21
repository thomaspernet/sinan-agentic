---
paths:
  - "**/*.py"
---

# Backend Rules

When this rule fires, run once at the top of the task to load the docs declared via `mandatory_for.rules: [backend]`:

```bash
devwatch --repo "$REPO" doc-read --rule backend --display
```

## Capability observability on every model-call branch in `BaseAgentRunner`

Every execution branch in `sinan_agentic_core/core/base_runner.py` that issues a model call and accepts a capability list must surface a capability hook on that branch:

- Branches that delegate to `Runner.run(...)` / `Runner.run_streamed(...)` must build the hook bundle via `self._build_hooks(capabilities)` and pass it as `hooks=` when non-empty — matching the `_execute_basic` and `_execute_streamed` shape.
- Branches that bypass the SDK and call the OpenAI client directly (e.g. the summarize-and-extract recovery branch inside `_execute_with_fallback`) must invoke dedicated `Capability` hooks (`on_fallback_start` / `on_fallback_end`, plus any future equivalents added for new bypass branches) for every registered capability — once before the call and once after the response, before any output parsing.

When adding a new execution branch, modifying an existing one, or reviewing changes that touch a model-call site, confirm the capability hook surface reaches that branch. Silent observability gaps do not raise at runtime.

**Why:** Two sibling regressions on the same class.

- Issue #29 — `_execute_with_fallback`'s normal-execution branch dropped the `hooks=` kwarg on its `Runner.run` call, silencing every `on_agent_start` / `on_tool_start` / `on_tool_end` callback on that path. Sibling paths `_execute_basic` and `_execute_streamed` had the wiring; the third drifted. The SDK accepts the missing kwarg and runs without callbacks, so the regression did not raise.
- Issue #30 — the recovery branch of `_execute_with_fallback` ran outside the SDK lifecycle entirely (a direct `AsyncOpenAI` chat-completions call). Audit-log, output-validator, and reliability-monitoring capabilities silently missed every rescued run — typically the runs that nearly failed and are operationally the most interesting. The fix added explicit `on_fallback_start` / `on_fallback_end` hooks invoked directly by the runner, since SDK lifecycle hooks cannot reach a branch that does not go through `Runner.run`.

**How to apply:** For SDK-delegating branches, use the `run_kwargs: dict[str, Any] = {...}` pattern from `_execute_basic` (`base_runner.py:324`): build `hooks = self._build_hooks(capabilities)`, then `if hooks: run_kwargs["hooks"] = hooks` before calling `Runner.run(**run_kwargs)`. For SDK-bypassing branches (direct chat-completions, future provider-specific escape hatches), declare a matched pair of hooks on the `Capability` protocol (default no-op) and loop `for cap in capabilities: cap.on_<event>(ctx_wrapper, ...)` immediately before and after the bypass call — see `base_runner.py:400` (`on_fallback_start`) and `base_runner.py:462` (`on_fallback_end`) for the established pattern. Do not fire tool-event hooks on a bypass branch when no tools are invoked there — that produces misleading telemetry.

## Reuse the configured default OpenAI client on every SDK-bypassing branch in `BaseAgentRunner`

Any execution branch in `sinan_agentic_core/core/base_runner.py` that issues a model call outside `Runner.run(...)` / `Runner.run_streamed(...)` (e.g. the summarize-and-extract recovery branch inside `_execute_with_fallback`) must obtain its OpenAI client from `agents.models._openai_shared.get_default_openai_client()` — the client that `configure_llm_provider(...)` installs via `set_default_openai_client(...)`. Construct a fresh `AsyncOpenAI(...)` only as a fallback when no default has been configured. Do not read provider keys, endpoints, or deployment names directly; do not call `AsyncOpenAI(api_key=get_default_openai_key())`.

**Why:** Issue #35 — the recovery branch built `AsyncOpenAI(api_key=get_default_openai_key())`, bypassing the configured default. Under `AzureOpenAIProviderConfig`, `configure_llm_provider` (`sinan_agentic_core/llm/factory.py:42`) installs an `AsyncAzureOpenAI` via `set_default_openai_client(...)` but never sets a default key, so the rescue call crashed at client construction with `OpenAIError: The api_key client option must be set...`. Under plain OpenAI it masked itself whenever `OPENAI_API_KEY` was exported in the environment, which is why CI stayed green and the regression only surfaced on the first Azure-hosted evaluation. The framework's contract is that `configure_llm_provider` is the single point that owns provider-specific client wiring; any model-call site that rebuilds its own client breaks that contract.

**How to apply:** Use the pattern at `base_runner.py:402-407`:

```python
from agents.models._openai_shared import get_default_openai_client

client = get_default_openai_client()
if client is None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
```

Apply this on every new SDK-bypassing branch — current and future — including recovery, repair, summarize-and-extract, and any provider-specific escape hatch added later. Tests that need to inject a stub client should patch `get_default_openai_client` (or `set_default_openai_client` upstream), not bypass the lookup.

## Run-level SDK kwarg wiring on every `Runner`-calling branch in `BaseAgentRunner`

Every branch in `sinan_agentic_core/core/base_runner.py` that calls `Runner.run(...)`, `Runner.run_streamed(...)`, or builds `agent.as_tool(...)` kwargs must route each per-run SDK setting through its own `_build_*` helper and set the corresponding kwarg when the helper returns non-`None` — matching all five branches wired for issue #41's `self._build_run_config(agent_def)` → `run_config=` (`_execute_basic` `base_runner.py:332`, `_execute_with_fallback` `:384`, `_execute_streamed` `:533`, the `as_tool` sub-agent branch `:780`, `run_agent()` `:699`), and issue #42's `self._build_error_handlers(agent_def)` → `error_handlers=`, wired identically on the four `Runner.run`/`Runner.run_streamed` branches (`_execute_basic` `:338`, `_execute_with_fallback` `:398`, `_execute_streamed` `:553`, `run_agent()` `:723`).

When adding a new execution branch, or a new `_build_*` helper for a new per-run SDK setting, confirm the setting reaches every branch that calls `Runner.run`/`Runner.run_streamed`/`as_tool` — not just the ones under active development. A branch marked deprecated or "kept for backward compatibility" is not exempt: it still calls the SDK and still needs the setting, or its outputs silently diverge from the other branches.

**Exception — genuine SDK gaps, not oversights:** a branch is not in violation if the underlying SDK call does not accept the kwarg for that call shape. Verify this against the installed SDK version — do not assume — then document the gap inline with a `NOTE:` comment at the call site naming the SDK method and version checked, so a future reviewer (or `/check-code-quality`) can tell a real limitation apart from a missed wiring site. Issue #42: `Agent.as_tool()` accepts `run_config` (wired at `:804-806`) but has no `error_handlers` parameter in `openai-agents==0.18.3`, so the `as_tool` branch is correctly left out of `_build_error_handlers` wiring, documented at `base_runner.py:808-811`. Re-check when the SDK version bumps.

**Why:** Issue #41 — `_build_run_config` (pre-approval for tool-input guardrails) was wired onto `_execute_basic`, `_execute_with_fallback`, `_execute_streamed`, and the `as_tool` branch, but `run_agent()` — kept only for backward compatibility — was left out on the first pass. `check-code-quality` failed twice citing the identical gap (`run_agent() ... without run_config wiring`) before a third pass wired it. Issue #42 is a second, independent instance of the same class: a new helper (`_build_error_handlers`) needed the identical per-branch wiring, and this time landed correctly on the first pass — but its one skipped branch (`as_tool`) needed to be told apart from a repeat of the #41 gap. Without a documented exception clause, a future contributor or reviewer could either wrongly flag `as_tool` as missing wiring, or wrongly assume any branch is exempt whenever a plausible-sounding reason is asserted. This is now the fourth distinct per-branch concern to require this discipline — see the capability-hook and client-reuse entries above in this file — because the class has five near-duplicate `Runner`-calling branches and no shared mechanism that enforces a new per-run setting reaches all of them.

**How to apply:** Use the pattern at `base_runner.py:332-334` (`_execute_basic`, `_build_run_config`) or `:338-344` (`_build_error_handlers`): `<setting> = self._build_<setting>(agent_def); if <setting> is not None: run_kwargs["<kwarg>"] = <setting>`, called with `agent_def` already resolved via `self._get_agent_definition(agent_name)`, and placed before the `Runner.run(**run_kwargs)` / `Runner.run_streamed(**run_kwargs)` / `tool_agent.as_tool(**as_tool_kwargs)` call. Before shipping any new `_build_*` helper or per-run SDK setting, grep `base_runner.py` for `Runner.run(`, `Runner.run_streamed(`, and `as_tool_kwargs` — every hit must either carry the new kwarg or a `NOTE:` comment citing the specific SDK method/version that lacks the parameter.
