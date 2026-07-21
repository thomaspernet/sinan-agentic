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

## Run-level SDK kwarg wiring on every `Runner`-calling site in `sinan_agentic_core`

Every call site anywhere in `sinan_agentic_core/` that calls `Runner.run(...)`, `Runner.run_streamed(...)`, or builds `agent.as_tool(...)` kwargs must route each per-run SDK setting through its own `_build_*`/`build_*` helper and set the corresponding kwarg when the helper returns non-`None` — whether that call site is a branch inside `BaseAgentRunner` or a standalone function elsewhere in the package. Confirmed sites: issue #41's `self._build_run_config(agent_def)` → `run_config=` on all five `base_runner.py` branches (`_execute_basic` `:332`, `_execute_with_fallback` `:384`, `_execute_streamed` `:533`, the `as_tool` sub-agent branch `:780`, `run_agent()` `:699`); issue #42's `self._build_error_handlers(agent_def)` → `error_handlers=` on the four `Runner.run`/`Runner.run_streamed` branches (`_execute_basic` `:338`, `_execute_with_fallback` `:398`, `_execute_streamed` `:553`, `run_agent()` `:723`); and issue #59's `build_run_config(resolved)` → `run_config=`, wired on all three `services/chat.py` functions (`chat()` `:137-139`, `chat_with_hooks()` `:214-216`, `chat_streamed()` `:298-300`) — call sites that are not `BaseAgentRunner` branches at all.

When adding a new execution branch, a new `_build_*`/`build_*` helper for a new per-run SDK setting, or a new module anywhere in `sinan_agentic_core/` that calls `Runner.run`/`Runner.run_streamed`/`as_tool`, confirm the setting reaches every such call site in the package — not just the ones under active development, and not just the ones inside `base_runner.py`. A branch marked deprecated or "kept for backward compatibility" is not exempt: it still calls the SDK and still needs the setting, or its outputs silently diverge from the other branches.

**Exception — genuine SDK gaps, not oversights:** a call site is not in violation if the underlying SDK call does not accept the kwarg for that call shape. Verify this against the installed SDK version — do not assume — then document the gap inline with a `NOTE:` comment at the call site naming the SDK method and version checked, so a future reviewer (or `/check-code-quality`) can tell a real limitation apart from a missed wiring site. Issue #42: `Agent.as_tool()` accepts `run_config` (wired at `:804-806`) but has no `error_handlers` parameter in `openai-agents==0.18.3`, so the `as_tool` branch is correctly left out of `_build_error_handlers` wiring, documented at `base_runner.py:808-811`. Re-check when the SDK version bumps.

**Why:** Issue #41 — `_build_run_config` (pre-approval for tool-input guardrails) was wired onto `_execute_basic`, `_execute_with_fallback`, `_execute_streamed`, and the `as_tool` branch, but `run_agent()` — kept only for backward compatibility — was left out on the first pass. `check-code-quality` failed twice citing the identical gap (`run_agent() ... without run_config wiring`) before a third pass wired it. Issue #42 is a second, independent instance of the same class: a new helper (`_build_error_handlers`) needed the identical per-branch wiring, and this time landed correctly on the first pass — but its one skipped branch (`as_tool`) needed to be told apart from a repeat of the #41 gap. Issue #59 is a third instance, one level up: the identical per-run setting (tool-input pre-approval) needed the identical wiring discipline on `services/chat.py`'s three functions, which are not `BaseAgentRunner` branches — a call site family this rule's wording did not previously name. It landed correctly on the first pass by reusing the extracted `build_run_config()` helper, but a rule worded exclusively around `base_runner.py` risks a future contributor concluding that a new `Runner`-calling module elsewhere in the package is exempt, since it is not a "branch in `BaseAgentRunner`". This is now the fifth distinct per-call-site concern to require this discipline — see the capability-hook and client-reuse entries above in this file — because the class spans multiple near-duplicate `Runner`-calling sites, inside and outside `BaseAgentRunner`, with no shared mechanism that enforces a new per-run setting reaches all of them.

**How to apply:** Use the pattern at `base_runner.py:332-334` (`_execute_basic`, `_build_run_config`), `:338-344` (`_build_error_handlers`), or `services/chat.py:137-139` (`chat()`, `build_run_config`): `<setting> = self._build_<setting>(agent_def)` (or `build_<setting>(agent)` outside `BaseAgentRunner`); `if <setting> is not None: run_kwargs["<kwarg>"] = <setting>`, placed before the `Runner.run(**run_kwargs)` / `Runner.run_streamed(**run_kwargs)` / `tool_agent.as_tool(**as_tool_kwargs)` call. Before shipping any new `_build_*`/`build_*` helper or per-run SDK setting, grep the whole package — `grep -rn 'Runner\.run(\|Runner\.run_streamed(\|as_tool_kwargs' sinan_agentic_core/` — not just `base_runner.py`. Every hit must either carry the new kwarg or a `NOTE:` comment citing the specific SDK method/version that lacks the parameter.

## Verify claims about a wrapped SDK object's behavior before asserting them in a validator or doc

Any `Field` constraint, `model_validator`, error message, docstring, or README paragraph that asserts what a wrapped `agents.*` object does or does not do with a given input must be checked against the installed SDK's actual behavior — read its source or drive the real object through a representative input — before being written. Do not derive the claim from how the object's name or parameters suggest it "should" behave.

**Why:** Issue #46 — `ToolOutputTrimConfig` (`sinan_agentic_core/core/tool_output_trim.py`) added a cross-field validator rejecting any policy where `preview_chars >= max_output_chars`, on the premise that such a preview "never shortens an output." The premise was never checked against `agents.extensions.ToolOutputTrimmer`: the SDK compares the assembled replacement's length to the *original tool output's* length, not to `max_output_chars`, so `preview_chars == max_output_chars == 500` still collapsed a 100,000-character output to 566 characters. The same false premise was written four times — the validator's error message, its docstring, `README.md`, and a test docstring — before a reviewer measured the real filter against the installed SDK and found the rejected configuration was one of the most natural policies to express (`preview_chars: 500` with `max_output_chars: 500`, "keep the first 500 characters of anything bigger"). The wrapper ended up more restrictive than the library it wraps, on a claim that did not hold, and the same unverified claim was carried into three more artifacts before anyone tested it.

**How to apply:** Before adding a validator, `Field` constraint, docstring, or doc paragraph in `sinan_agentic_core/core/` that says a wrapped SDK object "always," "never," or "would never" does something with valid input, drive the installed object through that input in a throwaway script or a test and read the actual result. This applies to every module that translates a YAML-declared config into an SDK class — `tool_output_trim.py`, `model_retry.py`, `tool_error_recovery.py`, and future siblings — whenever the config layer adds a constraint the SDK class itself does not enforce. This is the same discipline as the SDK-version-gap exception in the entry above, applied to behavior instead of parameter presence: verify against the installed package first, then write the claim once it is confirmed true.

## Read a Registry through its own public accessors, never its private dict

Code outside `sinan_agentic_core/registry/` must not read a `Registry` class's leading-underscore attribute (`ToolRegistry._tools`, `AgentRegistry._agents`, `GuardrailRegistry._guardrails`) directly. If an existing public accessor covers the need (`get`, `get_all_functions`, `get_tool_functions`, `list_names`, `list_all`), call it. If none fits, add the accessor to the registry class itself — in `sinan_agentic_core/registry/` — rather than reaching past it. Do not add an accessor with no caller; add exactly the one the call site needs.

**Why:** Issue #58 — `BaseAgentRunner.__init__` built `tool_map` by comprehending over `self.tool_registry._tools.items()` (`base_runner.py:75`), and `_build_tools()`/`_build_handoffs()` checked membership and fetched definitions via `self.agent_registry._agents` at three more sites (`:821`, prior duplicate lookup, and `:980`), instead of the registry's own `.get()`. All four were the last sites in a repo-wide sweep tracked from epic #39 through #41: after the fix, `ToolRegistry` gained `get_all_functions()` (mirroring the pre-existing `GuardrailRegistry.get_all_functions()`), and both call sites in `base_runner.py` switched to `AgentRegistry.get(name)`. A grep across `sinan_agentic_core/` and `examples/` after the fix found zero remaining `_tools`/`_agents`/`_guardrails` reads outside the registry modules that own them — confirming the class of mistake, not a one-off.

**How to apply:** Before writing new code in `sinan_agentic_core/core/`, `sinan_agentic_core/registry/agent_factory.py`, or `examples/` that inspects a `ToolRegistry`, `AgentRegistry`, or `GuardrailRegistry` instance, check for an existing public accessor first — see `ToolRegistry.get_all_functions()` (`registry/tool_registry.py:60-62`) and `AgentRegistry.get()` (`registry/agent_registry.py:65-67`) for the pattern each new accessor should follow. This rule applies only to this codebase's own `Registry` classes — it does not cover third-party objects (e.g. the MCP SDK's `server._tool_manager._tools`, tracked separately under the SDK-internals sweep). Before shipping, `grep -rn '\._tools\b\|\._agents\b\|\._guardrails\b' sinan_agentic_core/ examples/` and confirm every hit is inside `sinan_agentic_core/registry/`.
