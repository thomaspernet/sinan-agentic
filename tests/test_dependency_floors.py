"""Tests for the runtime dependency floors declared in ``pyproject.toml``.

This package is distributed as a library, so the declared floor — not the
lockfile — decides which dependency versions a consumer resolves. A floor that
drifts below a fix version silently reintroduces the bug for every consumer.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

# openai-agents 0.19.4 defers the non-streamed session save until the output
# guardrails have run (openai/openai-agents-python#4184), and 0.19.3 established
# the same blocked-output retention policy on the streamed path (#4148). Before
# them an answer an output guardrail rejected was still written to the session,
# so the caller never saw it but the model read it back as history on the next
# turn. ``BaseAgentRunner`` wires ``output_guardrails`` onto every agent it
# builds and passes a session into every run path, so both paths are live here.
#
# 0.19.3 carries a second, unrelated fix this package depends on: before it,
# ``ToolOutputTrimmer`` walked the name-keyed maps of a replayed tool schema
# (``$defs``, ``definitions``, ``patternProperties``, ``dependentSchemas``) as if
# their keys were schema keywords, so a definition or parameter named
# ``description``/``title``/``examples`` was deleted while the ``$ref`` pointing
# at it survived (#4110; #4036 fixed the same bug for ``properties`` in 0.19.2).
# Pydantic keys ``$defs`` by class name, so the collision is reachable through the
# SDK's own schema generation. ``ToolOutputTrimConfig`` makes that filter
# declarable from ``agents.yaml``, and the corruption is silent — the model just
# stops being able to call the tool.
#
# 0.19.2 carries a third fix (#4071): before it, the run data attached to a
# guardrail tripwire held every completed result under ``Runner.run_streamed()``
# but an empty list under ``Runner.run()`` and ``Runner.run_sync()``, which
# discarded their accumulator when the tripwire raised. ``run_error_payload()``
# reports that set, and ``chat()``, ``chat_with_hooks()``, and
# ``chat_streamed()`` all funnel through it — so below this floor one handler
# would describe the same rejection differently depending on the entry point the
# caller picked.
#
# 0.19.2 carries a fourth fix (#4038): ``FunctionTool`` exposes the callable
# ``@function_tool`` decorated through a read-only ``__wrapped__`` descriptor,
# retained on the generated invoker so it survives ``copy``, ``deepcopy``, and
# the ``dataclasses.replace`` the registries hand records out through. The MCP
# tool adapter probes it to call a decorated tool directly, which is what lets
# an unguarded tool skip the argument round-trip ``on_invoke_tool`` needs. Below
# this floor the probe finds nothing and every MCP call falls back to the SDK
# invoker — slower, but not wrong, so nothing raises to signal the drop.
#
# This floor subsumes the earlier 0.18.1 one, which closes Chat Completions
# streams on early exit (#3689) — the case ``chat_streamed()`` and
# ``BaseAgentRunner._execute_streamed()`` hit whenever a caller breaks out of
# the event loop, and the one the Azure provider's ``api_mode="chat_completions"``
# default puts every Azure deployment on.
MIN_OPENAI_AGENTS = (0, 19, 4)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# ``tomllib`` is 3.11+ and this package supports 3.10, so read the requirement
# specifier out of the manifest text rather than parsing the whole document.
_FLOOR_PATTERN = '"{package}>=([^",\\s]+)"'


def _parse_version(raw: str) -> tuple[int, ...]:
    """Return the leading ``(major, minor, patch)`` numbers of a version string."""
    return tuple(int(part) for part in re.findall(r"\d+", raw)[:3])


def _declared_floor(package: str) -> tuple[int, ...]:
    """Return the ``>=`` floor declared for *package* in ``pyproject.toml``."""
    manifest = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(_FLOOR_PATTERN.format(package=re.escape(package)), manifest)
    assert match is not None, f"{package} declares no '>=' floor in pyproject.toml"
    return _parse_version(match.group(1))


def test_declared_openai_agents_floor_guarantees_guardrail_session_fixes() -> None:
    """The manifest floor must not resolve consumers onto an SDK that persists blocked output."""
    assert _declared_floor("openai-agents") >= MIN_OPENAI_AGENTS


def test_resolved_openai_agents_has_guardrail_session_fixes() -> None:
    """The SDK this suite runs against must carry the fixes, not just allow them."""
    assert _parse_version(version("openai-agents")) >= MIN_OPENAI_AGENTS
