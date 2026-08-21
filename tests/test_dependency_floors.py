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
