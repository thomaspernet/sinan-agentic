"""Tests for the runtime dependency floors declared in ``pyproject.toml``.

This package is distributed as a library, so the declared floor — not the
lockfile — decides which dependency versions a consumer resolves. A floor that
drifts below a fix version silently reintroduces the bug for every consumer.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

# openai-agents 0.18.1 closes Chat Completions streams on early exit
# (openai/openai-agents-python#3689). ``chat_streamed()`` and
# ``BaseAgentRunner._execute_streamed()`` relay SDK stream events to a caller
# that may break out of the loop early — a cancel, a first-token-only read, or a
# timeout — which is exactly the case that leaked the HTTP connection before the
# fix. The Azure provider defaults to ``api_mode="chat_completions"``, putting
# the default Azure deployment on the affected code path.
MIN_OPENAI_AGENTS = (0, 18, 1)

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


def test_declared_openai_agents_floor_guarantees_stream_close_fix() -> None:
    """The manifest floor must not resolve consumers onto a leaking SDK."""
    assert _declared_floor("openai-agents") >= MIN_OPENAI_AGENTS


def test_resolved_openai_agents_has_stream_close_fix() -> None:
    """The SDK this suite runs against must carry the fix, not just allow it."""
    assert _parse_version(version("openai-agents")) >= MIN_OPENAI_AGENTS
