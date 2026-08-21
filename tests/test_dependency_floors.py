"""Tests for the runtime dependency floors declared in ``pyproject.toml``.

This package is distributed as a library, so the declared floor — not the
lockfile — decides which dependency versions a consumer resolves. A floor that
drifts below a fix version silently reintroduces the bug for every consumer.

Two kinds of floor live in the manifest and each is checked differently:

- ``openai-agents`` is this package's own decision — it is pinned to the
  release that carries the SDK feature the framework builds on, so the floor
  is a constant here with the reason next to it.
- ``openai``, ``pydantic``, and ``mcp`` are shared with the SDK, which
  constrains them itself. Resolving ``openai-agents`` already forces at least
  the SDK's floor on every consumer, so those are checked against what the
  installed SDK declares rather than against a number copied into this file
  that would go stale on the next bump.

A floor is only half a range. Where this package supports less than the SDK
admits it also declares a ``<`` ceiling, and the resolve has to land inside it:
an unbounded requirement hands a consumer the next major the moment it is
published, which is how ``mcp`` 2.0.0 — no ``mcp.server.fastmcp`` — reached the
optional server builder.
"""

from __future__ import annotations

import re
from importlib.metadata import requires, version
from pathlib import Path

import pytest

SDK_DISTRIBUTION = "openai-agents"

# openai-agents 0.21.1 added ``ModelSettings.timeout``
# (openai/openai-agents-python#4428), the per-attempt bound the declarative
# ``model_timeout:`` key translates into. Below this floor the field does not
# exist, so ``apply_declared_model_settings`` raises while building the agent
# and every declaring agent fails outright — the one dependency here whose
# absence is a hard failure rather than a silently reintroduced bug.
#
# This floor subsumes the earlier 0.18.1 one, which closes Chat Completions
# streams on early exit (#3689) — the case ``chat_streamed()`` and
# ``BaseAgentRunner._execute_streamed()`` hit whenever a caller breaks out of
# the event loop, and the one the Azure provider's ``api_mode="chat_completions"``
# default puts every Azure deployment on.
MIN_OPENAI_AGENTS = (0, 21, 1)

# The dependencies this manifest declares directly that ``openai-agents`` also
# constrains. ``openai`` backs the ``AsyncOpenAI`` / ``AsyncAzureOpenAI``
# clients ``llm/factory.py`` builds and the typed ``APIStatusError``
# ``core/run_errors.py`` classifies; ``pydantic`` types every model crossing a
# layer boundary; ``mcp`` backs the optional server builder.
SHARED_WITH_SDK = ("mcp", "openai", "pydantic")

# Every package whose resolved version the suite must run against, so a green
# run means "this works on what a consumer gets", not "on whatever the lock
# happened to freeze".
DECLARED_PACKAGES = (SDK_DISTRIBUTION, *SHARED_WITH_SDK)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# ``tomllib`` is 3.11+ and this package supports 3.10, so read the requirement
# specifier out of the manifest text rather than parsing the whole document.
# Every requirement here opens with ``>=``, so anchoring on it keeps the capture
# off same-prefix names (``mcp-types``) while admitting a trailing ceiling.
_REQUIREMENT_PATTERN = '"{package}(>=[^"]+)"'


def _parse_version(raw: str) -> tuple[int, ...]:
    """Return the leading ``(major, minor, patch)`` numbers of a version string."""
    return tuple(int(part) for part in re.findall(r"\d+", raw)[:3])


def _declared_requirements(package: str) -> list[str]:
    """Return every version specifier declared for *package* in ``pyproject.toml``.

    A package may be declared more than once — ``mcp`` appears in both the
    ``mcp`` and ``dev`` extras — and the weakest declaration is the one that
    decides what a consumer can resolve, so all of them are read.
    """
    manifest = PYPROJECT.read_text(encoding="utf-8")
    declared = re.findall(_REQUIREMENT_PATTERN.format(package=re.escape(package)), manifest)
    assert declared, f"{package} declares no version specifier in pyproject.toml"
    return declared


def _declared_floor(package: str) -> tuple[int, ...]:
    """Return the lowest ``>=`` floor declared for *package* in ``pyproject.toml``."""
    floors = [re.search(r">=\s*([\d.]+)", spec) for spec in _declared_requirements(package)]
    assert all(floors), f"{package} declares no '>=' floor in pyproject.toml"
    return min(_parse_version(found.group(1)) for found in floors if found)


def _declared_ceiling(package: str) -> tuple[int, ...]:
    """Return the lowest ``<`` ceiling declared for *package* in ``pyproject.toml``."""
    ceilings = [re.search(r"<\s*([\d.]+)", spec) for spec in _declared_requirements(package)]
    assert all(ceilings), f"{package} declares no '<' ceiling in pyproject.toml"
    return min(_parse_version(found.group(1)) for found in ceilings if found)


def _sdk_floor(package: str) -> tuple[int, ...]:
    """Return the ``>=`` floor the installed ``openai-agents`` declares for *package*."""
    for requirement in requires(SDK_DISTRIBUTION) or ():
        # Strip the environment marker first: it carries its own ``>=``
        # comparisons (``python_version >= '3.10'``) that are not version floors.
        specifier, _, _marker = requirement.partition(";")
        name = re.match(r"[A-Za-z0-9._-]+", specifier)
        if name is None or name.group(0) != package:
            continue
        floor = re.search(r">=\s*([^,\s]+)", specifier)
        assert floor is not None, f"{SDK_DISTRIBUTION} declares no '>=' floor for {package}"
        return _parse_version(floor.group(1))
    raise AssertionError(f"{SDK_DISTRIBUTION} does not depend on {package}")


def test_declared_openai_agents_floor_guarantees_model_timeout_field() -> None:
    """The manifest floor must not resolve consumers onto an SDK without the field."""
    assert _declared_floor(SDK_DISTRIBUTION) >= MIN_OPENAI_AGENTS


@pytest.mark.parametrize("package", SHARED_WITH_SDK)
def test_declared_floor_is_not_below_the_sdk_floor(package: str) -> None:
    """A floor under the SDK's describes a resolve no consumer can actually get.

    ``openai-agents`` pins its own shared dependencies, so the version a
    consumer installs is at least the SDK's floor whatever this manifest says.
    Declaring less claims support for a combination that never installs — and
    that this suite therefore never runs.
    """
    assert _declared_floor(package) >= _sdk_floor(package)


@pytest.mark.parametrize("package", DECLARED_PACKAGES)
def test_resolved_version_meets_the_declared_floor(package: str) -> None:
    """The suite must run against versions a consumer can actually resolve."""
    assert _parse_version(version(package)) >= _declared_floor(package)


def test_resolved_mcp_stays_within_the_declared_ceiling() -> None:
    """The optional extra must resolve onto the MCP major the builder targets.

    ``mcp/server_builder.py`` imports ``FastMCP`` from ``mcp.server.fastmcp``,
    which exists in MCP SDK v1 only. Without a ceiling the extra resolves the
    newest major and the builder raises on import, so the declared range — not
    the lockfile — is what has to hold the consumer inside v1.
    """
    assert _parse_version(version("mcp")) < _declared_ceiling("mcp")
