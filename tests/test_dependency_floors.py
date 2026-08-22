"""Tests for the dependency ranges declared in ``pyproject.toml``.

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

A floor is only half a range. Every declared requirement also carries a ``<``
ceiling, and the resolve has to land inside it: an unbounded requirement hands a
consumer the next major the moment it is published, which is how ``mcp`` 2.0.0 —
no ``mcp.server.fastmcp`` — reached the optional server builder and how ``openai``
3.0 — httpx2 in place of httpx — reached a CI run.

The manifest also has to name every distribution the code imports. A module that
arrives only as a transitive of something else is not declared support, it is a
coincidence of the current resolve: ``httpx`` was in the environment for as long
as ``openai`` happened to be built on it, and vanished the day it was not.
"""

from __future__ import annotations

import ast
import re
import sys
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
# It subsumes the 0.19.x floors this package reached for before it:
#
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
# 0.19.0 carries a fifth fix (#3931): before it, a failed non-streamed model
# request rewound session items on every retry, not only when a server
# conversation tracker owned the conversation. The rewind popped the turn's
# prepared input — history already committed to a plain local session included —
# so the retried request ran against a session missing earlier turns.
# ``rewind_session_items`` skips a session without ``pop_item``, but both session
# objects that reach a run here implement it: ``AgentSession`` and the
# ``_CollectingSessionWrapper`` the recovery branch runs behind. ``model_retry:``
# is what puts an agent on the retry path at all, and ``apply_declared_model_settings``
# overlays the policy onto every agent-building path, so any declaring agent is
# exposed. The streamed path never rewound session items and is unaffected.
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

# The two the SDK does not constrain, so their ranges answer to this package
# alone: ``pyyaml`` parses the agent and tool catalogs, and ``httpx2`` is the
# transport ``openai`` is built on — ``conftest.make_provider_status_error``
# builds the request/response pair ``APIStatusError`` reads back out of it.
NOT_SHARED_WITH_SDK = ("httpx2", "pyyaml")

# Every package whose resolved version the suite must run against, so a green
# run means "this works on what a consumer gets", not "on whatever the lock
# happened to freeze".
DECLARED_PACKAGES = (SDK_DISTRIBUTION, *SHARED_WITH_SDK, *NOT_SHARED_WITH_SDK)

# Modules that come from this repository rather than from a distribution.
FIRST_PARTY_MODULES = frozenset({"sinan_agentic_core", "tests"})

# The distribution each third-party module imported here is installed from.
# The link is spelled out rather than looked up: module and distribution names
# diverge often enough (``agents`` ships in ``openai-agents``, ``yaml`` in
# ``pyyaml``), and ``importlib.metadata.packages_distributions`` reads the link
# off ``top_level.txt``, which modern wheels no longer ship — on 3.10, the
# oldest interpreter supported here, it answers ``None`` for most of them.
MODULE_DISTRIBUTIONS = {
    "agents": "openai-agents",
    "httpx2": "httpx2",
    "mcp": "mcp",
    "openai": "openai",
    "pydantic": "pydantic",
    "pytest": "pytest",
    "yaml": "pyyaml",
}

REPO_ROOT = Path(__file__).resolve().parent.parent

# The trees whose imports the manifest has to cover: the shipped package and the
# suite that exercises it.
SOURCE_ROOTS = ("sinan_agentic_core", "tests")

PYPROJECT = REPO_ROOT / "pyproject.toml"

# ``tomllib`` is 3.11+ and this package supports 3.10, so read the requirement
# specifier out of the manifest text rather than parsing the whole document.
# Every requirement here opens with ``>=``, so anchoring on it keeps the capture
# off same-prefix names (``mcp-types``) while admitting a trailing ceiling.
_REQUIREMENT_PATTERN = '"{package}(>=[^"]+)"'

# A requirement's distribution name: a quoted name, optional extras, then the
# first comparison operator of its specifier.
_DECLARED_NAME_PATTERN = r'"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~]=|[<>])'


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


def _normalize(distribution: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _declared_distributions() -> set[str]:
    """Return every distribution ``pyproject.toml`` declares a specifier for."""
    manifest = PYPROJECT.read_text(encoding="utf-8")
    return {_normalize(name) for name in re.findall(_DECLARED_NAME_PATTERN, manifest)}


def _imported_top_level_modules() -> set[str]:
    """Return the third-party top-level modules this repository imports.

    Read out of the source rather than off ``sys.modules``: a module the
    manifest fails to declare is missing from the environment entirely, and by
    the time anything here could inspect the interpreter the suite has already
    aborted collection on the ``ImportError``.
    """
    modules: set[str] = set()
    for root in SOURCE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.partition(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules.add(node.module.partition(".")[0])
    return modules - set(sys.stdlib_module_names) - FIRST_PARTY_MODULES


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


@pytest.mark.parametrize("package", DECLARED_PACKAGES)
def test_resolved_version_stays_within_the_declared_ceiling(package: str) -> None:
    """Every declared range must be bounded above, and the resolve inside it.

    ``_declared_ceiling`` fails outright on a requirement with no ``<``, so this
    is also what stops a new dependency from being declared unbounded. The cost
    of an unbounded one is not hypothetical: ``mcp/server_builder.py`` imports
    ``FastMCP`` from ``mcp.server.fastmcp``, which exists in MCP SDK v1 only, and
    the suite builds provider errors on ``httpx2``, which arrived with ``openai``
    3. In both cases the declared range — not the lockfile — is what has to hold
    a consumer on the major the code targets.
    """
    assert _parse_version(version(package)) < _declared_ceiling(package)


def test_every_imported_module_names_its_distribution() -> None:
    """The map has to stay exactly the set of third-party modules imported here.

    An import with no entry is a dependency nobody decided to take; an entry with
    no import is a requirement the manifest keeps carrying for nothing. Either
    way the answer is to edit this map, which is what puts the declaration in
    front of a reviewer.
    """
    assert _imported_top_level_modules() == set(MODULE_DISTRIBUTIONS)


@pytest.mark.parametrize(("module", "distribution"), sorted(MODULE_DISTRIBUTIONS.items()))
def test_imported_module_is_declared(module: str, distribution: str) -> None:
    """Nothing imported here may reach the environment as someone else's transitive.

    ``tests/conftest.py`` imported ``httpx`` for as long as ``openai`` happened
    to be built on it. When ``openai`` 3 moved to ``httpx2`` the module stopped
    being installed and collection aborted before a single test ran — the import
    had never been declared, so nothing in the manifest had to change for it to
    disappear.
    """
    assert _normalize(distribution) in _declared_distributions(), (
        f"{module!r} is imported here but {distribution!r} is declared nowhere in "
        f"pyproject.toml, so it only arrives as a transitive of another requirement"
    )
