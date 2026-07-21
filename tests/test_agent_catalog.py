"""Tests for AgentCatalog — tool groups, conditional tools, agent-level conditions, knowledge."""

import textwrap
from types import SimpleNamespace
from typing import Any

import pytest
from agents import RunContextWrapper
from pydantic import ValidationError

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.model_retry import RetryTrigger
from sinan_agentic_core.core.tool_error_recovery import ToolErrorRecovery
from sinan_agentic_core.core.turn_budget import TurnBudget
from sinan_agentic_core.registry.agent_catalog import (
    AgentCatalog,
    AgentYamlEntry,
    CapabilityRef,
    TurnBudgetConfig,
    _check_condition,
    _load_knowledge_dir,
    _parse_capabilities,
    _resolve_dot_path,
    _resolve_knowledge,
    _resolve_tools,
    load_agent_catalog,
)
from sinan_agentic_core.registry.capability_registry import (
    CapabilityNotFoundError,
    get_capability_registry,
    register_capability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> SimpleNamespace:
    """Build a nested SimpleNamespace from dot-path kwargs.

    Example: _make_config(**{"agents.web_search.enabled": True})
    """
    root = SimpleNamespace()
    for path, value in kwargs.items():
        parts = path.split(".")
        current = root
        for part in parts[:-1]:
            if not hasattr(current, part):
                setattr(current, part, SimpleNamespace())
            current = getattr(current, part)
        setattr(current, parts[-1], value)
    return root


# ---------------------------------------------------------------------------
# _resolve_dot_path
# ---------------------------------------------------------------------------


class TestResolveDotPath:
    def test_simple_path(self):
        cfg = _make_config(**{"agents.web_search.enabled": True})
        assert _resolve_dot_path(cfg, "agents.web_search.enabled") is True

    def test_missing_segment_returns_none(self):
        cfg = SimpleNamespace()
        assert _resolve_dot_path(cfg, "agents.web_search.enabled") is None

    def test_single_segment(self):
        cfg = SimpleNamespace(debug=True)
        assert _resolve_dot_path(cfg, "debug") is True

    def test_falsy_value(self):
        cfg = _make_config(**{"feature.enabled": False})
        assert _resolve_dot_path(cfg, "feature.enabled") is False


# ---------------------------------------------------------------------------
# _check_condition
# ---------------------------------------------------------------------------


class TestCheckCondition:
    def test_no_when_returns_true(self):
        assert _check_condition(None, None) is True
        assert _check_condition("", None) is True

    def test_no_config_returns_false(self):
        assert _check_condition("agents.web_search.enabled", None) is False

    def test_truthy_condition(self):
        cfg = _make_config(**{"agents.web_search.enabled": True})
        assert _check_condition("agents.web_search.enabled", cfg) is True

    def test_falsy_condition(self):
        cfg = _make_config(**{"agents.web_search.enabled": False})
        assert _check_condition("agents.web_search.enabled", cfg) is False

    def test_missing_path_is_false(self):
        cfg = SimpleNamespace()
        assert _check_condition("agents.web_search.enabled", cfg) is False


# ---------------------------------------------------------------------------
# _resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_plain_strings(self):
        assert _resolve_tools(["a", "b", "c"], {}, None) == ["a", "b", "c"]

    def test_group_expansion(self):
        groups = {"nav": ["discover", "search", "read"]}
        result = _resolve_tools([{"group": "nav"}, "think"], groups, None)
        assert result == ["discover", "search", "read", "think"]

    def test_unknown_group_raises(self):
        with pytest.raises(KeyError, match="Tool group 'missing'"):
            _resolve_tools([{"group": "missing"}], {}, None)

    def test_conditional_tool_included(self):
        cfg = _make_config(**{"feature.enabled": True})
        raw = [{"tool": "web_search", "when": "feature.enabled"}]
        assert _resolve_tools(raw, {}, cfg) == ["web_search"]

    def test_conditional_tool_excluded(self):
        cfg = _make_config(**{"feature.enabled": False})
        raw = [{"tool": "web_search", "when": "feature.enabled"}]
        assert _resolve_tools(raw, {}, cfg) == []

    def test_conditional_without_config_excluded(self):
        raw = [{"tool": "web_search", "when": "feature.enabled"}]
        assert _resolve_tools(raw, {}, None) == []

    def test_conditional_without_when_included(self):
        raw = [{"tool": "web_search"}]
        assert _resolve_tools(raw, {}, None) == ["web_search"]

    def test_mixed_entries(self):
        groups = {"reasoning": ["think", "plan"]}
        cfg = _make_config(**{"web.enabled": True, "beta.enabled": False})
        raw = [
            "base_tool",
            {"group": "reasoning"},
            {"tool": "web_search", "when": "web.enabled"},
            {"tool": "beta_tool", "when": "beta.enabled"},
        ]
        result = _resolve_tools(raw, groups, cfg)
        assert result == ["base_tool", "think", "plan", "web_search"]

    def test_empty_list(self):
        assert _resolve_tools([], {}, None) == []


# ---------------------------------------------------------------------------
# AgentCatalog
# ---------------------------------------------------------------------------


class TestAgentCatalog:
    def _make_catalog(self):
        return AgentCatalog(
            tool_groups={"nav": ["discover", "search"]},
            raw_agents={
                "chatbot": {
                    "model": "reasoning",
                    "description": "Main agent",
                    "tools": [
                        {"group": "nav"},
                        "think",
                        {"tool": "web_search", "when": "web.enabled"},
                    ],
                },
                "web_agent": {
                    "model": "fast",
                    "description": "Web search",
                    "tools": [],
                    "when": "web.enabled",
                },
                "always_on": {
                    "model": "default",
                    "description": "Always enabled",
                    "tools": ["tool_a"],
                },
            },
        )

    def test_get_resolves_groups_and_conditions(self):
        catalog = self._make_catalog()
        cfg = _make_config(**{"web.enabled": True})
        entry = catalog.get("chatbot", config=cfg)
        assert entry.model == "reasoning"
        assert entry.tools == ["discover", "search", "think", "web_search"]

    def test_get_without_config_skips_conditionals(self):
        catalog = self._make_catalog()
        entry = catalog.get("chatbot")
        assert entry.tools == ["discover", "search", "think"]

    def test_get_missing_agent_raises(self):
        catalog = self._make_catalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.get("nonexistent")

    def test_get_resolves_guardrails(self):
        catalog = AgentCatalog(
            tool_groups={},
            raw_agents={
                "guarded": {
                    "model": "reasoning",
                    "description": "Guarded agent",
                    "guardrails": ["block_pii", "block_destructive_cypher"],
                },
            },
        )
        entry = catalog.get("guarded")
        assert entry.guardrails == ["block_pii", "block_destructive_cypher"]

    def test_get_defaults_guardrails_to_empty(self):
        catalog = self._make_catalog()
        assert catalog.get("always_on").guardrails == []

    def test_is_enabled_true(self):
        catalog = self._make_catalog()
        cfg = _make_config(**{"web.enabled": True})
        assert catalog.is_enabled("web_agent", config=cfg) is True

    def test_is_enabled_false(self):
        catalog = self._make_catalog()
        cfg = _make_config(**{"web.enabled": False})
        assert catalog.is_enabled("web_agent", config=cfg) is False

    def test_is_enabled_no_when_always_true(self):
        catalog = self._make_catalog()
        assert catalog.is_enabled("always_on") is True

    def test_is_enabled_missing_agent_false(self):
        catalog = self._make_catalog()
        assert catalog.is_enabled("nonexistent") is False

    def test_is_enabled_no_config_false(self):
        catalog = self._make_catalog()
        assert catalog.is_enabled("web_agent") is False

    def test_list_agents(self):
        catalog = self._make_catalog()
        assert set(catalog.list_agents()) == {"chatbot", "web_agent", "always_on"}


# ---------------------------------------------------------------------------
# load_agent_catalog
# ---------------------------------------------------------------------------


class TestLoadAgentCatalog:
    def test_load_from_file(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            tool_groups:
              reasoning: [think, plan]

            agents:
              my_agent:
                model: fast
                description: Test agent
                tools:
                  - group: reasoning
                  - custom_tool
                  - tool: optional
                    when: feature.on
        """)
        path = tmp_path / "agents.yaml"
        path.write_text(yaml_content)

        catalog = load_agent_catalog(path)
        assert "my_agent" in catalog.list_agents()

        # Without config — conditional skipped
        entry = catalog.get("my_agent")
        assert entry.tools == ["think", "plan", "custom_tool"]

        # With config — conditional included
        cfg = _make_config(**{"feature.on": True})
        entry = catalog.get("my_agent", config=cfg)
        assert entry.tools == ["think", "plan", "custom_tool", "optional"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        catalog = load_agent_catalog(tmp_path / "nonexistent.yaml")
        assert catalog.list_agents() == []

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "agents.yaml"
        path.write_text("")
        catalog = load_agent_catalog(path)
        assert catalog.list_agents() == []

    def test_load_no_tool_groups(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              simple:
                model: default
                description: No groups
                tools: [a, b]
        """)
        path = tmp_path / "agents.yaml"
        path.write_text(yaml_content)

        catalog = load_agent_catalog(path)
        entry = catalog.get("simple")
        assert entry.tools == ["a", "b"]


# ---------------------------------------------------------------------------
# AgentYamlEntry
# ---------------------------------------------------------------------------


class TestAgentYamlEntry:
    def test_defaults(self):
        entry = AgentYamlEntry(model="fast", description="test")
        assert entry.tools == []
        assert entry.knowledge_text == ""

    def test_with_tools(self):
        entry = AgentYamlEntry(model="fast", description="test", tools=["a", "b"])
        assert entry.tools == ["a", "b"]

    def test_with_knowledge(self):
        entry = AgentYamlEntry(model="fast", description="test", knowledge_text="Domain knowledge.")
        assert entry.knowledge_text == "Domain knowledge."

    def test_no_turn_budget_by_default(self):
        entry = AgentYamlEntry(model="fast", description="test")
        assert entry.turn_budget is None

    def test_build_turn_budget_none_when_not_configured(self):
        entry = AgentYamlEntry(model="fast", description="test")
        assert entry.build_turn_budget() is None

    def test_build_turn_budget_creates_budget(self):
        entry = AgentYamlEntry(
            model="fast",
            description="test",
            max_turns=30,
            turn_budget=TurnBudgetConfig(
                default_turns=15,
                reminder_at=3,
                max_extensions=2,
                extension_size=5,
            ),
        )
        budget = entry.build_turn_budget()
        assert budget is not None
        assert budget.default_turns == 15
        assert budget.reminder_at == 3
        assert budget.max_extensions == 2
        assert budget.extension_size == 5
        assert budget.absolute_max == 30

    def test_build_turn_budget_defaults_absolute_max(self):
        entry = AgentYamlEntry(
            model="fast",
            description="test",
            turn_budget=TurnBudgetConfig(default_turns=8),
        )
        budget = entry.build_turn_budget()
        assert budget.absolute_max == 25  # fallback when max_turns is None


# ---------------------------------------------------------------------------
# _resolve_knowledge
# ---------------------------------------------------------------------------


class TestResolveKnowledge:
    def test_empty_scopes(self):
        assert _resolve_knowledge([], {"global": "content"}) == ""

    def test_single_scope(self):
        knowledge = {"global": "Global knowledge."}
        assert _resolve_knowledge(["global"], knowledge) == "Global knowledge."

    def test_multiple_scopes_concatenated(self):
        knowledge = {"global": "Global.", "chatbot": "Chatbot."}
        result = _resolve_knowledge(["global", "chatbot"], knowledge)
        assert result == "Global.\n\nChatbot."

    def test_missing_scope_skipped(self):
        knowledge = {"global": "Global."}
        result = _resolve_knowledge(["global", "missing"], knowledge)
        assert result == "Global."

    def test_all_scopes_missing(self):
        assert _resolve_knowledge(["missing"], {}) == ""

    def test_whitespace_stripped(self):
        knowledge = {"global": "  Global.  \n"}
        assert _resolve_knowledge(["global"], knowledge) == "Global."


# ---------------------------------------------------------------------------
# _load_knowledge_dir
# ---------------------------------------------------------------------------


class TestLoadKnowledgeDir:
    def test_loads_yaml_files(self, tmp_path):
        (tmp_path / "global.yaml").write_text("content: |\n  Global knowledge.\n")
        (tmp_path / "chatbot.yaml").write_text("content: |\n  Chatbot knowledge.\n")

        result = _load_knowledge_dir(tmp_path)
        assert "global" in result
        assert "chatbot" in result
        assert "Global knowledge." in result["global"]
        assert "Chatbot knowledge." in result["chatbot"]

    def test_ignores_non_yaml_files(self, tmp_path):
        (tmp_path / "global.yaml").write_text("content: |\n  Global.\n")
        (tmp_path / "notes.md").write_text("Not a knowledge file.")

        result = _load_knowledge_dir(tmp_path)
        assert "global" in result
        assert "notes" not in result

    def test_skips_empty_content(self, tmp_path):
        (tmp_path / "empty.yaml").write_text("content: ''")

        result = _load_knowledge_dir(tmp_path)
        assert "empty" not in result

    def test_nonexistent_dir(self, tmp_path):
        result = _load_knowledge_dir(tmp_path / "nonexistent")
        assert result == {}

    def test_empty_dir(self, tmp_path):
        result = _load_knowledge_dir(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# AgentCatalog with knowledge
# ---------------------------------------------------------------------------


class TestAgentCatalogKnowledge:
    def test_get_resolves_knowledge(self):
        catalog = AgentCatalog(
            tool_groups={},
            raw_agents={
                "chatbot": {
                    "model": "reasoning",
                    "description": "Main",
                    "tools": [],
                    "knowledge": ["global", "chatbot"],
                },
            },
            knowledge={"global": "Graph model.", "chatbot": "Tool workflows."},
        )
        entry = catalog.get("chatbot")
        assert entry.knowledge_text == "Graph model.\n\nTool workflows."

    def test_no_knowledge_key_returns_empty(self):
        catalog = AgentCatalog(
            tool_groups={},
            raw_agents={
                "simple": {"model": "fast", "description": "No knowledge", "tools": []},
            },
            knowledge={"global": "Something."},
        )
        entry = catalog.get("simple")
        assert entry.knowledge_text == ""

    def test_no_knowledge_loaded_returns_empty(self):
        catalog = AgentCatalog(
            tool_groups={},
            raw_agents={
                "chatbot": {
                    "model": "fast",
                    "description": "Test",
                    "tools": [],
                    "knowledge": ["global"],
                },
            },
        )
        entry = catalog.get("chatbot")
        assert entry.knowledge_text == ""


# ---------------------------------------------------------------------------
# load_agent_catalog with knowledge_dir
# ---------------------------------------------------------------------------


class TestLoadAgentCatalogWithKnowledge:
    def test_load_with_knowledge_dir(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              chatbot:
                model: reasoning
                description: Main agent
                tools: []
                knowledge: [global, chatbot]
              simple:
                model: fast
                description: No knowledge
                tools: []
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / "global.yaml").write_text("content: |\n  Graph model.\n")
        (kdir / "chatbot.yaml").write_text("content: |\n  Tool workflows.\n")

        catalog = load_agent_catalog(tmp_path / "agents.yaml", knowledge_dir=kdir)

        entry = catalog.get("chatbot")
        assert "Graph model." in entry.knowledge_text
        assert "Tool workflows." in entry.knowledge_text

        simple = catalog.get("simple")
        assert simple.knowledge_text == ""

    def test_relative_knowledge_dir(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              agent:
                model: fast
                description: Test
                tools: []
                knowledge: [global]
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / "global.yaml").write_text("content: |\n  Domain info.\n")

        catalog = load_agent_catalog(tmp_path / "agents.yaml", knowledge_dir="knowledge")
        entry = catalog.get("agent")
        assert "Domain info." in entry.knowledge_text

    def test_no_knowledge_dir_param(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              agent:
                model: fast
                description: Test
                tools: []
                knowledge: [global]
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)

        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("agent")
        assert entry.knowledge_text == ""


# ---------------------------------------------------------------------------
# TurnBudgetConfig + catalog integration
# ---------------------------------------------------------------------------


class TestCatalogTurnBudget:
    def test_load_turn_budget_from_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              chatbot:
                model: reasoning
                max_turns: 30
                description: Main agent
                tools: []
                turn_budget:
                  default_turns: 15
                  reminder_at: 3
                  max_extensions: 2
                  extension_size: 5
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("chatbot")

        assert entry.turn_budget is not None
        assert entry.turn_budget.default_turns == 15
        assert entry.turn_budget.reminder_at == 3
        assert entry.turn_budget.max_extensions == 2
        assert entry.turn_budget.extension_size == 5

        budget = entry.build_turn_budget()
        assert budget.absolute_max == 30
        assert budget.default_turns == 15

    def test_no_turn_budget_in_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              simple:
                model: fast
                max_turns: 10
                description: Simple agent
                tools: []
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("simple")

        assert entry.turn_budget is None
        assert entry.build_turn_budget() is None

    def test_turn_budget_defaults_in_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              agent:
                model: fast
                max_turns: 25
                description: Test
                tools: []
                turn_budget:
                  default_turns: 12
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("agent")

        assert entry.turn_budget.default_turns == 12
        assert entry.turn_budget.reminder_at == 2  # default
        assert entry.turn_budget.max_extensions == 3  # default
        assert entry.turn_budget.extension_size == 5  # default


# ---------------------------------------------------------------------------
# Capability YAML parsing
# ---------------------------------------------------------------------------


class _SpyCapability(Capability):
    """Custom capability used in tests for the explicit list form."""

    def __init__(self, label: str = "spy", level: int = 0) -> None:
        self.label = label
        self.level = level

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        return f"spy({self.label}/{self.level})"


@pytest.fixture
def spy_capability_registered():
    """Register a spy capability under a unique name for the test, then clean up."""
    name = "_test_spy_capability"

    @register_capability(name)
    def factory(config: dict[str, Any]) -> Capability:
        return _SpyCapability(**config)

    yield name
    get_capability_registry()._factories.pop(name, None)


class TestParseCapabilities:
    def test_empty_returns_empty(self) -> None:
        assert _parse_capabilities("a", []) == []
        assert _parse_capabilities("a", None) == []

    def test_string_form(self) -> None:
        refs = _parse_capabilities("a", ["turn_budget"])
        assert refs == [CapabilityRef(name="turn_budget", config={})]

    def test_dict_form_with_config(self) -> None:
        refs = _parse_capabilities(
            "a",
            [{"name": "turn_budget", "config": {"default_turns": 5}}],
        )
        assert refs == [CapabilityRef(name="turn_budget", config={"default_turns": 5})]

    def test_dict_form_without_config(self) -> None:
        refs = _parse_capabilities("a", [{"name": "error_recovery"}])
        assert refs == [CapabilityRef(name="error_recovery", config={})]

    def test_dict_form_with_null_config(self) -> None:
        # YAML may parse a missing value as None — must normalize to {}.
        refs = _parse_capabilities("a", [{"name": "turn_budget", "config": None}])
        assert refs == [CapabilityRef(name="turn_budget", config={})]

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(CapabilityNotFoundError, match="unknown capability"):
            _parse_capabilities("a", ["does_not_exist"])

    def test_missing_name_key_raises(self) -> None:
        with pytest.raises(ValueError, match="missing the required 'name'"):
            _parse_capabilities("a", [{"config": {}}])

    def test_invalid_item_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a string or mapping"):
            _parse_capabilities("a", [123])

    def test_non_list_raises(self) -> None:
        with pytest.raises(TypeError, match="expected a list"):
            _parse_capabilities("a", {"name": "turn_budget"})


class TestCatalogParsesCapabilitiesBlock:
    def test_string_form_in_yaml(self, tmp_path) -> None:
        yaml_content = textwrap.dedent("""\
            agents:
              chatbot:
                model: fast
                description: Test
                tools: []
                capabilities:
                  - turn_budget
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("chatbot")
        assert entry.capabilities == [CapabilityRef(name="turn_budget", config={})]

    def test_dict_form_in_yaml(self, tmp_path) -> None:
        yaml_content = textwrap.dedent("""\
            agents:
              chatbot:
                model: fast
                description: Test
                tools: []
                capabilities:
                  - name: turn_budget
                    config:
                      default_turns: 6
                      reminder_at: 1
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("chatbot")
        assert entry.capabilities == [
            CapabilityRef(
                name="turn_budget",
                config={"default_turns": 6, "reminder_at": 1},
            )
        ]

    def test_unknown_capability_name_raises_on_get(self, tmp_path) -> None:
        yaml_content = textwrap.dedent("""\
            agents:
              chatbot:
                model: fast
                description: Test
                tools: []
                capabilities:
                  - never_registered_capability
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        with pytest.raises(CapabilityNotFoundError, match="never_registered_capability"):
            catalog.get("chatbot")

    def test_no_capabilities_block_returns_empty(self, tmp_path) -> None:
        yaml_content = textwrap.dedent("""\
            agents:
              chatbot:
                model: fast
                description: Test
                tools: []
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("chatbot")
        assert entry.capabilities == []


class TestBuildCapabilities:
    def test_no_capabilities_no_shorthand(self) -> None:
        entry = AgentYamlEntry(model="fast", description="test", error_recovery=False)
        assert entry.build_capabilities() == []

    def test_shorthand_only_turn_budget(self) -> None:
        entry = AgentYamlEntry(
            model="fast",
            description="test",
            max_turns=20,
            turn_budget=TurnBudgetConfig(default_turns=10),
            error_recovery=False,
        )
        built = entry.build_capabilities()
        assert len(built) == 1
        assert isinstance(built[0], TurnBudget)
        assert built[0].default_turns == 10
        assert built[0].absolute_max == 20

    def test_shorthand_only_error_recovery(self) -> None:
        entry = AgentYamlEntry(model="fast", description="test", error_recovery=True)
        built = entry.build_capabilities()
        assert len(built) == 1
        assert isinstance(built[0], ToolErrorRecovery)

    def test_explicit_list_only(self, spy_capability_registered) -> None:
        entry = AgentYamlEntry(
            model="fast",
            description="test",
            error_recovery=False,
            capabilities=[
                CapabilityRef(
                    name=spy_capability_registered,
                    config={"label": "x", "level": 2},
                )
            ],
        )
        built = entry.build_capabilities()
        assert len(built) == 1
        assert isinstance(built[0], _SpyCapability)
        assert built[0].label == "x"
        assert built[0].level == 2

    def test_mixing_shorthand_and_explicit_list(self, spy_capability_registered) -> None:
        entry = AgentYamlEntry(
            model="fast",
            description="test",
            max_turns=15,
            turn_budget=TurnBudgetConfig(default_turns=8),
            error_recovery=True,
            capabilities=[
                CapabilityRef(
                    name=spy_capability_registered,
                    config={"label": "extra"},
                )
            ],
        )
        built = entry.build_capabilities()
        # Order: shorthand first (turn_budget, error_recovery), then explicit list.
        assert [type(c) for c in built] == [
            TurnBudget,
            ToolErrorRecovery,
            _SpyCapability,
        ]
        assert built[2].label == "extra"

    def test_explicit_list_built_in_name(self) -> None:
        entry = AgentYamlEntry(
            model="fast",
            description="test",
            error_recovery=False,
            capabilities=[CapabilityRef(name="turn_budget", config={"default_turns": 4})],
        )
        built = entry.build_capabilities()
        assert len(built) == 1
        assert isinstance(built[0], TurnBudget)
        assert built[0].default_turns == 4


class TestCatalogIntegrationWithCapabilities:
    def test_full_yaml_with_mixed_forms(self, tmp_path, spy_capability_registered) -> None:
        yaml_content = textwrap.dedent(f"""\
            agents:
              chatbot:
                model: fast
                description: Test
                tools: []
                max_turns: 30
                turn_budget:
                  default_turns: 12
                error_recovery: true
                capabilities:
                  - name: {spy_capability_registered}
                    config:
                      label: from_yaml
                      level: 9
        """)
        (tmp_path / "agents.yaml").write_text(yaml_content)
        catalog = load_agent_catalog(tmp_path / "agents.yaml")
        entry = catalog.get("chatbot")

        built = entry.build_capabilities()
        assert [type(c) for c in built] == [
            TurnBudget,
            ToolErrorRecovery,
            _SpyCapability,
        ]
        assert built[0].default_turns == 12
        assert built[2].label == "from_yaml"
        assert built[2].level == 9


# ---------------------------------------------------------------------------
# invalid_output_recovery key
# ---------------------------------------------------------------------------


class TestInvalidOutputRecovery:
    def _catalog(self, entry: dict) -> AgentCatalog:
        return AgentCatalog(tool_groups={}, raw_agents={"extractor": entry})

    def test_defaults_to_enabled(self) -> None:
        catalog = self._catalog({"model": "fast", "description": "Extracts"})
        assert catalog.get("extractor").invalid_output_recovery is True

    def test_can_be_disabled(self) -> None:
        catalog = self._catalog(
            {"model": "fast", "description": "Extracts", "invalid_output_recovery": False}
        )
        assert catalog.get("extractor").invalid_output_recovery is False

    def test_loads_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "agents.yaml"
        path.write_text(textwrap.dedent("""\
                agents:
                  extractor:
                    model: fast
                    description: Extracts
                    invalid_output_recovery: false
            """))
        assert load_agent_catalog(path).get("extractor").invalid_output_recovery is False


# ---------------------------------------------------------------------------
# model_retry key
# ---------------------------------------------------------------------------


class TestModelRetry:
    def _catalog(self, entry: dict) -> AgentCatalog:
        return AgentCatalog(tool_groups={}, raw_agents={"researcher": entry})

    def test_defaults_to_off(self) -> None:
        catalog = self._catalog({"model": "fast", "description": "Reads papers"})
        assert catalog.get("researcher").model_retry is None

    def test_parses_the_declared_policy(self) -> None:
        catalog = self._catalog(
            {
                "model": "fast",
                "description": "Reads papers",
                "model_retry": {"max_retries": 3, "retry_on": ["retry_after"]},
            }
        )
        retry = catalog.get("researcher").model_retry

        assert retry.max_retries == 3
        assert retry.retry_on == [RetryTrigger.RETRY_AFTER]

    def test_rejects_an_unknown_trigger(self) -> None:
        catalog = self._catalog(
            {
                "model": "fast",
                "description": "Reads papers",
                "model_retry": {"retry_on": ["sometimes"]},
            }
        )
        with pytest.raises(ValidationError):
            catalog.get("researcher")

    def test_loads_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "agents.yaml"
        path.write_text(textwrap.dedent("""\
                agents:
                  researcher:
                    model: fast
                    description: Reads papers
                    model_retry:
                      max_retries: 3
                      retry_on: [provider_suggested]
                      backoff:
                        initial_delay: 0.5
                        max_delay: 8.0
            """))
        retry = load_agent_catalog(path).get("researcher").model_retry

        assert retry.max_retries == 3
        assert retry.retry_on == [RetryTrigger.PROVIDER_SUGGESTED]
        assert retry.backoff.initial_delay == 0.5
        assert retry.backoff.max_delay == 8.0


# ---------------------------------------------------------------------------
# tool_output_trim key
# ---------------------------------------------------------------------------


class TestToolOutputTrim:
    def _catalog(self, entry: dict) -> AgentCatalog:
        return AgentCatalog(tool_groups={}, raw_agents={"researcher": entry})

    def test_defaults_to_off(self) -> None:
        catalog = self._catalog({"model": "fast", "description": "Reads papers"})
        assert catalog.get("researcher").tool_output_trim is None

    def test_an_empty_mapping_opts_in_with_sdk_defaults(self) -> None:
        catalog = self._catalog(
            {"model": "fast", "description": "Reads papers", "tool_output_trim": {}}
        )
        trim = catalog.get("researcher").tool_output_trim

        assert trim is not None
        assert trim.max_output_chars is None

    def test_parses_the_declared_policy(self) -> None:
        catalog = self._catalog(
            {
                "model": "fast",
                "description": "Reads papers",
                "tool_output_trim": {"max_output_chars": 4000, "trimmable_tools": ["web_search"]},
            }
        )
        trim = catalog.get("researcher").tool_output_trim

        assert trim.max_output_chars == 4000
        assert trim.trimmable_tools == ["web_search"]

    def test_rejects_an_out_of_range_policy_at_load_time(self) -> None:
        """The SDK would only raise on the first model call; the catalog raises now."""
        catalog = self._catalog(
            {
                "model": "fast",
                "description": "Reads papers",
                "tool_output_trim": {"recent_turns": 0},
            }
        )
        with pytest.raises(ValidationError):
            catalog.get("researcher")

    def test_loads_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "agents.yaml"
        path.write_text(textwrap.dedent("""\
                agents:
                  researcher:
                    model: fast
                    description: Reads papers
                    tool_output_trim:
                      recent_turns: 3
                      max_output_chars: 4000
                      preview_chars: 500
                      trimmable_tools: [web_search]
            """))
        trim = load_agent_catalog(path).get("researcher").tool_output_trim

        assert trim.recent_turns == 3
        assert trim.max_output_chars == 4000
        assert trim.preview_chars == 500
        assert trim.trimmable_tools == ["web_search"]
