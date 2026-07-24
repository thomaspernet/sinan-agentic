"""Tests for agent, tool, and guardrail registries."""

import copy
from typing import Any
from unittest.mock import Mock, patch

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    ToolGuardrailFunctionOutput,
    function_tool,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
)

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.model_retry import ModelRetryConfig
from sinan_agentic_core.core.turn_budget import TurnBudget
from sinan_agentic_core.registry.agent_registry import (
    AgentDefinition,
    AgentRegistry,
    resolve_agent_definition,
)
from sinan_agentic_core.registry.guardrail_registry import (
    GuardrailCategory,
    GuardrailDefinition,
    GuardrailRegistry,
    ResolvedGuardrails,
    attach_tool_input_guardrails,
)
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry
from tests.conftest import collection_field_names, edit_every_level

# -- AgentDefinition -----------------------------------------------------------


class TestAgentDefinition:
    def test_valid_creation(self):
        a = AgentDefinition(name="a1", description="desc", instructions="do stuff")
        assert a.name == "a1"
        assert a.model == "gpt-4o-mini"
        assert a.tools == []

    def test_missing_instructions_raises(self):
        with pytest.raises(ValueError, match="must have instructions"):
            AgentDefinition(name="bad", description="d")

    def test_callable_instructions(self):
        def fn(ctx, agent):
            return "dynamic"

        a = AgentDefinition(name="a2", description="d", instructions=fn)
        assert callable(a.instructions)

    def test_default_fields(self):
        a = AgentDefinition(name="a", description="d", instructions="i")
        assert a.guardrails == []
        assert a.handoffs == []
        assert a.hosted_tools == []
        assert a.output_dataclass is None
        assert a.requires_schema_injection is False

    def test_hosted_tools_default_empty(self):
        a = AgentDefinition(name="a", description="d", instructions="i")
        assert a.hosted_tools == []

    def test_hosted_tools_set(self):
        def factory():
            return "mock_tool"

        a = AgentDefinition(name="a", description="d", instructions="i", hosted_tools=[factory])
        assert len(a.hosted_tools) == 1
        assert a.hosted_tools[0]() == "mock_tool"


class TestAgentDefinitionOwnsItsListFields:
    """The definition is a fixed record of what was declared, not a live view of the caller."""

    def test_the_supplied_lists_are_readable(self):
        a = AgentDefinition(
            name="a", description="d", instructions="i", tools=["search"], guardrails=["safety"]
        )

        assert a.tools == ["search"]
        assert a.guardrails == ["safety"]

    def test_a_tool_added_by_the_caller_later_is_not_visible(self):
        declared = ["search"]
        a = AgentDefinition(name="a", description="d", instructions="i", tools=declared)

        declared.append("added_late")

        assert a.tools == ["search"]

    def test_a_capability_added_by_the_caller_later_is_not_visible(self):
        declared = [Mock(spec=Capability)]
        a = AgentDefinition(name="a", description="d", instructions="i", capabilities=declared)

        declared.append(Mock(spec=Capability))

        assert len(a.capabilities) == 1

    def test_editing_the_definition_does_not_reach_the_callers_list(self):
        declared = ["search"]
        a = AgentDefinition(name="a", description="d", instructions="i", tools=declared)

        a.tools.append("added_by_reader")

        assert declared == ["search"]

    def test_two_definitions_built_from_one_list_do_not_share_it(self):
        declared = ["search"]
        first = AgentDefinition(name="a", description="d", instructions="i", tools=declared)
        second = AgentDefinition(name="b", description="d", instructions="i", tools=declared)

        first.tools.append("added_late")

        assert second.tools == ["search"]

    def test_the_capabilities_themselves_stay_shared_with_the_caller(self):
        """Only the container is detached — the runner clones these and reads their state back."""
        budget = TurnBudget()

        a = AgentDefinition(name="a", description="d", instructions="i", capabilities=[budget])

        assert a.capabilities[0] is budget

    def test_every_collection_field_is_detached_from_the_caller(self):
        """A collection field added later without a matching copy fails here, rather than drifting."""
        seeds: dict[str, Any] = {
            "tools": ["search"],
            "guardrails": ["safety"],
            "handoffs": ["specialist"],
            "hosted_tools": ["web_search"],
            "capabilities": ["budget"],
        }
        assert set(collection_field_names(AgentDefinition)) == set(seeds), (
            "AgentDefinition gained or lost a collection field — seed it here "
            "and copy it in __post_init__"
        )
        seeded_as_supplied = copy.deepcopy(seeds)

        a = AgentDefinition(name="a", description="d", instructions="i", **seeds)
        for seeded in seeds.values():
            edit_every_level(seeded)

        for name, as_supplied in seeded_as_supplied.items():
            assert getattr(a, name) == as_supplied, f"{name} is aliased to the caller's collection"


# -- AgentRegistry -------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_get(self):
        reg = AgentRegistry()
        a = AgentDefinition(name="agent1", description="d", instructions="i")
        reg.register(a)
        got = reg.get("agent1")
        assert got.name == "agent1"
        assert got.description == "d"

    def test_get_missing_returns_none(self):
        reg = AgentRegistry()
        assert reg.get("missing") is None

    def test_list_all(self):
        reg = AgentRegistry()
        reg.register(AgentDefinition(name="a", description="d", instructions="i"))
        reg.register(AgentDefinition(name="b", description="d", instructions="i"))
        names = reg.list_all()
        assert set(names) == {"a", "b"}

    def test_overwrite_existing(self):
        reg = AgentRegistry()
        a1 = AgentDefinition(name="x", description="old", instructions="i")
        a2 = AgentDefinition(name="x", description="new", instructions="i")
        reg.register(a1)
        reg.register(a2)
        assert reg.get("x").description == "new"

    def test_register_agent_global_helper(self):
        from sinan_agentic_core.registry.agent_registry import get_agent_registry, register_agent

        a = AgentDefinition(name="_global_helper_agent", description="d", instructions="i")
        register_agent(a)
        got = get_agent_registry().get("_global_helper_agent")
        assert got is not None
        assert got.name == "_global_helper_agent"


class TestAgentRegistryOwnsItsMap:
    """The registry owns the mapping it accumulates into, not the caller's dict."""

    @staticmethod
    def _agent(name):
        return AgentDefinition(name=name, description="d", instructions="i")

    def test_an_agent_added_to_the_callers_dict_later_is_not_visible(self):
        seed = {"a": self._agent("a")}
        reg = AgentRegistry(_agents=seed)

        seed["late"] = self._agent("late")

        assert reg.get("late") is None

    def test_two_registries_built_from_one_dict_do_not_share(self):
        seed = {"a": self._agent("a")}
        first = AgentRegistry(_agents=seed)
        second = AgentRegistry(_agents=seed)

        first.register(self._agent("b"))

        assert second.get("b") is None


class TestAgentRegistryHandsOutSnapshots:
    """``get`` returns a record, not the registry's live definition.

    ``AgentDefinition`` carries five mutable list fields and the registry is
    process-wide, so a reader that mutated the object ``get`` returned would
    rewrite what every later lookup resolves. ``get`` re-runs ``__post_init__``
    via ``dataclasses.replace``, so the read boundary inherits the same
    container copy the inbound side already makes.
    """

    @staticmethod
    def _agent(name="a"):
        return AgentDefinition(name=name, description="d", instructions="i", tools=["search"])

    def test_two_gets_return_distinct_objects(self):
        reg = AgentRegistry()
        reg.register(self._agent())

        assert reg.get("a") is not reg.get("a")

    def test_appending_to_a_returned_list_does_not_reach_the_registry(self):
        reg = AgentRegistry()
        reg.register(self._agent())

        reg.get("a").tools.append("added_by_reader")

        assert reg.get("a").tools == ["search"]

    def test_the_returned_record_carries_the_registered_values(self):
        reg = AgentRegistry()
        reg.register(self._agent())

        got = reg.get("a")

        assert got.name == "a"
        assert got.tools == ["search"]

    def test_the_capabilities_themselves_stay_shared(self):
        """Only the containers are detached — capabilities stay the live objects."""
        budget = TurnBudget()
        reg = AgentRegistry()
        reg.register(
            AgentDefinition(name="a", description="d", instructions="i", capabilities=[budget])
        )

        assert reg.get("a").capabilities[0] is budget

    def test_every_collection_field_is_detached_from_the_registry(self):
        """A list field added later without the read-boundary copy fails here."""
        seeds: dict[str, Any] = {
            "tools": ["search"],
            "guardrails": ["safety"],
            "handoffs": ["specialist"],
            "hosted_tools": ["web_search"],
            "capabilities": ["budget"],
        }
        assert set(collection_field_names(AgentDefinition)) == set(seeds)
        reg = AgentRegistry()
        reg.register(
            AgentDefinition(name="a", description="d", instructions="i", **copy.deepcopy(seeds))
        )

        got = reg.get("a")
        for name in seeds:
            edit_every_level(getattr(got, name))

        fresh = reg.get("a")
        for name, original in seeds.items():
            assert getattr(fresh, name) == original, f"{name} leaked from the read boundary"


# -- resolve_agent_definition --------------------------------------------------


class TestResolveAgentDefinition:
    """A built ``Agent`` carries no slot for a definition-level setting.

    Every run-level reader that holds only a built agent — the trim policy in
    ``core/run_config.py``, the recovery flag in ``core/output_recovery.py`` —
    goes back to the declaration through this one resolver, so the by-name
    correspondence is not re-derived once per setting.
    """

    @pytest.fixture
    def registry(self):
        registry = AgentRegistry()
        registry.register(
            AgentDefinition(name="registered_agent", description="d", instructions="i")
        )

        with patch(
            "sinan_agentic_core.registry.agent_registry.get_agent_registry", return_value=registry
        ):
            yield registry

    def test_resolves_the_definition_registered_under_the_agent_name(self, registry):
        agent_def = resolve_agent_definition(Agent(name="registered_agent"))

        assert agent_def is not None
        assert agent_def.name == "registered_agent"

    def test_an_unregistered_name_resolves_to_none(self, registry):
        """A hand-assembled agent under an unknown name has no declaration."""
        assert resolve_agent_definition(Agent(name="assembled_by_hand")) is None

    def test_a_hand_assembled_agent_picks_up_the_name_it_was_given(self, registry):
        """Resolution is by name alone — the factory is not involved."""
        agent = Agent(name="registered_agent", instructions="assembled elsewhere")

        assert resolve_agent_definition(agent).description == "d"

    def test_reads_the_global_registry_when_nothing_is_patched(self):
        """Unpatched, the resolver reaches the same registry ``register_agent`` writes to."""
        from sinan_agentic_core.registry.agent_registry import register_agent

        agent_def = AgentDefinition(
            name="_global_resolved_agent", description="d", instructions="i"
        )
        register_agent(agent_def)

        resolved = resolve_agent_definition(Agent(name="_global_resolved_agent"))
        assert resolved is not None
        assert resolved.name == "_global_resolved_agent"


# -- ToolRegistry --------------------------------------------------------------


class TestToolRegistry:
    @staticmethod
    def _make_tool(name="t1", category="utility"):
        return ToolDefinition(
            name=name,
            description="desc",
            function=lambda: None,
            category=category,
            parameters_description="none",
            returns_description="none",
        )

    def test_register_and_get_tool(self):
        reg = ToolRegistry()
        t = self._make_tool()
        reg.register(t)
        got = reg.get_tool("t1")
        assert got.name == "t1"
        assert got.category == "utility"

    def test_get_tool_missing(self):
        reg = ToolRegistry()
        assert reg.get_tool("missing") is None

    def test_get_tools_by_category(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("a", "search"))
        reg.register(self._make_tool("b", "search"))
        reg.register(self._make_tool("c", "analysis"))
        search_tools = reg.get_tools_by_category("search")
        assert len(search_tools) == 2

    def test_get_tool_functions(self):
        reg = ToolRegistry()

        def fn():
            return "hello"

        reg.register(
            ToolDefinition(
                name="t",
                function=fn,
                description="d",
                category="c",
                parameters_description="p",
                returns_description="r",
            )
        )
        funcs = reg.get_tool_functions(["t", "missing"])
        assert len(funcs) == 1
        assert funcs[0] is fn

    def test_get_all_functions(self):
        reg = ToolRegistry()

        def fn1():
            return 1

        def fn2():
            return 2

        reg.register(ToolDefinition(name="a", function=fn1))
        reg.register(ToolDefinition(name="b", function=fn2))
        all_fns = reg.get_all_functions()
        assert all_fns == {"a": fn1, "b": fn2}

    def test_get_all_functions_empty(self):
        assert ToolRegistry().get_all_functions() == {}

    def test_to_instruction_text(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("my_tool", "search"))
        text = reg.to_instruction_text()
        assert "my_tool" in text
        assert "Search" in text

    def test_to_instruction_text_filtered(self):
        reg = ToolRegistry()
        reg.register(self._make_tool("tool_alpha", "search"))
        reg.register(self._make_tool("tool_beta", "analysis"))
        text = reg.to_instruction_text(tool_names=["tool_alpha"])
        assert "tool_alpha" in text
        assert "tool_beta" not in text


class TestToolRegistryOwnsItsMap:
    """The registry owns the mapping it accumulates into, not the caller's dict."""

    @staticmethod
    def _tool(name):
        return ToolDefinition(name=name, function=lambda: None)

    def test_a_tool_added_to_the_callers_dict_later_is_not_visible(self):
        seed = {"a": self._tool("a")}
        reg = ToolRegistry(_tools=seed)

        seed["late"] = self._tool("late")

        assert reg.get_tool("late") is None

    def test_two_registries_built_from_one_dict_do_not_share(self):
        seed = {"a": self._tool("a")}
        first = ToolRegistry(_tools=seed)
        second = ToolRegistry(_tools=seed)

        first.register(self._tool("b"))

        assert second.get_tool("b") is None


class TestToolRegistryHandsOutSnapshots:
    """``get_tool`` returns a record, not the registry's live definition.

    ``ToolDefinition`` has no nested container, so the exposure is field
    reassignment on the process-wide record: a reader that rewrote a field
    would change what every later lookup resolves and what
    ``to_instruction_text`` renders into the next prompt.
    """

    @staticmethod
    def _tool(name="t1"):
        return ToolDefinition(name=name, function=lambda: None, description="orig", category="c")

    def test_two_gets_return_distinct_objects(self):
        reg = ToolRegistry()
        reg.register(self._tool())

        assert reg.get_tool("t1") is not reg.get_tool("t1")

    def test_editing_a_returned_record_does_not_reach_the_registry(self):
        reg = ToolRegistry()
        reg.register(self._tool())

        reg.get_tool("t1").description = "rewritten"

        assert reg.get_tool("t1").description == "orig"

    def test_the_function_reference_stays_shared(self):
        """The record is copied, but the function it points at is the same object."""
        fn = lambda: None  # noqa: E731
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="t1", function=fn, description="orig"))

        assert reg.get_tool("t1").function is fn

    def test_get_tools_by_category_hands_out_records(self):
        reg = ToolRegistry()
        reg.register(self._tool("a"))

        [tool] = reg.get_tools_by_category("c")
        tool.description = "rewritten"

        assert reg.get_tool("a").description == "orig"


# -- register_tool decorator ---------------------------------------------------


class TestRegisterToolDecorator:
    def test_decorator_registers_and_returns_function(self):
        from sinan_agentic_core.registry.tool_registry import get_tool_registry, register_tool

        @register_tool(
            name="_deco_tool",
            description="decorated tool",
            category="test",
            parameters_description="none",
            returns_description="none",
        )
        def my_func():
            return "hello"

        # Decorator returns the original function unchanged
        assert my_func() == "hello"

        # Tool was registered in the global registry
        reg = get_tool_registry()
        tool = reg.get_tool("_deco_tool")
        assert tool is not None
        assert tool.function is my_func
        assert tool.category == "test"


# -- GuardrailRegistry ---------------------------------------------------------


class TestGuardrailRegistry:
    @staticmethod
    def _make_guardrail(name="g1", category="input"):
        return GuardrailDefinition(
            name=name, description="desc", function=lambda: None, category=category
        )

    def test_register_and_get(self):
        reg = GuardrailRegistry()
        g = self._make_guardrail()
        reg.register(g)
        got = reg.get_guardrail("g1")
        assert got.name == "g1"
        assert got.category is GuardrailCategory.INPUT

    def test_get_missing(self):
        reg = GuardrailRegistry()
        assert reg.get_guardrail("missing") is None

    def test_get_by_category(self):
        reg = GuardrailRegistry()
        reg.register(self._make_guardrail("a", "input"))
        reg.register(self._make_guardrail("b", "output"))
        reg.register(self._make_guardrail("c", "input"))
        assert len(reg.get_guardrails_by_category("input")) == 2

    def test_get_guardrail_functions(self):
        reg = GuardrailRegistry()

        def fn():
            return "check"

        reg.register(GuardrailDefinition("g", "d", fn, "input"))
        funcs = reg.get_guardrail_functions(["g", "missing"])
        assert len(funcs) == 1
        assert funcs[0] is fn

    def test_get_all_functions(self):
        reg = GuardrailRegistry()

        def fn1():
            return 1

        def fn2():
            return 2

        reg.register(GuardrailDefinition("a", "d", fn1, "input"))
        reg.register(GuardrailDefinition("b", "d", fn2, "output"))
        all_fns = reg.get_all_functions()
        assert set(all_fns.keys()) == {"a", "b"}

    def test_list_names(self):
        reg = GuardrailRegistry()
        reg.register(GuardrailDefinition("a", "d", lambda: None, "input"))
        reg.register(GuardrailDefinition("b", "d", lambda: None, "output"))
        assert reg.list_names() == ["a", "b"]


class TestGuardrailRegistryOwnsItsMap:
    """The registry owns the mapping it accumulates into, not the caller's dict."""

    @staticmethod
    def _guard(name):
        return GuardrailDefinition(
            name=name, description="d", function=lambda: None, category="input"
        )

    def test_a_guardrail_added_to_the_callers_dict_later_is_not_visible(self):
        seed = {"a": self._guard("a")}
        reg = GuardrailRegistry(_guardrails=seed)

        seed["late"] = self._guard("late")

        assert reg.get_guardrail("late") is None

    def test_two_registries_built_from_one_dict_do_not_share(self):
        seed = {"a": self._guard("a")}
        first = GuardrailRegistry(_guardrails=seed)
        second = GuardrailRegistry(_guardrails=seed)

        first.register(self._guard("b"))

        assert second.get_guardrail("b") is None


class TestGuardrailRegistryHandsOutSnapshots:
    """``get_guardrail`` returns a record, not the registry's live definition.

    ``GuardrailDefinition`` carries no nested container — the exposure is field
    reassignment on a process-wide object — but the registry answers the
    read-boundary question the same way its two siblings do. No enrichment path
    patches these records, so nothing depends on them being live.
    """

    @staticmethod
    def _guard(name="g1"):
        return GuardrailDefinition(
            name=name, description="orig", function=lambda: None, category="input"
        )

    def test_two_gets_return_distinct_objects(self):
        reg = GuardrailRegistry()
        reg.register(self._guard())

        assert reg.get_guardrail("g1") is not reg.get_guardrail("g1")

    def test_editing_a_returned_record_does_not_reach_the_registry(self):
        reg = GuardrailRegistry()
        reg.register(self._guard())

        reg.get_guardrail("g1").description = "rewritten"

        assert reg.get_guardrail("g1").description == "orig"

    def test_the_function_reference_stays_shared(self):
        """The record is copied, but the function it points at is the same object."""
        fn = lambda: None  # noqa: E731
        reg = GuardrailRegistry()
        reg.register(GuardrailDefinition("g1", "orig", fn, "input"))

        got = reg.get_guardrail("g1")
        assert got.function is fn
        assert got.category is GuardrailCategory.INPUT

    def test_get_guardrails_by_category_hands_out_records(self):
        reg = GuardrailRegistry()
        reg.register(self._guard("a"))

        [guard] = reg.get_guardrails_by_category(GuardrailCategory.INPUT)
        guard.description = "rewritten"

        assert reg.get_guardrail("a").description == "orig"


# -- Guardrail categories ------------------------------------------------------


class TestGuardrailCategory:
    def test_string_category_is_coerced(self):
        guard = GuardrailDefinition("g", "d", lambda: None, "tool_input")
        assert guard.category is GuardrailCategory.TOOL_INPUT

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="unknown category 'sideways'"):
            GuardrailDefinition("g", "d", lambda: None, "sideways")

    def test_error_lists_valid_categories(self):
        with pytest.raises(ValueError, match="input, output, tool_input"):
            GuardrailDefinition("g", "d", lambda: None, "sideways")


class TestResolveGuardrails:
    @staticmethod
    def _registry():
        reg = GuardrailRegistry()
        reg.register(GuardrailDefinition("gi", "d", lambda: "in", GuardrailCategory.INPUT))
        reg.register(GuardrailDefinition("go", "d", lambda: "out", GuardrailCategory.OUTPUT))
        reg.register(GuardrailDefinition("gt", "d", lambda: "tool", GuardrailCategory.TOOL_INPUT))
        return reg

    def test_splits_by_category(self):
        resolved = self._registry().resolve(["gi", "go", "gt"])
        assert [f() for f in resolved.input_guardrails] == ["in"]
        assert [f() for f in resolved.output_guardrails] == ["out"]
        assert [f() for f in resolved.tool_input_guardrails] == ["tool"]

    def test_unknown_names_are_skipped(self):
        resolved = self._registry().resolve(["gi", "missing"])
        assert len(resolved.input_guardrails) == 1
        assert resolved.output_guardrails == []

    def test_empty_names_resolve_to_empty_buckets(self):
        resolved = self._registry().resolve([])
        assert resolved.input_guardrails == []
        assert resolved.output_guardrails == []
        assert resolved.tool_input_guardrails == []

    def test_has_category_true(self):
        assert self._registry().has_category(["go", "gt"], GuardrailCategory.TOOL_INPUT) is True

    def test_has_category_false(self):
        assert self._registry().has_category(["gi", "go"], GuardrailCategory.TOOL_INPUT) is False

    def test_has_category_ignores_unknown_names(self):
        assert self._registry().has_category(["missing"], GuardrailCategory.INPUT) is False


class TestResolvedGuardrailsOwnsItsLists:
    """A resolved bundle is a fixed record of one resolution, not a live view.

    ``resolve`` passes fresh buckets, but the type is exported from the package
    root, so a caller can build one directly with lists it keeps.
    """

    def test_a_guardrail_added_by_the_caller_later_is_not_visible(self):
        declared = ["g"]
        bundle = ResolvedGuardrails(input_guardrails=declared)

        declared.append("added_late")

        assert bundle.input_guardrails == ["g"]

    def test_editing_the_bundle_does_not_reach_the_callers_list(self):
        declared = ["g"]
        bundle = ResolvedGuardrails(input_guardrails=declared)

        bundle.input_guardrails.append("added_by_reader")

        assert declared == ["g"]

    def test_two_bundles_built_from_one_list_do_not_share_it(self):
        declared = ["g"]
        first = ResolvedGuardrails(output_guardrails=declared)
        second = ResolvedGuardrails(output_guardrails=declared)

        first.output_guardrails.append("added_late")

        assert second.output_guardrails == ["g"]

    def test_the_elements_stay_shared(self):
        """Only the outer list is detached — SDK guardrail objects match by identity."""
        sentinel = object()
        bundle = ResolvedGuardrails(tool_input_guardrails=[sentinel])

        assert bundle.tool_input_guardrails[0] is sentinel

    def test_all_three_lists_are_detached(self):
        inputs, outputs, tool_inputs = ["i"], ["o"], ["t"]
        bundle = ResolvedGuardrails(
            input_guardrails=inputs,
            output_guardrails=outputs,
            tool_input_guardrails=tool_inputs,
        )

        inputs.append("x")
        outputs.append("x")
        tool_inputs.append("x")

        assert bundle.input_guardrails == ["i"]
        assert bundle.output_guardrails == ["o"]
        assert bundle.tool_input_guardrails == ["t"]


# -- attach_tool_input_guardrails ----------------------------------------------


@pytest.fixture
def echo_tool():
    """A local function tool, fresh per test so attachments cannot leak."""

    @function_tool
    def echo(value: str) -> str:
        """Echo a value.

        Args:
            value: Text to echo back.
        """
        return value

    return echo


class TestAttachToolInputGuardrails:
    def test_attaches_to_function_tools(self, echo_tool):
        result = attach_tool_input_guardrails([echo_tool], ["g"])

        assert result[0].tool_input_guardrails == ["g"]

    def test_skips_non_function_tools(self):
        hosted = object()
        assert attach_tool_input_guardrails([hosted], ["g"]) == [hosted]

    def test_without_guardrails_returns_same_list(self):
        tools = [object()]
        assert attach_tool_input_guardrails(tools, []) is tools

    def test_preserves_existing_tool_guardrails(self, echo_tool):
        preset = copy.copy(echo_tool)
        preset.tool_input_guardrails = ["existing"]

        result = attach_tool_input_guardrails([preset], ["added"])

        assert result[0].tool_input_guardrails == ["existing", "added"]

    def test_source_tool_is_not_mutated(self, echo_tool):
        attach_tool_input_guardrails([echo_tool], ["g"])

        assert echo_tool.tool_input_guardrails is None


# -- register_guardrail decorator + get_guardrail_registry ---------------------


class TestRegisterGuardrailDecorator:
    def test_get_guardrail_registry_returns_singleton(self):
        from sinan_agentic_core.registry.guardrail_registry import get_guardrail_registry

        reg1 = get_guardrail_registry()
        reg2 = get_guardrail_registry()
        assert reg1 is reg2

    def test_decorator_registers_and_returns_function(self):
        from sinan_agentic_core.registry.guardrail_registry import (
            get_guardrail_registry,
            register_guardrail,
        )

        @register_guardrail(
            name="_deco_guard",
            description="decorated guardrail",
            category="input",
        )
        def my_guard():
            return "checked"

        assert my_guard() == "checked"

        reg = get_guardrail_registry()
        guard = reg.get_guardrail("_deco_guard")
        assert guard is not None
        assert guard.function is my_guard
        assert guard.category is GuardrailCategory.INPUT

    def test_decorator_reports_unknown_category_with_guardrail_name(self):
        from sinan_agentic_core.registry.guardrail_registry import register_guardrail

        with pytest.raises(ValueError, match="Guardrail '_bad_guard' has unknown category"):

            @register_guardrail(
                name="_bad_guard",
                description="bad category",
                category="sideways",
            )
            def my_guard():
                return "checked"


class TestGuardrailPackageExports:
    """The guardrail surface is reachable from the package root, not just the sub-package."""

    GUARDRAIL_SYMBOLS = (
        "GuardrailCategory",
        "GuardrailDefinition",
        "GuardrailRegistry",
        "ResolvedGuardrails",
        "get_guardrail_registry",
        "register_guardrail",
    )

    @pytest.mark.parametrize("symbol", GUARDRAIL_SYMBOLS)
    def test_exported_from_package_root(self, symbol):
        import sinan_agentic_core as pkg

        assert hasattr(pkg, symbol)
        assert symbol in pkg.__all__

    @pytest.mark.parametrize("symbol", GUARDRAIL_SYMBOLS)
    def test_exported_from_registry_package(self, symbol):
        import sinan_agentic_core.registry as registry

        assert hasattr(registry, symbol)
        assert symbol in registry.__all__


# -- AgentFactory --------------------------------------------------------------


class TestAgentFactory:
    def test_create_agent_success(self):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry
        from sinan_agentic_core.registry.agent_registry import get_agent_registry
        from sinan_agentic_core.registry.tool_registry import get_tool_registry

        tool_reg = get_tool_registry()

        def tool_fn():
            return None

        tool_reg.register(
            ToolDefinition(
                name="_factory_tool",
                function=tool_fn,
                description="desc",
                category="cat",
                parameters_description="p",
                returns_description="r",
            )
        )

        get_agent_registry().register(
            AgentDefinition(
                name="_factory_agent",
                description="test",
                instructions="You are a factory test agent",
                tools=["_factory_tool"],
            )
        )

        agent = create_agent_from_registry("_factory_agent")
        assert agent.name == "_factory_agent"

    def test_create_agent_not_found(self):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        with pytest.raises(ValueError, match="not found"):
            create_agent_from_registry("__nonexistent_factory_agent__")

    def test_model_override(self):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry
        from sinan_agentic_core.registry.agent_registry import get_agent_registry

        get_agent_registry().register(
            AgentDefinition(
                name="_override_agent",
                description="test",
                instructions="test",
                model="gpt-4o-mini",
            )
        )

        agent = create_agent_from_registry("_override_agent", model_override="gpt-4o")
        assert agent.model == "gpt-4o"


# -- AgentFactory guardrail wiring ---------------------------------------------


@pytest.fixture
def factory_guardrails(echo_tool, monkeypatch):
    """Global agent/tool registries carrying one guardrail per category.

    The factory reads the global registries, so the guardrail registry is
    swapped for an isolated one and the agent/tool entries use ``_``-prefixed
    names like the rest of the factory tests.
    """
    from sinan_agentic_core.registry import agent_factory
    from sinan_agentic_core.registry.agent_registry import get_agent_registry
    from sinan_agentic_core.registry.tool_registry import get_tool_registry

    @input_guardrail
    def guard_input(ctx, agent, agent_input):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    @output_guardrail
    def guard_output(ctx, agent, agent_output):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    @tool_input_guardrail
    def guard_tool_input(data):
        return ToolGuardrailFunctionOutput.allow()

    guardrail_reg = GuardrailRegistry()
    guardrail_reg.register(GuardrailDefinition("_fg_in", "d", guard_input, GuardrailCategory.INPUT))
    guardrail_reg.register(
        GuardrailDefinition("_fg_out", "d", guard_output, GuardrailCategory.OUTPUT)
    )
    guardrail_reg.register(
        GuardrailDefinition("_fg_tool", "d", guard_tool_input, GuardrailCategory.TOOL_INPUT)
    )
    monkeypatch.setattr(agent_factory, "get_guardrail_registry", lambda: guardrail_reg)

    get_tool_registry().register(ToolDefinition(name="_fg_echo", function=echo_tool))
    get_agent_registry().register(
        AgentDefinition(
            name="_fg_guarded_agent",
            description="guarded",
            instructions="You are guarded",
            tools=["_fg_echo"],
            guardrails=["_fg_in", "_fg_out", "_fg_tool"],
        )
    )
    get_agent_registry().register(
        AgentDefinition(
            name="_fg_plain_agent",
            description="plain",
            instructions="You are plain",
            tools=["_fg_echo"],
        )
    )

    return echo_tool


class TestAgentFactoryGuardrails:
    def test_input_and_output_guardrails_land_in_their_slots(self, factory_guardrails):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        agent = create_agent_from_registry("_fg_guarded_agent")

        assert [g.get_name() for g in agent.input_guardrails] == ["guard_input"]
        assert [g.get_name() for g in agent.output_guardrails] == ["guard_output"]

    def test_tool_input_guardrails_attach_to_tools(self, factory_guardrails):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        agent = create_agent_from_registry("_fg_guarded_agent")

        assert [g.get_name() for g in agent.tools[0].tool_input_guardrails] == ["guard_tool_input"]

    def test_registry_tool_is_not_mutated(self, factory_guardrails):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        create_agent_from_registry("_fg_guarded_agent")

        assert factory_guardrails.tool_input_guardrails is None

    def test_agent_without_guardrails_gets_empty_slots(self, factory_guardrails):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        agent = create_agent_from_registry("_fg_plain_agent")

        assert agent.input_guardrails == []
        assert agent.output_guardrails == []
        assert agent.tools[0].tool_input_guardrails is None


# -- AgentFactory model retry wiring -------------------------------------------


@pytest.fixture
def factory_retry():
    """Global registry carrying one agent declaring a retry policy and one without."""
    from sinan_agentic_core.registry.agent_registry import get_agent_registry

    get_agent_registry().register(
        AgentDefinition(
            name="_fr_retrying_agent",
            description="retries",
            instructions="You retry",
            model_retry=ModelRetryConfig(max_retries=4),
        )
    )
    get_agent_registry().register(
        AgentDefinition(
            name="_fr_plain_agent",
            description="plain",
            instructions="You are plain",
        )
    )


class TestAgentFactoryModelRetry:
    def test_declared_policy_rides_on_the_agent_model_settings(self, factory_retry):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        agent = create_agent_from_registry("_fr_retrying_agent")

        assert agent.model_settings.retry.max_retries == 4
        assert agent.model_settings.retry.policy is not None

    def test_agent_without_a_policy_keeps_the_sdk_default_settings(self, factory_retry):
        """Omitting the kwarg matters: Agent.model_settings must stay a real object."""
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        agent = create_agent_from_registry("_fr_plain_agent")

        assert agent.model_settings.retry is None

    def test_model_override_leaves_the_policy_in_place(self, factory_retry):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        agent = create_agent_from_registry("_fr_retrying_agent", model_override="gpt-4o")

        assert agent.model == "gpt-4o"
        assert agent.model_settings.retry.max_retries == 4


# -- AgentFactory run-level settings -------------------------------------------


class TestAgentFactoryRunLevelSettings:
    """What ``build_run_config`` / ``build_error_handlers`` resolve off a factory agent.

    The factory docstring tells callers driving ``Runner`` themselves to pass
    both resolvers, and states that error handlers stay ``None`` because the
    factory does not carry ``output_dataclass`` onto the agent. These pin that
    claim to the code, so wiring an output type later fails here rather than
    leaving the docstring wrong.
    """

    def test_guarded_agent_resolves_a_run_config(self, factory_guardrails):
        from sinan_agentic_core.core.run_config import build_run_config
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry

        run_config = build_run_config(create_agent_from_registry("_fg_guarded_agent"))

        assert run_config is not None
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    def test_declared_trim_policy_resolves_a_run_config(self):
        """The factory builds no run config, so trimming has to survive the round trip."""
        from sinan_agentic_core.core.run_config import build_run_config
        from sinan_agentic_core.core.tool_output_trim import ToolOutputTrimConfig
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry
        from sinan_agentic_core.registry.agent_registry import get_agent_registry

        get_agent_registry().register(
            AgentDefinition(
                name="_frl_trimming_agent",
                description="trims",
                instructions="You answer",
                tool_output_trim=ToolOutputTrimConfig(max_output_chars=4000),
            )
        )

        run_config = build_run_config(create_agent_from_registry("_frl_trimming_agent"))

        assert run_config is not None
        assert run_config.call_model_input_filter.max_output_chars == 4000

    def test_declared_output_dataclass_resolves_no_error_handlers(self):
        from dataclasses import dataclass

        from sinan_agentic_core.core.output_recovery import build_error_handlers
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry
        from sinan_agentic_core.registry.agent_registry import get_agent_registry

        @dataclass
        class _Extraction:
            value: str

        get_agent_registry().register(
            AgentDefinition(
                name="_frl_structured_agent",
                description="structured",
                instructions="You extract",
                output_dataclass=_Extraction,
            )
        )

        agent = create_agent_from_registry("_frl_structured_agent")

        assert agent.output_type is None
        assert build_error_handlers(agent) is None
