"""Tests for agent, tool, and guardrail registries."""

import pytest

from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry
from sinan_agentic_core.registry.guardrail_registry import GuardrailDefinition, GuardrailRegistry


# -- AgentDefinition -----------------------------------------------------------


class TestAgentDefinition:
    def test_valid_creation(self):
        a = AgentDefinition(name="a1", description="desc", instructions="do stuff")
        assert a.name == "a1"
        assert a.model == "gpt-4o-mini"
        assert a.tools == []

    def test_missing_instructions_raises(self):
        with pytest.raises(ValueError, match="must have either instructions"):
            AgentDefinition(name="bad", description="d")

    def test_callable_instructions(self):
        fn = lambda ctx, agent: "dynamic"
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
        factory = lambda: "mock_tool"
        a = AgentDefinition(name="a", description="d", instructions="i", hosted_tools=[factory])
        assert len(a.hosted_tools) == 1
        assert a.hosted_tools[0]() == "mock_tool"


# -- AgentRegistry -------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_get(self):
        reg = AgentRegistry()
        a = AgentDefinition(name="agent1", description="d", instructions="i")
        reg.register(a)
        assert reg.get("agent1") is a

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
        from sinan_agentic_core.registry.agent_registry import register_agent, get_agent_registry

        a = AgentDefinition(name="_global_helper_agent", description="d", instructions="i")
        register_agent(a)
        assert get_agent_registry().get("_global_helper_agent") is a


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
        assert reg.get_tool("t1") is t

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
        fn = lambda: "hello"
        reg.register(ToolDefinition(name="t", function=fn, description="d", category="c", parameters_description="p", returns_description="r"))
        funcs = reg.get_tool_functions(["t", "missing"])
        assert len(funcs) == 1
        assert funcs[0] is fn

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


# -- register_tool decorator ---------------------------------------------------


class TestRegisterToolDecorator:
    def test_decorator_registers_and_returns_function(self):
        from sinan_agentic_core.registry.tool_registry import register_tool, get_tool_registry

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
        assert reg.get_guardrail("g1") is g

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
        fn = lambda: "check"
        reg.register(GuardrailDefinition("g", "d", fn, "input"))
        funcs = reg.get_guardrail_functions(["g", "missing"])
        assert len(funcs) == 1
        assert funcs[0] is fn

    def test_get_all_functions(self):
        reg = GuardrailRegistry()
        fn1 = lambda: 1
        fn2 = lambda: 2
        reg.register(GuardrailDefinition("a", "d", fn1, "input"))
        reg.register(GuardrailDefinition("b", "d", fn2, "output"))
        all_fns = reg.get_all_functions()
        assert set(all_fns.keys()) == {"a", "b"}


# -- register_guardrail decorator + get_guardrail_registry ---------------------


class TestRegisterGuardrailDecorator:
    def test_get_guardrail_registry_returns_singleton(self):
        from sinan_agentic_core.registry.guardrail_registry import get_guardrail_registry

        reg1 = get_guardrail_registry()
        reg2 = get_guardrail_registry()
        assert reg1 is reg2

    def test_decorator_registers_and_returns_function(self):
        from sinan_agentic_core.registry.guardrail_registry import (
            register_guardrail,
            get_guardrail_registry,
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
        assert guard.category == "input"


# -- AgentFactory --------------------------------------------------------------


class TestAgentFactory:
    def test_create_agent_success(self):
        from sinan_agentic_core.registry.agent_factory import create_agent_from_registry
        from sinan_agentic_core.registry.agent_registry import get_agent_registry
        from sinan_agentic_core.registry.tool_registry import get_tool_registry

        tool_reg = get_tool_registry()
        tool_fn = lambda: None
        tool_reg.register(
            ToolDefinition(name="_factory_tool", function=tool_fn, description="desc", category="cat", parameters_description="p", returns_description="r")
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
