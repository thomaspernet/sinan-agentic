"""Tests for BaseAgentRunner (core/base_runner.py)."""

import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from agents import Usage

from sinan_agentic_core.core.base_runner import BaseAgentRunner, _CollectingSessionWrapper
from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from sinan_agentic_core.registry.guardrail_registry import GuardrailDefinition, GuardrailRegistry
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry
from sinan_agentic_core.session.agent_session import AgentSession


@pytest.fixture
def _registries():
    """Build isolated registries with sample data."""
    agent_reg = AgentRegistry()
    tool_reg = ToolRegistry()
    guardrail_reg = GuardrailRegistry()

    tool_fn = lambda: "result"
    tool_reg.register(ToolDefinition(name="test_tool", function=tool_fn, description="desc", category="cat", parameters_description="p", returns_description="r"))

    guardrail_fn = lambda: True
    guardrail_reg.register(GuardrailDefinition("test_guard", "desc", guardrail_fn, "output"))

    agent_reg.register(
        AgentDefinition(
            name="basic_agent",
            description="basic",
            instructions="You are a basic agent",
            tools=["test_tool"],
            guardrails=["test_guard"],
        )
    )

    return agent_reg, tool_reg, guardrail_reg


@pytest.fixture
def runner(_registries):
    """Instantiate BaseAgentRunner with patched registries."""
    agent_reg, tool_reg, guardrail_reg = _registries

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch("sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg),
    ):
        return BaseAgentRunner()


# ------------------------------------------------------------------ #
# Init and setup helpers
# ------------------------------------------------------------------ #


class TestBaseAgentRunnerInit:
    def test_loads_tool_map(self, runner):
        assert "test_tool" in runner.tool_map

    def test_loads_guardrail_map(self, runner):
        assert "test_guard" in runner.guardrail_map

    def test_is_not_abstract(self):
        """BaseAgentRunner should be instantiable directly (not ABC)."""
        with (
            patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=AgentRegistry()),
            patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=ToolRegistry()),
            patch(
                "sinan_agentic_core.core.base_runner.get_guardrail_registry",
                return_value=GuardrailRegistry(),
            ),
        ):
            runner = BaseAgentRunner()
            assert runner is not None


class TestSetupHelpers:
    def test_setup_context(self, runner):
        ctx = runner.setup_context(database_connector=Mock())
        assert isinstance(ctx, AgentContext)
        assert ctx.has_data is False

    def test_setup_session_with_id(self, runner):
        session = runner.setup_session(session_id="my-id")
        assert session.session_id == "my-id"

    def test_setup_session_generates_uuid(self, runner):
        session = runner.setup_session()
        assert len(session.session_id) > 0

    def test_setup_session_with_history(self, runner):
        history = [{"role": "user", "content": "hello"}]
        session = runner.setup_session(session_id="h1", initial_history=history)
        assert session.session_id == "h1"


class TestAggregateUsage:
    def test_single_response(self, runner, mock_run_result):
        usage = runner._aggregate_usage(mock_run_result)
        assert usage["requests"] == 1
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_empty_responses(self, runner):
        result = Mock()
        result.raw_responses = []
        usage = runner._aggregate_usage(result)
        assert usage["total_tokens"] == 0


# ------------------------------------------------------------------ #
# create_agent
# ------------------------------------------------------------------ #


class TestCreateAgent:
    async def test_basic_agent(self, runner):
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("basic_agent", ctx)
        assert agent.name == "basic_agent"

    async def test_not_found_raises(self, runner):
        ctx = AgentContext(database_connector=Mock())
        with pytest.raises(ValueError, match="not found"):
            await runner.create_agent("nonexistent", ctx)

    async def test_callable_instructions(self, runner):
        runner.agent_registry.register(
            AgentDefinition(
                name="dynamic_agent",
                description="dynamic",
                instructions=lambda ctx, agent: "dynamic instructions",
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("dynamic_agent", ctx)
        assert agent.name == "dynamic_agent"

    async def test_output_dataclass_type(self, runner):
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str

        runner.agent_registry.register(
            AgentDefinition(
                name="typed_agent",
                description="typed",
                instructions="test",
                output_dataclass=MyOutput,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("typed_agent", ctx)
        assert agent.name == "typed_agent"

    async def test_output_dataclass_string(self, runner):
        runner.agent_registry.register(
            AgentDefinition(
                name="str_typed_agent",
                description="typed by name",
                instructions="test",
                output_dataclass="ChatResponse",
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("str_typed_agent", ctx)
        assert agent.name == "str_typed_agent"

    async def test_handoffs(self, runner):
        runner.agent_registry.register(
            AgentDefinition(name="target_agent", description="target", instructions="target")
        )
        runner.agent_registry.register(
            AgentDefinition(
                name="source_agent",
                description="source",
                instructions="source",
                handoffs=["target_agent"],
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("source_agent", ctx)
        assert agent.name == "source_agent"

    async def test_agent_as_tool(self, runner):
        runner.agent_registry.register(
            AgentDefinition(name="sub_agent", description="sub desc", instructions="sub")
        )
        runner.agent_registry.register(
            AgentDefinition(
                name="parent_agent",
                description="parent",
                instructions="parent",
                tools=["sub_agent"],
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("parent_agent", ctx)
        assert agent.name == "parent_agent"

    async def test_model_settings_fn(self, runner):
        from agents import ModelSettings

        runner.agent_registry.register(
            AgentDefinition(
                name="settings_agent",
                description="with settings",
                instructions="test",
                model_settings_fn=lambda ctx: ModelSettings(temperature=0.5),
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("settings_agent", ctx)
        assert agent.name == "settings_agent"

    async def test_hosted_tools_included(self, runner):
        mock_tool = Mock()
        runner.agent_registry.register(
            AgentDefinition(
                name="hosted_agent",
                description="has hosted tools",
                instructions="test",
                hosted_tools=[lambda: mock_tool],
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("hosted_agent", ctx)
        assert agent.name == "hosted_agent"
        assert mock_tool in agent.tools


# ------------------------------------------------------------------ #
# run_agent (backward-compatible)
# ------------------------------------------------------------------ #


class TestRunAgent:
    async def test_returns_output_and_usage(self, runner, mock_run_result):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="run-test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)
            result = await runner.run_agent("basic_agent", session, ctx, "hello")

        assert result["output"] == "Test response"
        assert result["usage"]["input_tokens"] == 100


# ------------------------------------------------------------------ #
# _build_hosted_tools
# ------------------------------------------------------------------ #


class TestBuildHostedTools:
    def test_callable_factory(self, runner):
        mock_tool = Mock()
        tools = runner._build_hosted_tools([lambda: mock_tool])
        assert len(tools) == 1
        assert tools[0] is mock_tool

    def test_direct_instance(self, runner):
        # A non-callable instance should be passed through directly
        mock_tool = "direct_tool_instance"  # str is not callable in _build_hosted_tools context
        # Actually, str IS callable. Use an object that isn't callable.
        class NonCallable:
            pass
        obj = NonCallable()
        tools = runner._build_hosted_tools([obj])
        assert len(tools) == 1
        assert tools[0] is obj

    def test_factory_error_handled(self, runner):
        def bad_factory():
            raise RuntimeError("broken")
        tools = runner._build_hosted_tools([bad_factory])
        assert len(tools) == 0

    def test_empty_list(self, runner):
        assert runner._build_hosted_tools([]) == []


# ------------------------------------------------------------------ #
# _CollectingSessionWrapper
# ------------------------------------------------------------------ #


class TestCollectingSessionWrapper:
    async def test_captures_raw_items(self):
        session = AgentSession(session_id="test")
        wrapper = _CollectingSessionWrapper(session)

        items = [
            {"role": "user", "content": "hello"},
            {"type": "function_call_output", "output": '{"data": "value"}'},
        ]
        await wrapper.add_items(items)
        assert len(wrapper.raw_items) == 2
        assert wrapper.raw_items[1]["type"] == "function_call_output"

    async def test_delegates_to_real_session(self):
        session = AgentSession(session_id="test")
        wrapper = _CollectingSessionWrapper(session)

        await wrapper.add_items([{"role": "user", "content": "hello"}])
        items = await wrapper.get_items()
        assert len(items) == 1

    async def test_clear_clears_both(self):
        session = AgentSession(session_id="test")
        wrapper = _CollectingSessionWrapper(session)

        await wrapper.add_items([{"role": "user", "content": "hello"}])
        await wrapper.clear_session()
        assert wrapper.raw_items == []
        items = await session.get_items()
        assert len(items) == 0

    def test_session_id_passthrough(self):
        session = AgentSession(session_id="original")
        wrapper = _CollectingSessionWrapper(session)
        assert wrapper.session_id == "original"
        wrapper.session_id = "new"
        assert session.session_id == "new"

    async def test_pop_item_delegates(self):
        session = AgentSession(session_id="test")
        wrapper = _CollectingSessionWrapper(session)
        await wrapper.add_items([{"role": "user", "content": "hello"}])
        item = await wrapper.pop_item()
        assert item is not None


# ------------------------------------------------------------------ #
# execute() — basic mode
# ------------------------------------------------------------------ #


class TestExecuteBasic:
    async def test_returns_final_output_directly(self, runner, mock_run_result):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="exec-test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)
            result = await runner.execute("basic_agent", ctx, session, input_text="hello")

        # execute() returns final_output directly, not wrapped in {"output": ...}
        assert result == "Test response"

    async def test_max_turns_passed(self, runner, mock_run_result):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)
            await runner.execute("basic_agent", ctx, session, max_turns=20)
            call_kwargs = mock_runner_cls.run.call_args.kwargs
            assert call_kwargs["max_turns"] == 20

    async def test_routes_to_basic_by_default(self, runner, mock_run_result):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "_execute_basic", new_callable=AsyncMock, return_value="ok") as m,
        ):
            result = await runner.execute("basic_agent", ctx, session)
            m.assert_called_once()
            assert result == "ok"


# ------------------------------------------------------------------ #
# execute() — streaming mode
# ------------------------------------------------------------------ #


class TestExecuteStreaming:
    async def test_routes_to_streamed(self, runner):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with patch.object(
            runner, "_execute_streamed", new_callable=AsyncMock, return_value="streamed"
        ) as m:
            result = await runner.execute(
                "basic_agent", ctx, session, streaming=True, on_event=lambda e: None
            )
            m.assert_called_once()
            assert result == "streamed"

    async def test_max_turns_passed_to_run_streamed(self, runner):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        mock_result = Mock()
        mock_result.final_output = "ok"
        mock_result.raw_responses = []

        async def mock_stream_events():
            return
            yield

        mock_result.stream_events = mock_stream_events

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run_streamed = Mock(return_value=mock_result)
            await runner._execute_streamed(
                "basic_agent", ctx, session, lambda e: None, 30, "hello"
            )
            call_kwargs = mock_runner_cls.run_streamed.call_args.kwargs
            assert call_kwargs["max_turns"] == 30

    async def test_on_event_receives_answer(self, runner):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")
        events = []

        # Build a mock streamed result
        mock_result = Mock()
        mock_result.final_output = "Streamed answer"
        mock_result.raw_responses = []

        async def mock_stream_events():
            return
            yield  # make it an async generator

        mock_result.stream_events = mock_stream_events

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run_streamed = Mock(return_value=mock_result)
            result = await runner._execute_streamed(
                "basic_agent", ctx, session, lambda e: events.append(e), 10, "hello"
            )

        assert result == "Streamed answer"
        # Should have received an answer event
        answer_events = [e for e in events if e["event"] == "answer"]
        assert len(answer_events) == 1
        assert answer_events[0]["data"]["response"] == "Streamed answer"
        assert "usage" in answer_events[0]["data"]


# ------------------------------------------------------------------ #
# execute() — fallback mode
# ------------------------------------------------------------------ #


class TestExecuteWithFallback:
    async def test_normal_success_returns_final_output(self, runner, mock_run_result):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)
            result = await runner.execute(
                "basic_agent", ctx, session, fallback_on_overflow=True, input_text="hello"
            )

        assert result == "Test response"

    async def test_routes_to_fallback(self, runner):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with patch.object(
            runner, "_execute_with_fallback", new_callable=AsyncMock, return_value="fallback"
        ) as m:
            result = await runner.execute(
                "basic_agent", ctx, session, fallback_on_overflow=True
            )
            m.assert_called_once()
            assert result == "fallback"

    async def test_non_recoverable_error_propagates(self, runner):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=RuntimeError("Something else broke"))
            with pytest.raises(RuntimeError, match="Something else broke"):
                await runner._execute_with_fallback(
                    "basic_agent", ctx, session, 10, "hello", None
                )

    async def test_fallback_on_max_turns(self, runner):
        """Fallback with str output_type returns raw text."""
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Fallback answer from LLM"

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=RuntimeError("Max turns exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_openai.return_value = mock_client

            result = await runner._execute_with_fallback(
                "basic_agent", ctx, session, 10, "hello", None
            )

        # basic_agent has output_type=str, so fallback returns raw text
        assert result == "Fallback answer from LLM"

    async def test_fallback_on_max_turns_structured(self, runner):
        """Fallback with structured output_type returns parsed object."""
        from pydantic import BaseModel

        class ExtractOutput(BaseModel):
            answer: str

        runner.agent_registry.register(
            AgentDefinition(
                name="structured_agent",
                description="structured",
                instructions="extract data",
                output_dataclass=ExtractOutput,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = '{"answer": "fallback result"}'

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=RuntimeError("Max turns exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_openai.return_value = mock_client

            result = await runner._execute_with_fallback(
                "structured_agent", ctx, session, 10, "hello", None
            )

        assert result.answer == "fallback result"

    async def test_custom_fallback_prompt_builder(self, runner):
        from pydantic import BaseModel

        class CustomOutput(BaseModel):
            custom: bool

        runner.agent_registry.register(
            AgentDefinition(
                name="custom_fb_agent",
                description="custom",
                instructions="custom instructions",
                output_dataclass=CustomOutput,
            )
        )

        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")
        builder_called_with = {}

        def custom_builder(instructions, raw_items, agent_def):
            builder_called_with["instructions"] = instructions
            builder_called_with["raw_items"] = raw_items
            return "Custom fallback prompt"

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = '{"custom": true}'

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_runner_cls.run = AsyncMock(
                side_effect=RuntimeError("context_length_exceeded")
            )
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_openai.return_value = mock_client

            result = await runner._execute_with_fallback(
                "custom_fb_agent", ctx, session, 10, "hello", custom_builder
            )

        assert "instructions" in builder_called_with
        assert result.custom is True

    async def test_fallback_str_output_type(self, runner):
        """When output_type is str, fallback returns raw text (not JSON)."""
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Plain text fallback"

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("openai.AsyncOpenAI") as mock_openai,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=RuntimeError("Max turns exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_openai.return_value = mock_client

            result = await runner._execute_with_fallback(
                "basic_agent", ctx, session, 10, "hello", None
            )

        assert result == "Plain text fallback"


# ------------------------------------------------------------------ #
# _default_fallback_prompt_builder
# ------------------------------------------------------------------ #


class TestDefaultFallbackPromptBuilder:
    def test_concatenates_tool_outputs(self):
        raw_items = [
            {"type": "function_call_output", "output": '{"summary": "found data"}'},
            {"type": "function_call_output", "output": '{"results": []}'},
            {"role": "user", "content": "ignored"},
        ]
        prompt = BaseAgentRunner._default_fallback_prompt_builder(
            "Test instructions", raw_items, None
        )
        assert "Test instructions" in prompt
        assert "found data" in prompt
        assert "Gathered Context" in prompt

    def test_no_tool_outputs(self):
        prompt = BaseAgentRunner._default_fallback_prompt_builder(
            "Instructions", [], None
        )
        assert "no tool outputs collected" in prompt

    def test_skips_non_dict_items(self):
        raw_items = ["string_item", 42, None]
        prompt = BaseAgentRunner._default_fallback_prompt_builder(
            "Instructions", raw_items, None
        )
        assert "no tool outputs collected" in prompt


# ------------------------------------------------------------------ #
# Private helper methods
# ------------------------------------------------------------------ #


class TestPrivateHelpers:
    def test_get_agent_definition_found(self, runner):
        agent_def = runner._get_agent_definition("basic_agent")
        assert agent_def.name == "basic_agent"

    def test_get_agent_definition_not_found(self, runner):
        with pytest.raises(ValueError, match="not found"):
            runner._get_agent_definition("nonexistent")

    def test_build_instructions_string(self, runner):
        agent_def = Mock()
        agent_def.instructions = "static instructions"
        ctx_wrapper = Mock()
        assert runner._build_instructions(agent_def, ctx_wrapper) == "static instructions"

    def test_build_instructions_callable(self, runner):
        agent_def = Mock()
        agent_def.instructions = lambda ctx, agent: "dynamic"
        ctx_wrapper = Mock()
        assert runner._build_instructions(agent_def, ctx_wrapper) == "dynamic"

    def test_resolve_output_type_none(self, runner):
        assert runner._resolve_output_type(None) is str

    def test_resolve_output_type_class(self, runner):
        class MyType:
            pass
        assert runner._resolve_output_type(MyType) is MyType

    def test_resolve_output_type_string(self, runner):
        result = runner._resolve_output_type("ChatResponse")
        from sinan_agentic_core.models.outputs import ChatResponse
        assert result is ChatResponse

    def test_resolve_output_type_unknown_string(self, runner):
        result = runner._resolve_output_type("NonexistentType")
        assert result is str

    def test_build_guardrails_found(self, runner):
        guardrails = runner._build_guardrails(["test_guard"])
        assert len(guardrails) == 1

    def test_build_guardrails_not_found(self, runner):
        guardrails = runner._build_guardrails(["nonexistent_guard"])
        assert len(guardrails) == 0

    def test_build_model_settings_none(self, runner):
        agent_def = Mock()
        agent_def.model_settings_fn = None
        assert runner._build_model_settings(agent_def, Mock()) is None

    def test_build_model_settings_error(self, runner):
        agent_def = Mock()
        agent_def.model_settings_fn = Mock(side_effect=RuntimeError("bad"))
        assert runner._build_model_settings(agent_def, Mock()) is None

    def test_build_agent_kwargs_basic(self, runner):
        agent_def = Mock()
        agent_def.name = "test"
        agent_def.model = "gpt-4o"
        kwargs = runner._build_agent_kwargs(
            agent_def=agent_def,
            instructions="inst",
            tools=[],
            guardrails=[],
            handoffs=[],
            output_type=str,
            model_settings=None,
        )
        assert kwargs["name"] == "test"
        assert "handoffs" not in kwargs
        assert "model_settings" not in kwargs

    def test_build_agent_kwargs_with_handoffs_and_settings(self, runner):
        agent_def = Mock()
        agent_def.name = "test"
        agent_def.model = "gpt-4o"
        kwargs = runner._build_agent_kwargs(
            agent_def=agent_def,
            instructions="inst",
            tools=[],
            guardrails=[],
            handoffs=["handoff1"],
            output_type=str,
            model_settings={"temperature": 0.5},
        )
        assert kwargs["handoffs"] == ["handoff1"]
        assert kwargs["model_settings"] == {"temperature": 0.5}


# ------------------------------------------------------------------ #
# Structured agent-as-tool
# ------------------------------------------------------------------ #


class TestStructuredAgentAsTool:
    async def test_build_tools_with_parameters(self, runner):
        """Agent-as-tool with as_tool_parameters passes parameters to as_tool()."""
        from dataclasses import dataclass

        @dataclass
        class ActionInput:
            action: str
            target_uuid: str

        runner.agent_registry.register(
            AgentDefinition(
                name="param_sub_agent",
                description="sub with params",
                instructions="sub",
                as_tool_parameters=ActionInput,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        tools = await runner._build_tools(["param_sub_agent"], ctx)
        assert len(tools) == 1
        assert tools[0].name == "param_sub_agent"

    async def test_build_tools_without_parameters(self, runner):
        """Agent-as-tool without as_tool_parameters uses default input."""
        runner.agent_registry.register(
            AgentDefinition(
                name="plain_sub_agent",
                description="sub without params",
                instructions="sub",
            )
        )
        ctx = AgentContext(database_connector=Mock())
        tools = await runner._build_tools(["plain_sub_agent"], ctx)
        assert len(tools) == 1
        assert tools[0].name == "plain_sub_agent"

    async def test_agent_def_as_tool_parameters_default_none(self):
        """AgentDefinition.as_tool_parameters defaults to None."""
        agent_def = AgentDefinition(
            name="test", description="test", instructions="test"
        )
        assert agent_def.as_tool_parameters is None


class TestBudgetAwareAgentAsTool:
    async def test_build_tools_with_turn_budget(self, runner):
        """Agent-as-tool with turn_budget gets hooks and max_turns from budget."""
        from sinan_agentic_core.core.turn_budget import TurnBudget

        budget = TurnBudget(default_turns=5, max_extensions=1, extension_size=3, absolute_max=10)
        runner.agent_registry.register(
            AgentDefinition(
                name="budget_sub_agent",
                description="sub with budget",
                instructions="sub",
                as_tool_turn_budget=budget,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        tools = await runner._build_tools(["budget_sub_agent"], ctx)
        assert len(tools) == 1
        assert tools[0].name == "budget_sub_agent"
        # Budget should have been reset
        assert budget.turns_used == 0

    async def test_budget_takes_precedence_over_max_turns(self, runner):
        """When both as_tool_turn_budget and as_tool_max_turns are set, budget wins."""
        from sinan_agentic_core.core.turn_budget import TurnBudget

        budget = TurnBudget(default_turns=5, absolute_max=10)
        runner.agent_registry.register(
            AgentDefinition(
                name="dual_config_agent",
                description="has both",
                instructions="sub",
                as_tool_max_turns=25,
                as_tool_turn_budget=budget,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        tools = await runner._build_tools(["dual_config_agent"], ctx)
        assert len(tools) == 1

    async def test_no_budget_falls_back_to_max_turns(self, runner):
        """Without turn_budget, as_tool_max_turns is used as before."""
        runner.agent_registry.register(
            AgentDefinition(
                name="max_turns_agent",
                description="max turns only",
                instructions="sub",
                as_tool_max_turns=8,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        tools = await runner._build_tools(["max_turns_agent"], ctx)
        assert len(tools) == 1

    async def test_agent_def_as_tool_turn_budget_default_none(self):
        """AgentDefinition.as_tool_turn_budget defaults to None."""
        agent_def = AgentDefinition(
            name="test", description="test", instructions="test"
        )
        assert agent_def.as_tool_turn_budget is None


# ------------------------------------------------------------------ #
# Structured error function
# ------------------------------------------------------------------ #


class TestStructuredToolError:
    def test_returns_json(self):
        from sinan_agentic_core.core.errors import structured_tool_error
        result = structured_tool_error(None, ValueError("page_uuid is required"))
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error_type"] == "ValueError"
        assert "page_uuid is required" in data["message"]
        assert "retry_hint" in data

    def test_max_turns_hint(self):
        from sinan_agentic_core.core.errors import structured_tool_error
        result = structured_tool_error(None, RuntimeError("Max turns exceeded"))
        data = json.loads(result)
        assert "simplify" in data["retry_hint"].lower() or "turns" in data["retry_hint"].lower()

    def test_not_found_hint(self):
        from sinan_agentic_core.core.errors import structured_tool_error
        result = structured_tool_error(None, ValueError("Page not found: abc-123"))
        data = json.loads(result)
        assert "uuid" in data["retry_hint"].lower()

    def test_required_hint(self):
        from sinan_agentic_core.core.errors import structured_tool_error
        result = structured_tool_error(None, ValueError("content is required"))
        data = json.loads(result)
        assert "required" in data["retry_hint"].lower()

    def test_generic_hint(self):
        from sinan_agentic_core.core.errors import structured_tool_error
        result = structured_tool_error(None, RuntimeError("something weird"))
        data = json.loads(result)
        assert "retry" in data["retry_hint"].lower()
