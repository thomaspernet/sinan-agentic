"""Tests for BaseAgentRunner (core/base_runner.py)."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelRetrySettings,
    ModelSettings,
    ToolGuardrailFunctionOutput,
    Usage,
    function_tool,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
)

from sinan_agentic_core.core.base_runner import BaseAgentRunner, _CollectingSessionWrapper
from sinan_agentic_core.core.model_retry import ModelRetryConfig
from sinan_agentic_core.core.output_recovery import recover_invalid_final_output
from sinan_agentic_core.core.tool_output_trim import ToolOutputTrimConfig
from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from sinan_agentic_core.registry.guardrail_registry import (
    GuardrailCategory,
    GuardrailDefinition,
    GuardrailRegistry,
    ResolvedGuardrails,
)
from sinan_agentic_core.registry.tool_registry import ToolDefinition, ToolRegistry
from sinan_agentic_core.session.agent_session import AgentSession
from tests.conftest import make_context_overflow_error


@pytest.fixture
def _registries():
    """Build isolated registries with sample data."""
    agent_reg = AgentRegistry()
    tool_reg = ToolRegistry()
    guardrail_reg = GuardrailRegistry()

    def tool_fn():
        return "result"

    tool_reg.register(
        ToolDefinition(
            name="test_tool",
            function=tool_fn,
            description="desc",
            category="cat",
            parameters_description="p",
            returns_description="r",
        )
    )

    def guardrail_fn():
        return True

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
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        return BaseAgentRunner()


# ------------------------------------------------------------------ #
# Init and setup helpers
# ------------------------------------------------------------------ #


class TestBaseAgentRunnerInit:
    def test_loads_tool_map(self, runner):
        assert runner.tool_map == {"test_tool": runner.tool_registry.get_tool("test_tool").function}

    def test_loads_guardrail_registry(self, runner):
        assert runner.guardrail_registry.get_guardrail("test_guard") is not None

    def test_is_not_abstract(self):
        """BaseAgentRunner should be instantiable directly (not ABC)."""
        with (
            patch(
                "sinan_agentic_core.core.base_runner.get_agent_registry",
                return_value=AgentRegistry(),
            ),
            patch(
                "sinan_agentic_core.core.base_runner.get_tool_registry", return_value=ToolRegistry()
            ),
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
# _build_tools / _build_handoffs registry lookups
# ------------------------------------------------------------------ #


class TestBuildToolsRegistryLookup:
    async def test_registry_tool_is_resolved(self, runner):
        ctx = AgentContext(database_connector=Mock())
        tools = await runner._build_tools(["test_tool"], ctx)
        assert tools == [runner.tool_registry.get_tool("test_tool").function]

    async def test_unknown_name_is_skipped(self, runner):
        ctx = AgentContext(database_connector=Mock())
        assert await runner._build_tools(["missing_tool"], ctx) == []


class TestBuildHandoffs:
    async def test_registered_handoff_is_built(self, runner):
        runner.agent_registry.register(
            AgentDefinition(name="handoff_target", description="target", instructions="target")
        )
        ctx = AgentContext(database_connector=Mock())
        handoffs = await runner._build_handoffs(["handoff_target"], ctx)
        assert [h.name for h in handoffs] == ["handoff_target"]

    async def test_unknown_handoff_is_skipped(self, runner):
        ctx = AgentContext(database_connector=Mock())
        assert await runner._build_handoffs(["missing_agent"], ctx) == []


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
            await runner._execute_streamed("basic_agent", ctx, session, lambda e: None, 30, "hello")
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
            result = await runner.execute("basic_agent", ctx, session, fallback_on_overflow=True)
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
                await runner._execute_with_fallback("basic_agent", ctx, session, 10, "hello", None)

    async def test_refusal_propagates_instead_of_being_rescued(self, runner):
        """A refusal is the model's answer, not an out-of-room failure -- the
        rescue call would re-ask it through a path that bypasses the run."""
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=ModelRefusalError("I can't help."))
            with pytest.raises(ModelRefusalError):
                await runner._execute_with_fallback("basic_agent", ctx, session, 10, "hello", None)

        mock_resolve.assert_not_called()

    async def test_error_quoting_the_old_needle_propagates(self, runner):
        """Regression for #47 -- recovery keys off the exception type, so an
        unrelated error whose text mentions the limit is no longer rescued."""
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(
                side_effect=RuntimeError("tool output mentioned context_length_exceeded")
            )
            with pytest.raises(RuntimeError, match="context_length_exceeded"):
                await runner._execute_with_fallback("basic_agent", ctx, session, 10, "hello", None)

        mock_resolve.assert_not_called()

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
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (10) exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_resolve.return_value = mock_client

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
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (10) exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_resolve.return_value = mock_client

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
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=make_context_overflow_error())
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_resolve.return_value = mock_client

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
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (10) exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_resolve.return_value = mock_client

            result = await runner._execute_with_fallback(
                "basic_agent", ctx, session, 10, "hello", None
            )

        assert result == "Plain text fallback"

    async def test_normal_path_wires_capability_hooks(self, runner, mock_run_result):
        """Capability lifecycle hooks fire on the normal-success fallback path."""
        from sinan_agentic_core.core.base_runner import _CompositeHooks
        from sinan_agentic_core.core.capabilities import Capability

        recorded_tool_starts: list[str] = []

        class _RecorderCapability(Capability):
            def on_tool_start(self, ctx, tool, args):
                recorded_tool_starts.append(getattr(tool, "name", "unknown"))

        recorder = _RecorderCapability()
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        captured_hooks: dict[str, object] = {}

        async def _fake_run(**kwargs):
            captured_hooks["hooks"] = kwargs.get("hooks")
            hooks = kwargs.get("hooks")
            if hooks is not None:
                fake_tool = Mock()
                fake_tool.name = "test_tool"
                fake_ctx = Mock()
                fake_ctx.tool_arguments = ""
                await hooks.on_tool_start(fake_ctx, Mock(), fake_tool)
            return mock_run_result

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=_fake_run)

            result = await runner._execute_with_fallback(
                "basic_agent",
                ctx,
                session,
                10,
                "hello",
                None,
                capabilities=[recorder],
            )

        assert result == "Test response"
        assert isinstance(captured_hooks["hooks"], _CompositeHooks)
        assert recorded_tool_starts == ["test_tool"]

    async def test_recovery_branch_fires_fallback_hooks(self, runner):
        """Recovery branch invokes on_fallback_start / on_fallback_end on capabilities.

        Tool-event hooks must NOT fire on the recovery branch (no tools run).
        """
        from sinan_agentic_core.core.capabilities import Capability

        class _RecorderCapability(Capability):
            def __init__(self):
                self.starts: list[tuple[str, list]] = []
                self.ends: list[tuple[str | None, dict | None]] = []
                self.tool_starts: list[str] = []
                self.tool_ends: list[str] = []

            def on_fallback_start(self, ctx, prompt, collected_items):
                self.starts.append((prompt, list(collected_items)))

            def on_fallback_end(self, ctx, response, usage):
                self.ends.append((response, usage))

            def on_tool_start(self, ctx, tool, args):
                self.tool_starts.append(getattr(tool, "name", "?"))

            def on_tool_end(self, ctx, tool, result):
                self.tool_ends.append(str(result))

        recorder = _RecorderCapability()
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Rescued output"
        mock_completion.usage = Mock(prompt_tokens=100, completion_tokens=20, total_tokens=120)

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (10) exceeded"))
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_resolve.return_value = mock_client

            result = await runner._execute_with_fallback(
                "basic_agent",
                ctx,
                session,
                10,
                "hello",
                None,
                capabilities=[recorder],
            )

        assert result == "Rescued output"
        assert len(recorder.starts) == 1
        prompt, collected_items = recorder.starts[0]
        assert "You are a basic agent" in prompt
        assert isinstance(collected_items, list)
        assert len(recorder.ends) == 1
        response, usage = recorder.ends[0]
        assert response == "Rescued output"
        assert usage is not None
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 20
        # Tool-event hooks must NOT fire on the recovery branch.
        assert recorder.tool_starts == []
        assert recorder.tool_ends == []

    async def test_normal_success_path_does_not_fire_fallback_hooks(self, runner, mock_run_result):
        """When Runner.run succeeds, fallback hooks must not fire."""
        from sinan_agentic_core.core.capabilities import Capability

        class _Recorder(Capability):
            def __init__(self):
                self.start_calls = 0
                self.end_calls = 0

            def on_fallback_start(self, ctx, prompt, collected_items):
                self.start_calls += 1

            def on_fallback_end(self, ctx, response, usage):
                self.end_calls += 1

        recorder = _Recorder()
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)
            await runner._execute_with_fallback(
                "basic_agent",
                ctx,
                session,
                10,
                "hello",
                None,
                capabilities=[recorder],
            )

        assert recorder.start_calls == 0
        assert recorder.end_calls == 0

    async def test_normal_path_omits_hooks_when_no_capabilities(self, runner, mock_run_result):
        """No hooks kwarg is passed when there are no capabilities."""
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)

            await runner._execute_with_fallback(
                "basic_agent", ctx, session, 10, "hello", None, capabilities=[]
            )

            call_kwargs = mock_runner_cls.run.call_args.kwargs
            assert "hooks" not in call_kwargs

    async def test_recovery_branch_reuses_configured_default_client(self, runner):
        """Recovery branch routes the rescue call through the resolved provider
        client (e.g. AsyncAzureOpenAI) instead of constructing a fresh
        AsyncOpenAI. Regression for #35.
        """
        from sinan_agentic_core.core.capabilities import Capability

        class _Recorder(Capability):
            def __init__(self):
                self.start_calls = 0
                self.end_calls = 0

            def on_fallback_start(self, ctx, prompt, collected_items):
                self.start_calls += 1

            def on_fallback_end(self, ctx, response, usage):
                self.end_calls += 1

        recorder = _Recorder()
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="test")

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Rescued via Azure"
        mock_completion.usage = Mock(prompt_tokens=50, completion_tokens=10, total_tokens=60)

        configured_client = AsyncMock(name="AsyncAzureOpenAI")
        configured_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch(
                "sinan_agentic_core.core.base_runner.resolve_openai_client",
                return_value=configured_client,
            ) as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (2) exceeded"))

            result = await runner._execute_with_fallback(
                "basic_agent",
                ctx,
                session,
                2,
                "hello",
                None,
                capabilities=[recorder],
            )

        assert result == "Rescued via Azure"
        configured_client.chat.completions.create.assert_awaited_once()
        # The branch asks the provider layer for its client rather than building one.
        mock_resolve.assert_called_once_with()
        # Capability hooks still fire exactly once each, in order.
        assert recorder.start_calls == 1
        assert recorder.end_calls == 1


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
        prompt = BaseAgentRunner._default_fallback_prompt_builder("Instructions", [], None)
        assert "no tool outputs collected" in prompt

    def test_skips_non_dict_items(self):
        raw_items = ["string_item", 42, None]
        prompt = BaseAgentRunner._default_fallback_prompt_builder("Instructions", raw_items, None)
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
            guardrails=ResolvedGuardrails(),
            handoffs=[],
            output_type=str,
            model_settings=None,
        )
        assert kwargs["name"] == "test"
        assert kwargs["input_guardrails"] == []
        assert kwargs["output_guardrails"] == []
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
            guardrails=ResolvedGuardrails(),
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
        agent_def = AgentDefinition(name="test", description="test", instructions="test")
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
        agent_def = AgentDefinition(name="test", description="test", instructions="test")
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

        result = structured_tool_error(None, MaxTurnsExceeded("Max turns (10) exceeded"))
        data = json.loads(result)
        assert "simplify" in data["retry_hint"].lower() or "turns" in data["retry_hint"].lower()

    def test_context_overflow_hint(self):
        from sinan_agentic_core.core.errors import structured_tool_error

        result = structured_tool_error(None, make_context_overflow_error())
        data = json.loads(result)
        assert "overflow" in data["retry_hint"].lower()

    def test_refusal_hint_tells_the_parent_not_to_re_send(self):
        from sinan_agentic_core.core.errors import structured_tool_error

        result = structured_tool_error(None, ModelRefusalError("I can't help with that."))
        data = json.loads(result)
        assert data["error_type"] == "ModelRefusalError"
        assert "refused" in data["retry_hint"].lower()

    def test_model_behavior_hint(self):
        from sinan_agentic_core.core.errors import structured_tool_error

        result = structured_tool_error(None, ModelBehaviorError("Invalid JSON"))
        data = json.loads(result)
        assert "schema" in data["retry_hint"].lower()

    def test_typed_hint_wins_over_message_text(self):
        """Regression for #47 -- the hint comes from the exception class, so a
        refusal whose text says "not found" still gets the refusal hint."""
        from sinan_agentic_core.core.errors import structured_tool_error

        result = structured_tool_error(None, ModelRefusalError("that record is not found"))
        data = json.loads(result)
        assert "refused" in data["retry_hint"].lower()

    def test_plain_error_quoting_max_turns_gets_the_generic_hint(self):
        """Regression for #47 -- message text alone no longer picks a hint."""
        from sinan_agentic_core.core.errors import structured_tool_error

        result = structured_tool_error(None, RuntimeError("Max turns mentioned in a log line"))
        data = json.loads(result)
        assert data["retry_hint"] == "Review the error message and retry with corrected input."

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


# ------------------------------------------------------------------ #
# Guardrail category wiring
# ------------------------------------------------------------------ #


@pytest.fixture
def _guardrail_registries():
    """Registries holding one guardrail per category and a real function tool."""
    agent_reg = AgentRegistry()
    tool_reg = ToolRegistry()
    guardrail_reg = GuardrailRegistry()

    @function_tool
    def echo(value: str) -> str:
        """Echo a value.

        Args:
            value: Text to echo back.
        """
        return value

    tool_reg.register(ToolDefinition(name="echo", function=echo))

    @input_guardrail
    def guard_input(ctx, agent, agent_input):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    @output_guardrail
    def guard_output(ctx, agent, agent_output):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    @tool_input_guardrail
    def guard_tool_input(data):
        return ToolGuardrailFunctionOutput.allow()

    guardrail_reg.register(GuardrailDefinition("g_in", "d", guard_input, GuardrailCategory.INPUT))
    guardrail_reg.register(
        GuardrailDefinition("g_out", "d", guard_output, GuardrailCategory.OUTPUT)
    )
    guardrail_reg.register(
        GuardrailDefinition("g_tool", "d", guard_tool_input, GuardrailCategory.TOOL_INPUT)
    )

    agent_reg.register(
        AgentDefinition(
            name="guarded_agent",
            description="guarded",
            instructions="You are guarded",
            tools=["echo"],
            guardrails=["g_in", "g_out", "g_tool"],
        )
    )
    agent_reg.register(
        AgentDefinition(
            name="plain_agent",
            description="plain",
            instructions="You are plain",
            tools=["echo"],
            guardrails=["g_in", "g_out"],
        )
    )

    return agent_reg, tool_reg, guardrail_reg, echo


@pytest.fixture
def guardrail_runner(_guardrail_registries):
    agent_reg, tool_reg, guardrail_reg, _ = _guardrail_registries

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        return BaseAgentRunner()


class TestGuardrailCategoryWiring:
    async def test_create_agent_splits_input_and_output_guardrails(self, guardrail_runner):
        agent = await guardrail_runner.create_agent(
            "guarded_agent", context=AgentContext(database_connector=Mock())
        )

        assert [g.get_name() for g in agent.input_guardrails] == ["guard_input"]
        assert [g.get_name() for g in agent.output_guardrails] == ["guard_output"]

    async def test_create_agent_attaches_tool_input_guardrails(self, guardrail_runner):
        agent = await guardrail_runner.create_agent(
            "guarded_agent", context=AgentContext(database_connector=Mock())
        )

        tool = agent.tools[0]
        assert [g.get_name() for g in tool.tool_input_guardrails] == ["guard_tool_input"]

    async def test_registry_tool_is_not_mutated(self, guardrail_runner, _guardrail_registries):
        _, _, _, echo = _guardrail_registries
        await guardrail_runner.create_agent(
            "guarded_agent", context=AgentContext(database_connector=Mock())
        )

        assert echo.tool_input_guardrails is None

    async def test_agent_without_tool_guardrails_keeps_tools_untouched(self, guardrail_runner):
        agent = await guardrail_runner.create_agent(
            "plain_agent", context=AgentContext(database_connector=Mock())
        )

        assert agent.tools[0].tool_input_guardrails is None

    def test_build_run_config_enables_pre_approval(self, guardrail_runner):
        agent_def = guardrail_runner._get_agent_definition("guarded_agent")
        run_config = guardrail_runner._build_run_config(agent_def)

        assert run_config is not None
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    def test_build_run_config_none_without_tool_guardrails(self, guardrail_runner):
        agent_def = guardrail_runner._get_agent_definition("plain_agent")
        assert guardrail_runner._build_run_config(agent_def) is None

    async def test_execute_basic_passes_run_config(self, guardrail_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", context_wrapper=Mock(usage=Usage()))

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await guardrail_runner.execute(
                "guarded_agent", AgentContext(database_connector=Mock()), session
            )

        run_config = mock_run.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_execute_basic_omits_run_config_without_tool_guardrails(self, guardrail_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", context_wrapper=Mock(usage=Usage()))

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await guardrail_runner.execute(
                "plain_agent", AgentContext(database_connector=Mock()), session
            )

        assert "run_config" not in mock_run.call_args.kwargs

    async def test_agent_as_tool_receives_pre_approval_run_config(self, guardrail_runner):
        ctx = AgentContext(database_connector=Mock())

        with patch.object(Agent, "as_tool", return_value=Mock()) as mock_as_tool:
            await guardrail_runner._build_tools(["guarded_agent"], ctx)

        run_config = mock_as_tool.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_agent_as_tool_omits_run_config_without_tool_guardrails(self, guardrail_runner):
        ctx = AgentContext(database_connector=Mock())

        with patch.object(Agent, "as_tool", return_value=Mock()) as mock_as_tool:
            await guardrail_runner._build_tools(["plain_agent"], ctx)

        assert "run_config" not in mock_as_tool.call_args.kwargs

    async def test_run_agent_passes_run_config(self, guardrail_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await guardrail_runner.run_agent(
                "guarded_agent", session, AgentContext(database_connector=Mock())
            )

        run_config = mock_run.call_args.kwargs["run_config"]
        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True

    async def test_run_agent_omits_run_config_without_tool_guardrails(self, guardrail_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await guardrail_runner.run_agent(
                "plain_agent", session, AgentContext(database_connector=Mock())
            )

        assert "run_config" not in mock_run.call_args.kwargs


# ------------------------------------------------------------------ #
# Invalid structured output recovery (error_handlers wiring)
# ------------------------------------------------------------------ #


@pytest.fixture
def recovery_runner(_registries):
    """Runner with one recovering agent and one that opts out."""
    from pydantic import BaseModel

    class Extraction(BaseModel):
        answer: str

    agent_reg, tool_reg, guardrail_reg = _registries
    for name, recovery in (("recovering_agent", True), ("strict_agent", False)):
        agent_reg.register(
            AgentDefinition(
                name=name,
                description="structured",
                instructions="extract data",
                output_dataclass=Extraction,
                invalid_output_recovery=recovery,
            )
        )

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        return BaseAgentRunner()


class TestInvalidOutputRecoveryWiring:
    def test_build_error_handlers_registers_invalid_final_output(self, recovery_runner):
        agent_def = recovery_runner._get_agent_definition("recovering_agent")
        handlers = recovery_runner._build_error_handlers(agent_def)

        assert handlers is not None
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    def test_build_error_handlers_none_when_opted_out(self, recovery_runner):
        agent_def = recovery_runner._get_agent_definition("strict_agent")
        assert recovery_runner._build_error_handlers(agent_def) is None

    async def test_execute_basic_passes_error_handlers(self, recovery_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await recovery_runner.execute(
                "recovering_agent", AgentContext(database_connector=Mock()), session
            )

        handlers = mock_run.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    async def test_execute_basic_omits_error_handlers_when_opted_out(self, recovery_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await recovery_runner.execute(
                "strict_agent", AgentContext(database_connector=Mock()), session
            )

        assert "error_handlers" not in mock_run.call_args.kwargs

    async def test_execute_with_fallback_passes_error_handlers(self, recovery_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await recovery_runner.execute(
                "recovering_agent",
                AgentContext(database_connector=Mock()),
                session,
                fallback_on_overflow=True,
            )

        handlers = mock_run.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    async def test_execute_streamed_passes_error_handlers(self, recovery_runner):
        session = AgentSession(session_id="s1")

        async def stream_events():
            return
            yield

        stream = Mock(final_output="ok", raw_responses=[], stream_events=stream_events)

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run_streamed", return_value=stream
        ) as mock_run:
            await recovery_runner.execute(
                "recovering_agent",
                AgentContext(database_connector=Mock()),
                session,
                streaming=True,
                on_event=Mock(),
            )

        handlers = mock_run.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output

    async def test_run_agent_passes_error_handlers(self, recovery_runner):
        session = AgentSession(session_id="s1")
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await recovery_runner.run_agent(
                "recovering_agent", session, AgentContext(database_connector=Mock())
            )

        handlers = mock_run.call_args.kwargs["error_handlers"]
        assert handlers["invalid_final_output"] is recover_invalid_final_output


class TestFallbackBranchOutputRecovery:
    """The recovery branch bypasses Runner.run, so it salvages directly."""

    async def _run_fallback(self, runner, agent_name, content):
        completion = Mock(usage=None)
        completion.choices = [Mock()]
        completion.choices[0].message.content = content

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client") as mock_resolve,
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (10) exceeded"))
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=completion)
            mock_resolve.return_value = client

            return await runner._execute_with_fallback(
                agent_name,
                AgentContext(database_connector=Mock()),
                AgentSession(session_id="s1"),
                10,
                "hello",
                None,
            )

    async def test_salvages_fenced_fallback_response(self, recovery_runner):
        result = await self._run_fallback(
            recovery_runner, "recovering_agent", '```json\n{"answer": "yes"}\n```'
        )
        assert result.answer == "yes"

    async def test_raises_when_nothing_salvageable(self, recovery_runner):
        with pytest.raises(ModelBehaviorError):
            await self._run_fallback(recovery_runner, "recovering_agent", "not json at all")

    async def test_raises_when_opted_out(self, recovery_runner):
        with pytest.raises(ModelBehaviorError):
            await self._run_fallback(
                recovery_runner, "strict_agent", '```json\n{"answer": "yes"}\n```'
            )


# ------------------------------------------------------------------ #
# Model retry policies (ModelSettings.retry wiring)
# ------------------------------------------------------------------ #


@pytest.fixture
def retry_runner(_registries):
    """Runner with one agent declaring a retry policy and one leaving it off."""
    agent_reg, tool_reg, guardrail_reg = _registries
    agent_reg.register(
        AgentDefinition(
            name="retrying_agent",
            description="retries",
            instructions="answer",
            model_retry=ModelRetryConfig(max_retries=4),
        )
    )
    agent_reg.register(
        AgentDefinition(name="plain_agent", description="plain", instructions="answer")
    )

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        return BaseAgentRunner()


class TestModelRetryWiring:
    def test_build_model_retry_translates_the_declared_config(self, retry_runner):
        agent_def = retry_runner._get_agent_definition("retrying_agent")
        settings = retry_runner._build_model_retry(agent_def)

        assert settings.max_retries == 4
        assert settings.policy is not None

    def test_build_model_retry_none_when_not_declared(self, retry_runner):
        agent_def = retry_runner._get_agent_definition("plain_agent")
        assert retry_runner._build_model_retry(agent_def) is None

    async def test_create_agent_attaches_retry(self, retry_runner, context):
        agent = await retry_runner.create_agent("retrying_agent", context)

        assert agent.model_settings.retry.max_retries == 4

    async def test_create_agent_leaves_retry_unset_when_not_declared(self, retry_runner, context):
        agent = await retry_runner.create_agent("plain_agent", context)

        assert agent.model_settings.retry is None

    async def test_retry_merges_with_computed_model_settings(self, retry_runner, context):
        """A dynamic model_settings_fn keeps its values and gains the retry policy."""
        agent_def = retry_runner._get_agent_definition("retrying_agent")
        agent_def.model_settings_fn = lambda ctx: ModelSettings(temperature=0.2)

        agent = await retry_runner.create_agent("retrying_agent", context)

        assert agent.model_settings.temperature == 0.2
        assert agent.model_settings.retry.max_retries == 4

    async def test_retry_survives_a_model_settings_override(self, retry_runner, context):
        """An override replaces the computed settings but must not drop the policy."""
        agent = await retry_runner.create_agent(
            "retrying_agent", context, model_settings_override=ModelSettings(temperature=0.9)
        )

        assert agent.model_settings.temperature == 0.9
        assert agent.model_settings.retry.max_retries == 4

    async def test_agent_as_tool_sub_agent_carries_the_policy(self, retry_runner, context):
        """Retry rides on model settings, so it reaches the branch error_handlers cannot."""
        with patch.object(Agent, "as_tool", autospec=True, return_value=Mock()) as mock_as_tool:
            await retry_runner._build_tools(["retrying_agent"], context)

        sub_agent = mock_as_tool.call_args.args[0]
        assert sub_agent.model_settings.retry.max_retries == 4

    async def test_explicit_retry_on_an_override_wins(self, retry_runner, context):
        agent = await retry_runner.create_agent(
            "retrying_agent",
            context,
            model_settings_override=ModelSettings(retry=ModelRetrySettings(max_retries=9)),
        )

        assert agent.model_settings.retry.max_retries == 9

    async def _run_fallback(self, runner, agent_name):
        """Drive the SDK-bypassing recovery branch and return its OpenAI client mock."""
        completion = Mock()
        completion.choices = [Mock()]
        completion.choices[0].message.content = "rescued"
        completion.usage = None

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=completion)
        client.with_options = Mock(return_value=client)

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("sinan_agentic_core.core.base_runner.Runner") as mock_runner_cls,
            patch("sinan_agentic_core.core.base_runner.resolve_openai_client", return_value=client),
        ):
            mock_runner_cls.run = AsyncMock(side_effect=MaxTurnsExceeded("Max turns (10) exceeded"))
            await runner._execute_with_fallback(
                agent_name,
                AgentContext(database_connector=Mock()),
                AgentSession(session_id="s1"),
                10,
                "hello",
                None,
            )
        return client

    async def test_fallback_branch_honors_the_declared_attempt_count(self, retry_runner):
        """The branch bypasses Runner.run, so the budget rides the client's own retry."""
        client = await self._run_fallback(retry_runner, "retrying_agent")

        client.with_options.assert_called_once_with(max_retries=4)

    async def test_fallback_branch_leaves_the_client_alone_when_not_declared(self, retry_runner):
        client = await self._run_fallback(retry_runner, "plain_agent")

        client.with_options.assert_not_called()


# ------------------------------------------------------------------ #
# Tool-output trimming (RunConfig.call_model_input_filter wiring)
# ------------------------------------------------------------------ #


@pytest.fixture
def trim_runner(_registries):
    """Runner with agents covering each combination of trimming and tool guardrails."""
    agent_reg, tool_reg, guardrail_reg = _registries
    guardrail_reg.register(
        GuardrailDefinition(
            "guard_tool_input",
            "desc",
            tool_input_guardrail(lambda data: ToolGuardrailFunctionOutput()),
            "tool_input",
        )
    )
    agent_reg.register(
        AgentDefinition(
            name="trimming_agent",
            description="trims",
            instructions="answer",
            tool_output_trim=ToolOutputTrimConfig(recent_turns=3, max_output_chars=4000),
        )
    )
    agent_reg.register(
        AgentDefinition(
            name="trimming_guarded_agent",
            description="trims and guards",
            instructions="answer",
            guardrails=["guard_tool_input"],
            tool_output_trim=ToolOutputTrimConfig(max_output_chars=4000),
        )
    )
    agent_reg.register(
        AgentDefinition(name="plain_agent", description="plain", instructions="answer")
    )

    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg
        ),
    ):
        return BaseAgentRunner()


class TestToolOutputTrimWiring:
    def test_run_config_carries_the_filter(self, trim_runner):
        agent_def = trim_runner._get_agent_definition("trimming_agent")
        run_config = trim_runner._build_run_config(agent_def)

        assert run_config.call_model_input_filter.recent_turns == 3
        assert run_config.call_model_input_filter.max_output_chars == 4000

    def test_run_config_omits_the_filter_when_not_declared(self, trim_runner):
        agent_def = trim_runner._get_agent_definition("plain_agent")
        assert trim_runner._build_run_config(agent_def) is None

    def test_trimming_composes_with_tool_input_pre_approval(self, trim_runner):
        """Both settings share one RunConfig — declaring either must not drop the other."""
        agent_def = trim_runner._get_agent_definition("trimming_guarded_agent")
        run_config = trim_runner._build_run_config(agent_def)

        assert run_config.tool_execution.pre_approval_tool_input_guardrails is True
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_execute_basic_passes_the_filter(self, trim_runner, context):
        result = Mock(final_output="ok", context_wrapper=Mock(usage=Usage()))

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await trim_runner.execute("trimming_agent", context, AgentSession(session_id="s1"))

        run_config = mock_run.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_execute_with_fallback_passes_the_filter(self, trim_runner, context):
        result = Mock(final_output="ok", context_wrapper=Mock(usage=Usage()))

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await trim_runner.execute(
                "trimming_agent",
                context,
                AgentSession(session_id="s1"),
                fallback_on_overflow=True,
            )

        run_config = mock_run.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_execute_streamed_passes_the_filter(self, trim_runner, context):
        stream = Mock(final_output="ok", raw_responses=[])

        async def no_events():
            return
            yield

        stream.stream_events = no_events

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run_streamed", return_value=stream
        ) as mock_run:
            await trim_runner.execute(
                "trimming_agent",
                context,
                AgentSession(session_id="s1"),
                streaming=True,
                on_event=lambda e: None,
            )

        run_config = mock_run.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_run_agent_passes_the_filter(self, trim_runner, context):
        result = Mock(final_output="ok", raw_responses=[])

        with patch(
            "sinan_agentic_core.core.base_runner.Runner.run", new=AsyncMock(return_value=result)
        ) as mock_run:
            await trim_runner.run_agent("trimming_agent", AgentSession(session_id="s1"), context)

        run_config = mock_run.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000

    async def test_agent_as_tool_sub_agent_passes_the_filter(self, trim_runner, context):
        with patch.object(Agent, "as_tool", return_value=Mock()) as mock_as_tool:
            await trim_runner._build_tools(["trimming_agent"], context)

        run_config = mock_as_tool.call_args.kwargs["run_config"]
        assert run_config.call_model_input_filter.max_output_chars == 4000
