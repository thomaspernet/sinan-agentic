"""Base runner for agent execution.

Provides the core execution engine for running agents. Single entry point:
execute() with flags for streaming, fallback_on_overflow, etc.

Also retains run_agent() for backward compatibility.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from agents import (
    Agent,
    ItemHelpers,
    ModelSettings,
    RunContextWrapper,
    RunHooks,
    Runner,
    Usage,
)
from agents.items import TResponseInputItem
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

from ..models import outputs as output_models
from ..models.context import AgentContext
from ..registry import get_agent_registry, get_guardrail_registry, get_tool_registry
from ..registry.agent_registry import AgentDefinition
from ..session import AgentSession, ConversationHistory
from .capabilities import Capability
from .errors import structured_tool_error
from .tool_error_recovery import ToolErrorRecovery
from .turn_budget import TurnBudget
from .turn_budget_tool import request_extension_tool

logger = logging.getLogger(__name__)


class BaseAgentRunner:
    """Agent execution engine with two entry points: execute() and run_agent().

    execute() is the preferred entry point -- returns final_output directly.
    run_agent() is kept for backward compatibility -- returns {"output": ..., "usage": ...}.

    Handles agent creation, tool/guardrail resolution, and three execution modes:
    - Basic: Runner.run() -> returns final_output
    - Fallback: Runner.run() with overflow recovery -> returns final_output
    - Streaming: Runner.run_streamed() with event callbacks -> returns final_output
    """

    def __init__(self) -> None:
        """Initialize registries and build tool/guardrail mappings."""
        self.agent_registry = get_agent_registry()
        self.tool_registry = get_tool_registry()
        self.guardrail_registry = get_guardrail_registry()
        self.last_usage: dict[str, Any] | None = None

        self.tool_map = {
            name: tool_def.function for name, tool_def in self.tool_registry._tools.items()
        }

        self.guardrail_map = {
            name: guardrail_def.function
            for name, guardrail_def in self.guardrail_registry._guardrails.items()
        }

        logger.debug(f"Loaded {len(self.tool_map)} tools: {list(self.tool_map.keys())}")
        logger.debug(
            f"Loaded {len(self.guardrail_map)} guardrails: {list(self.guardrail_map.keys())}"
        )

    def setup_context(self, **context_data: Any) -> AgentContext:
        """Setup context with provided data.

        Args:
            **context_data: Arbitrary context data (neo4j_connector, filters, etc.)

        Returns:
            Initialized AgentContext
        """
        return AgentContext(**context_data)

    def setup_session(
        self,
        session_id: str | None = None,
        initial_history: list[Any] | None = None,
    ) -> AgentSession:
        """Setup session for agent execution.

        Args:
            session_id: Optional session ID for continuity
            initial_history: Optional conversation history

        Returns:
            Initialized AgentSession
        """
        if session_id is None:
            import uuid

            session_id = str(uuid.uuid4())

        history: ConversationHistory | None = None
        if initial_history:
            history = ConversationHistory()
            history.messages = list(initial_history)

        return AgentSession(session_id=session_id, initial_history=history)

    async def create_agent(
        self,
        agent_name: str,
        context: Any,
        model_override: str | None = None,
        model_settings_override: ModelSettings | None = None,
        capabilities: list[Capability] | None = None,
    ) -> Agent:
        """Create an agent instance with proper tools and configuration.

        Args:
            agent_name: Name of registered agent to create
            context: Context for dynamic instruction generation
            model_override: If set, replaces the agent definition's model.
            model_settings_override: If set, replaces the computed model settings.
            capabilities: Cloned per-run capabilities to expose tools from.
                If None, falls back to ``agent_def.capabilities`` (no clone).

        Returns:
            Configured Agent instance

        Raises:
            ValueError: If agent not found in registry
        """
        agent_def = self._get_agent_definition(agent_name)
        ctx_wrapper = RunContextWrapper(context)

        instructions = self._build_instructions(agent_def, ctx_wrapper)
        agent_tools = await self._build_tools(agent_def.tools, context)
        hosted = self._build_hosted_tools(agent_def.hosted_tools)
        agent_tools.extend(hosted)
        for cap in capabilities or agent_def.capabilities:
            agent_tools.extend(cap.tools())
        agent_guardrails = self._build_guardrails(agent_def.guardrails)
        handoffs = await self._build_handoffs(agent_def.handoffs, context)
        output_type = self._resolve_output_type(agent_def.output_dataclass)

        model_settings: ModelSettings | None
        if model_settings_override is not None:
            model_settings = model_settings_override
        else:
            model_settings = self._build_model_settings(agent_def, ctx_wrapper)

        effective_model = model_override or agent_def.model

        agent_kwargs = self._build_agent_kwargs(
            agent_def=agent_def,
            instructions=instructions,
            tools=agent_tools,
            guardrails=agent_guardrails,
            handoffs=handoffs,
            output_type=output_type,
            model_settings=model_settings,
            model_override=model_override,
        )

        agent = Agent(**agent_kwargs)

        settings_info = ""
        if model_settings:
            if model_settings.reasoning:
                settings_info = f", effort={model_settings.reasoning.effort}"
            elif model_settings.temperature is not None:
                settings_info = f", temp={model_settings.temperature}"
        logger.info(f"Created agent: {agent_name} (model: {effective_model}{settings_info})")

        return agent

    # ------------------------------------------------------------------ #
    # execute() — preferred entry point
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        streaming: bool = False,
        on_event: Callable[..., Any] | None = None,
        fallback_on_overflow: bool = False,
        fallback_prompt_builder: Callable[..., Any] | None = None,
        max_turns: int = 10,
        input_text: str = "",
        turn_budget: TurnBudget | None = None,
        error_recovery: ToolErrorRecovery | bool | None = None,
        model_override: str | None = None,
        model_settings_override: ModelSettings | None = None,
    ) -> Any:
        """Run an agent and return its final_output directly.

        Capabilities flow through one pipeline. Three sources are merged:
        1. ``agent_def.capabilities`` (declarative; cloned per run)
        2. ``turn_budget=`` kwarg (runtime; used in place)
        3. ``error_recovery=`` kwarg (runtime; used in place)

        All effective capabilities have ``reset()`` called at the start of
        the run; cloned ones are isolated from their declarative templates.

        Args:
            agent_name: Name of registered agent to run
            context: Context object passed to agent instructions and tools
            session: AgentSession with conversation history
            streaming: Use Runner.run_streamed() with token-level events
            on_event: Callback for streaming events (required when streaming=True)
            fallback_on_overflow: Catch max_turns/context overflow, fallback to
                direct LLM call
            fallback_prompt_builder: Custom function(instructions, raw_items, agent_def)
                -> prompt string. Used in fallback mode to build the condensed LLM
                prompt. If None, uses a default builder that concatenates tool outputs.
            max_turns: Maximum agent turns before stopping
            input_text: Input message for the agent (added to session automatically)
            turn_budget: Optional soft turn budget. Can also be supplied via
                ``AgentDefinition.capabilities``.
            error_recovery: Optional tool error recovery. Pass True to auto-create
                with defaults. Can also be supplied via
                ``AgentDefinition.capabilities``.

        Returns:
            Agent's final_output (dataclass, dict, or string)
        """
        agent_def = self._get_agent_definition(agent_name)

        # Auto-create error recovery if True was passed
        if error_recovery is True:
            error_recovery = ToolErrorRecovery(tool_registry=self.tool_registry)
        elif error_recovery is False:
            error_recovery = None

        capabilities = self._build_run_capabilities(
            agent_def=agent_def,
            turn_budget=turn_budget,
            error_recovery=error_recovery,
            on_event=on_event,
        )

        # Determine SDK max_turns: hardest TurnBudget ceiling wins, else max_turns kwarg.
        sdk_max_turns = max_turns
        for cap in capabilities:
            if isinstance(cap, TurnBudget):
                sdk_max_turns = cap.absolute_max
                # Wire the budget so InstructionBuilder.turn_budget_section
                # and the request_extension tool can locate it.
                context._turn_budget = cap
                break

        if streaming:
            return await self._execute_streamed(
                agent_name,
                context,
                session,
                on_event,
                sdk_max_turns,
                input_text,
                capabilities=capabilities,
                model_override=model_override,
                model_settings_override=model_settings_override,
            )
        elif fallback_on_overflow:
            return await self._execute_with_fallback(
                agent_name,
                context,
                session,
                sdk_max_turns,
                input_text,
                fallback_prompt_builder,
                capabilities=capabilities,
                model_override=model_override,
                model_settings_override=model_settings_override,
            )
        else:
            return await self._execute_basic(
                agent_name,
                context,
                session,
                sdk_max_turns,
                input_text,
                capabilities=capabilities,
                model_override=model_override,
                model_settings_override=model_settings_override,
            )

    async def _execute_basic(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        max_turns: int,
        input_text: str,
        capabilities: list[Capability] | None = None,
        model_override: str | None = None,
        model_settings_override: ModelSettings | None = None,
    ) -> Any:
        """Run agent via Runner.run() and return final_output."""
        capabilities = capabilities or []
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
            model_override=model_override,
            model_settings_override=model_settings_override,
            capabilities=capabilities,
        )

        self._apply_dynamic_instructions(agent, capabilities)

        logger.info(f"Running agent: {agent_name}")

        run_kwargs: dict[str, Any] = {
            "starting_agent": agent,
            "input": input_text,
            "session": session,
            "context": context,
            "max_turns": max_turns,
        }

        hooks = self._build_hooks(capabilities)
        if hooks:
            run_kwargs["hooks"] = hooks

        result = await Runner.run(**run_kwargs)

        self.last_usage = self._aggregate_usage(result)
        logger.info(f"Agent '{agent_name}' completed successfully")
        return result.final_output

    async def _execute_with_fallback(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        max_turns: int,
        input_text: str,
        fallback_prompt_builder: Callable[..., Any] | None,
        capabilities: list[Capability] | None = None,
        model_override: str | None = None,
        model_settings_override: ModelSettings | None = None,
    ) -> Any:
        """Run agent with automatic fallback on context overflow.

        If the agent hits max_turns or context_length_exceeded, collects
        all gathered tool outputs and makes a single condensed LLM call.
        """
        capabilities = capabilities or []
        agent_def = self._get_agent_definition(agent_name)
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
            model_override=model_override,
            model_settings_override=model_settings_override,
            capabilities=capabilities,
        )

        collecting = _CollectingSessionWrapper(session)
        logger.info(f"Running agent with fallback: {agent_name}")

        run_kwargs: dict[str, Any] = {
            "starting_agent": agent,
            "input": input_text,
            "session": collecting,
            "context": context,
            "max_turns": max_turns,
        }

        hooks = self._build_hooks(capabilities)
        if hooks:
            run_kwargs["hooks"] = hooks

        try:
            run_result = await Runner.run(**run_kwargs)
            self.last_usage = self._aggregate_usage(run_result)
            logger.info(f"Agent '{agent_name}' completed successfully")
            return run_result.final_output

        except Exception as err:
            err_str = str(err)
            is_recoverable = "Max turns" in err_str or "context_length_exceeded" in err_str
            if not is_recoverable:
                raise

            logger.warning(
                f"Agent '{agent_name}' hit limit: {err_str}. "
                f"Falling back to summarize-and-extract."
            )

            ctx_wrapper = RunContextWrapper(context)
            instructions = self._build_instructions(agent_def, ctx_wrapper)

            builder = fallback_prompt_builder or self._default_fallback_prompt_builder
            prompt = builder(instructions, collecting.raw_items, agent_def)

            for cap in capabilities:
                cap.on_fallback_start(ctx_wrapper, prompt, collecting.raw_items)

            from agents.models._openai_shared import get_default_openai_client

            client = get_default_openai_client()
            if client is None:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()

            output_type = self._resolve_output_type(agent_def.output_dataclass)
            use_json = output_type and output_type is not str

            # Reuse or build the SDK's AgentOutputSchema so the fallback
            # LLM sees the identical JSON schema (including the "response"
            # wrapper for dataclass output types) and we can parse correctly.
            output_schema = None
            if use_json:
                from agents.agent_output import AgentOutputSchema, AgentOutputSchemaBase

                if isinstance(output_type, AgentOutputSchemaBase):
                    output_schema = output_type
                else:
                    output_schema = AgentOutputSchema(
                        output_type,
                        strict_json_schema=False,
                    )

            messages: list[dict[str, str]] = []
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": model_override or agent_def.model or "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.3,
            }
            if output_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output",
                        "schema": output_schema.json_schema(),
                    },
                }
            elif use_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            # Capture fallback usage from the direct LLM call
            fallback_usage: dict[str, Any] | None = None
            if response.usage:
                fallback_usage = {
                    "requests": 1,
                    "input_tokens": response.usage.prompt_tokens or 0,
                    "output_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                }
                self.last_usage = fallback_usage

            for cap in capabilities:
                cap.on_fallback_end(ctx_wrapper, content, fallback_usage)

            if not use_json:
                return content

            # Use the SDK's schema to parse — handles "response" unwrapping
            if output_schema:
                return output_schema.validate_json(content)

            return json.loads(content)

    async def _execute_streamed(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        on_event: Callable[..., Any] | None,
        max_turns: int,
        input_text: str,
        capabilities: list[Capability] | None = None,
        model_override: str | None = None,
        model_settings_override: ModelSettings | None = None,
    ) -> Any:
        """Run agent with token-level streaming via Runner.run_streamed().

        Adds user message to session, streams events via on_event callback,
        and returns final_output.
        """
        capabilities = capabilities or []
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
            model_override=model_override,
            model_settings_override=model_settings_override,
            capabilities=capabilities,
        )

        self._apply_dynamic_instructions(agent, capabilities)

        if input_text:
            await session.add_items([{"role": "user", "content": input_text}])
        history = await session.get_items()

        logger.info(f"Running agent (streamed): {agent_name}")

        run_kwargs: dict[str, Any] = {
            "starting_agent": agent,
            "input": history,
            "context": context,
            "max_turns": max_turns,
        }

        hooks = self._build_hooks(capabilities)
        if hooks:
            run_kwargs["hooks"] = hooks

        result = Runner.run_streamed(**run_kwargs)

        tools_called: list[str] = []

        total_input_tokens = 0
        total_output_tokens = 0
        last_input_tokens = 0
        request_count = 0

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if on_event and event.data.delta:
                        on_event({"event": "text_delta", "data": {"delta": event.data.delta}})
                elif isinstance(event.data, ResponseCompletedEvent):
                    resp_usage = getattr(event.data.response, "usage", None)
                    if resp_usage:
                        total_input_tokens += resp_usage.input_tokens or 0
                        total_output_tokens += resp_usage.output_tokens or 0
                        last_input_tokens = resp_usage.input_tokens or 0
                        request_count += 1

            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    name = getattr(item, "name", None) or getattr(raw, "name", None) or "unknown"
                    tools_called.append(name)
                    if on_event:
                        on_event(
                            {
                                "event": "tool_call",
                                "data": {
                                    "tool": name,
                                    "message": f"Calling {name.replace('_', ' ')}...",
                                },
                            }
                        )
                elif item.type == "tool_call_output_item":
                    if on_event:
                        on_event(
                            {
                                "event": "tool_output",
                                "data": {"output": str(item.output)[:500]},
                            }
                        )
                elif item.type == "message_output_item":
                    if on_event:
                        on_event(
                            {
                                "event": "message_output",
                                "data": {"text": ItemHelpers.text_message_output(item)},
                            }
                        )

            elif event.type == "agent_updated_stream_event":
                if on_event:
                    on_event(
                        {
                            "event": "agent_updated",
                            "data": {"agent": event.new_agent.name},
                        }
                    )

        response = result.final_output
        await session.add_items([{"role": "assistant", "content": response}])

        stream_usage = Usage(
            requests=request_count,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )
        usage = self._build_usage_dict(stream_usage)
        # Last response's input_tokens = actual context window usage (not sum)
        usage["last_input_tokens"] = last_input_tokens
        self.last_usage = usage

        if on_event:
            on_event(
                {
                    "event": "answer",
                    "data": {"response": response, "tools_called": tools_called, "usage": usage},
                }
            )

        logger.info(f"Agent '{agent_name}' (streamed) completed, {len(tools_called)} tool calls")
        return response

    # ------------------------------------------------------------------ #
    # run_agent() — backward-compatible entry point
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_usage_dict(usage: Usage) -> dict[str, Any]:
        """Convert a Usage object to a plain dict."""
        return {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens_details": {
                "cached_tokens": usage.input_tokens_details.cached_tokens,
            },
            "output_tokens_details": {
                "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
            },
        }

    @staticmethod
    def _aggregate_usage(result: Any) -> dict[str, Any]:
        """Aggregate token usage from a non-streaming RunResult."""
        usage = Usage()
        try:
            raw_responses = getattr(result, "raw_responses", None) or []
            for response in raw_responses:
                resp_usage = getattr(response, "usage", None)
                if resp_usage:
                    usage.add(resp_usage)
        except TypeError:
            pass
        return BaseAgentRunner._build_usage_dict(usage)

    async def run_agent(
        self,
        agent_name: str,
        session: AgentSession,
        context: Any,
        input_message: str = "",
    ) -> dict[str, Any]:
        """Run agent and return structured output with token usage.

        .. deprecated::
            Use execute() for new code. run_agent() returns
            {"output": ..., "usage": ...}; execute() returns
            final_output directly.

        Args:
            agent_name: Name of agent to run
            session: Session with conversation history
            context: Context with required data
            input_message: Optional input message for the run

        Returns:
            Dict with ``output`` (agent's structured output) and ``usage``
            (token usage dict).
        """
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
        )

        logger.info(f"Running agent: {agent_name}")

        result = await Runner.run(
            starting_agent=agent,
            input=input_message,
            session=session,
            context=context,
        )

        logger.info(f"Agent '{agent_name}' completed successfully")

        return {
            "output": result.final_output if hasattr(result, "final_output") else result,
            "usage": self._aggregate_usage(result),
        }

    # ------------------------------------------------------------------ #
    # Private helpers — agent construction
    # ------------------------------------------------------------------ #

    def _get_agent_definition(self, agent_name: str) -> AgentDefinition:
        """Get agent definition from registry with validation.

        Args:
            agent_name: Name of agent to retrieve

        Returns:
            Agent definition from registry

        Raises:
            ValueError: If agent not found in registry
        """
        agent_def = self.agent_registry.get(agent_name)
        if not agent_def:
            available = self.agent_registry.list_all()
            raise ValueError(
                f"Agent '{agent_name}' not found in registry. " f"Available agents: {available}"
            )
        return agent_def

    def _build_instructions(self, agent_def: Any, ctx_wrapper: RunContextWrapper[Any]) -> str:
        """Build agent instructions, handling both static and dynamic.

        Args:
            agent_def: Agent definition with instructions
            ctx_wrapper: Context wrapper for dynamic instruction generation

        Returns:
            Processed instructions string
        """
        instructions = agent_def.instructions
        if callable(instructions):
            instructions = instructions(ctx_wrapper, agent_def)
        return str(instructions)

    async def _build_tools(self, tool_names: list[str], context: Any) -> list[Any]:
        """Build agent tools list, handling regular tools and agents-as-tools.

        Args:
            tool_names: List of tool names from agent definition
            context: Context for agent-as-tool creation

        Returns:
            List of configured tool functions
        """
        agent_tools: list[Any] = []

        for tool_name in tool_names:
            if tool_name in self.tool_map:
                agent_tools.append(self.tool_map[tool_name])
            elif tool_name in self.agent_registry._agents:
                tool_agent = await self.create_agent(
                    agent_name=tool_name,
                    context=context,
                )
                agent_def = self.agent_registry._agents[tool_name]
                as_tool_kwargs: dict[str, Any] = {
                    "tool_name": tool_name,
                    "tool_description": agent_def.description,
                    "failure_error_function": structured_tool_error,
                }
                if agent_def.as_tool_parameters is not None:
                    as_tool_kwargs["parameters"] = agent_def.as_tool_parameters

                budget = agent_def.as_tool_turn_budget
                if budget:
                    budget.reset()
                    sub_caps: list[Capability] = [budget]
                    self._apply_dynamic_instructions(tool_agent, sub_caps)
                    tool_agent.tools.append(request_extension_tool)
                    as_tool_kwargs["hooks"] = _CompositeHooks(sub_caps)
                    as_tool_kwargs["max_turns"] = budget.absolute_max
                elif agent_def.as_tool_max_turns is not None:
                    as_tool_kwargs["max_turns"] = agent_def.as_tool_max_turns

                agent_tools.append(tool_agent.as_tool(**as_tool_kwargs))
            else:
                logger.warning(f"Tool '{tool_name}' not found in tool or agent registry")

        return agent_tools

    def _build_hosted_tools(self, hosted_tools: list[Any]) -> list[Any]:
        """Build hosted tools list (e.g., WebSearchTool, FileSearchTool).

        Hosted tools are OpenAI SDK tools that run on LLM servers alongside
        the AI models. Each entry can be a callable (factory) or a direct
        tool instance.

        Args:
            hosted_tools: List of callables or tool instances

        Returns:
            List of hosted tool instances
        """
        tools: list[Any] = []

        for tool_factory in hosted_tools:
            try:
                if callable(tool_factory):
                    tool = tool_factory()
                    tools.append(tool)
                    logger.info(f"Added hosted tool: {type(tool).__name__}")
                else:
                    tools.append(tool_factory)
                    logger.info(f"Added hosted tool: {type(tool_factory).__name__}")
            except Exception as e:
                logger.error(f"Failed to create hosted tool: {e}")

        return tools

    def _build_guardrails(self, guardrail_names: list[str]) -> list[Any]:
        """Build agent guardrails list.

        Args:
            guardrail_names: List of guardrail names from agent definition

        Returns:
            List of configured guardrail functions
        """
        agent_guardrails: list[Any] = []

        for guardrail_name in guardrail_names:
            if guardrail_name in self.guardrail_map:
                agent_guardrails.append(self.guardrail_map[guardrail_name])
            else:
                logger.warning(f"Guardrail '{guardrail_name}' not found in registry")

        return agent_guardrails

    async def _build_handoffs(self, handoff_names: list[str], context: Any) -> list[Any]:
        """Build agent handoffs list.

        Args:
            handoff_names: List of handoff agent names from agent definition
            context: Context for handoff agent creation

        Returns:
            List of configured handoff agent instances
        """
        handoffs: list[Any] = []

        for handoff_name in handoff_names:
            if handoff_name in self.agent_registry._agents:
                handoff_agent = await self.create_agent(
                    agent_name=handoff_name,
                    context=context,
                )
                handoffs.append(handoff_agent)
            else:
                logger.warning(f"Handoff agent '{handoff_name}' not found in registry")

        return handoffs

    def _resolve_output_type(self, output_dataclass: Any) -> type[Any]:
        """Resolve output type from agent definition.

        Args:
            output_dataclass: Output dataclass specification (string, class, or None)

        Returns:
            Resolved output type class
        """
        if not output_dataclass:
            return str

        if isinstance(output_dataclass, str):
            try:
                resolved: type[Any] = getattr(output_models, output_dataclass)
                return resolved
            except AttributeError:
                logger.warning(f"Output dataclass '{output_dataclass}' not found")
                return str

        cls: type[Any] = output_dataclass
        return cls

    def _build_model_settings(
        self, agent_def: Any, ctx_wrapper: RunContextWrapper[Any]
    ) -> ModelSettings | None:
        """Build model settings if provided.

        Args:
            agent_def: Agent definition with optional model settings function
            ctx_wrapper: Context wrapper for dynamic model settings generation

        Returns:
            Model settings or None
        """
        if not agent_def.model_settings_fn:
            return None

        try:
            settings: ModelSettings | None = agent_def.model_settings_fn(ctx_wrapper)
            return settings
        except Exception as e:
            logger.error(f"Error building model settings: {e}")
            return None

    def _build_agent_kwargs(
        self,
        agent_def: Any,
        instructions: str,
        tools: list[Any],
        guardrails: list[Any],
        handoffs: list[Any],
        output_type: Any,
        model_settings: Any,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Build agent constructor kwargs.

        Args:
            agent_def: Agent definition
            instructions: Processed instructions
            tools: Configured tools list
            guardrails: Configured guardrails list
            handoffs: Configured handoffs list
            output_type: Resolved output type
            model_settings: Model settings or None
            model_override: If set, replaces agent_def.model

        Returns:
            Dictionary of agent constructor arguments
        """
        agent_kwargs: dict[str, Any] = {
            "name": agent_def.name,
            "instructions": instructions,
            "tools": tools,
            "output_guardrails": guardrails if guardrails else [],
            "model": model_override or agent_def.model,
            "output_type": output_type,
        }

        if handoffs:
            agent_kwargs["handoffs"] = handoffs

        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings

        return agent_kwargs

    # ------------------------------------------------------------------ #
    # Capability composition
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_run_capabilities(
        agent_def: Any,
        turn_budget: TurnBudget | None,
        error_recovery: ToolErrorRecovery | None,
        on_event: Callable[..., Any] | None,
    ) -> list[Capability]:
        """Compose the effective capability list for one ``execute()`` call.

        Declarative capabilities (``agent_def.capabilities``) are cloned for
        per-run isolation. Runtime kwargs are used in place so callers can
        inspect their state after the run. Every effective capability is
        reset and gets ``on_event`` wired through.
        """
        effective: list[Capability] = []

        for cap in agent_def.capabilities:
            clone = cap.clone()
            effective.append(clone)

        if turn_budget is not None:
            effective.append(turn_budget)
        if error_recovery is not None:
            effective.append(error_recovery)

        for cap in effective:
            cap.reset()
            cap.on_event = on_event

        return effective

    @staticmethod
    def _apply_dynamic_instructions(
        agent: Agent,
        capabilities: list[Capability],
    ) -> None:
        """Wrap agent.instructions to merge in capability fragments per turn.

        The SDK evaluates callable instructions before each LLM call, so
        capability sections update dynamically as state evolves.
        """
        if not capabilities:
            return

        base_instructions = agent.instructions

        if callable(base_instructions):
            original_fn = base_instructions

            def dynamic_instructions(
                ctx_wrapper: RunContextWrapper[Any], agent_obj: Agent[Any]
            ) -> str:
                base = original_fn(ctx_wrapper, agent_obj)
                if not isinstance(base, str):
                    base = str(base)
                return _merge_capability_instructions(base, ctx_wrapper, capabilities)

            agent.instructions = dynamic_instructions
        else:
            static_text = base_instructions or ""

            def dynamic_from_static(
                ctx_wrapper: RunContextWrapper[Any], agent_obj: Agent[Any]
            ) -> str:
                return _merge_capability_instructions(static_text, ctx_wrapper, capabilities)

            agent.instructions = dynamic_from_static

    @staticmethod
    def _build_hooks(
        capabilities: list[Capability],
    ) -> RunHooks | None:
        """Wrap capabilities in a single RunHooks adapter, or None when empty."""
        if not capabilities:
            return None
        return _CompositeHooks(capabilities)

    # ------------------------------------------------------------------ #
    # Fallback prompt builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_fallback_prompt_builder(
        instructions: str,
        raw_items: list[dict[str, Any]],
        agent_def: Any,
    ) -> str:
        """Default fallback prompt: concatenate instructions + tool outputs.

        Used by _execute_with_fallback() when no custom builder is provided.

        Args:
            instructions: Agent's resolved instructions string
            raw_items: All raw session items captured by _CollectingSessionWrapper
            agent_def: Agent definition (unused in default builder)

        Returns:
            Prompt string for the condensed LLM call
        """
        tool_outputs = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call_output":
                continue
            output = item.get("output", "")
            if output:
                tool_outputs.append(str(output)[:2000])

        context_str = (
            "\n\n---\n\n".join(tool_outputs) if tool_outputs else "(no tool outputs collected)"
        )
        return (
            f"{instructions}\n\n"
            f"## Gathered Context:\n{context_str}\n\n"
            f"Produce your output now based on the context above."
        )


class _CollectingSessionWrapper:
    """Thin wrapper around a Session that captures all raw items.

    AgentSession.add_items() drops items without a ``content`` key
    (e.g. ``function_call_output``).  This wrapper intercepts add_items
    to store every raw item so fallback logic can extract tool outputs.
    """

    def __init__(self, real_session: AgentSession) -> None:
        self._real = real_session
        self.raw_items: list[dict[str, Any]] = []

    @property
    def session_id(self) -> str:
        return self._real.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._real.session_id = value

    @property
    def session_settings(self) -> Any:
        return getattr(self._real, "session_settings", None)

    @session_settings.setter
    def session_settings(self, value: Any) -> None:
        self._real.session_settings = value

    async def get_items(self, limit: int | None = None) -> list[Any]:
        return await self._real.get_items(limit)

    async def add_items(self, items: list[Any]) -> None:
        self.raw_items.extend(items)
        await self._real.add_items(items)

    async def pop_item(self) -> "TResponseInputItem | None":
        return await self._real.pop_item()

    async def clear_session(self) -> None:
        self.raw_items.clear()
        await self._real.clear_session()


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #


def _merge_capability_instructions(
    base: str,
    ctx_wrapper: RunContextWrapper[Any],
    capabilities: list[Capability],
) -> str:
    """Append each capability's current instruction fragment to base."""
    parts = [base]
    for cap in capabilities:
        fragment = cap.instructions(ctx_wrapper)
        if fragment:
            parts.append(fragment)
    return "\n\n".join(parts)


class _CompositeHooks(RunHooks):
    """Adapt a list of capabilities into a single SDK ``RunHooks`` instance.

    Each SDK lifecycle event is dispatched in registration order to every
    capability's matching method. ``on_tool_start`` extracts ``tool_arguments``
    from the SDK's ``ToolContext`` so the abstracted Capability signature
    receives them directly.
    """

    def __init__(self, capabilities: list[Capability]) -> None:
        self._capabilities = capabilities

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        for cap in self._capabilities:
            cap.on_agent_start(context, agent)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        for cap in self._capabilities:
            cap.on_agent_end(context, agent, output)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        for cap in self._capabilities:
            cap.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        args = getattr(context, "tool_arguments", "")
        for cap in self._capabilities:
            cap.on_tool_start(context, tool, args)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        for cap in self._capabilities:
            cap.on_tool_end(context, tool, result)

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        for cap in self._capabilities:
            cap.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        for cap in self._capabilities:
            cap.on_llm_end(context, agent, response)
