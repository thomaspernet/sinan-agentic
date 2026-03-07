"""Base runner for agent execution.

Provides the core execution engine for running agents. Single entry point:
execute() with flags for streaming, fallback_on_overflow, etc.

Also retains run_agent() for backward compatibility.
"""

import json
import logging
from typing import Optional, Dict, Any, List, Callable

from agents import Agent, Runner, RunContextWrapper, ItemHelpers, Usage
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

from ..session import AgentSession
from ..models.context import AgentContext
from ..models import outputs as output_models
from ..registry import get_agent_registry, get_tool_registry, get_guardrail_registry
from .errors import structured_tool_error
from .turn_budget import TurnBudget, TurnBudgetHooks
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

    def __init__(self):
        """Initialize registries and build tool/guardrail mappings."""
        self.agent_registry = get_agent_registry()
        self.tool_registry = get_tool_registry()
        self.guardrail_registry = get_guardrail_registry()

        self.tool_map = {
            name: tool_def.function
            for name, tool_def in self.tool_registry._tools.items()
        }

        self.guardrail_map = {
            name: guardrail_def.function
            for name, guardrail_def in self.guardrail_registry._guardrails.items()
        }

        logger.info(f"Loaded {len(self.tool_map)} tools: {list(self.tool_map.keys())}")
        logger.info(f"Loaded {len(self.guardrail_map)} guardrails: {list(self.guardrail_map.keys())}")

    def setup_context(self, **context_data) -> AgentContext:
        """Setup context with provided data.

        Args:
            **context_data: Arbitrary context data (neo4j_connector, filters, etc.)

        Returns:
            Initialized AgentContext
        """
        return AgentContext(**context_data)

    def setup_session(
        self,
        session_id: Optional[str] = None,
        initial_history: Optional[list] = None,
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

        return AgentSession(session_id=session_id, initial_history=initial_history)

    async def create_agent(
        self,
        agent_name: str,
        context: Any,
    ) -> Agent:
        """Create an agent instance with proper tools and configuration.

        Args:
            agent_name: Name of registered agent to create
            context: Context for dynamic instruction generation

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
        agent_guardrails = self._build_guardrails(agent_def.guardrails)
        handoffs = await self._build_handoffs(agent_def.handoffs, context)
        output_type = self._resolve_output_type(agent_def.output_dataclass)
        model_settings = self._build_model_settings(agent_def, ctx_wrapper)

        agent_kwargs = self._build_agent_kwargs(
            agent_def=agent_def,
            instructions=instructions,
            tools=agent_tools,
            guardrails=agent_guardrails,
            handoffs=handoffs,
            output_type=output_type,
            model_settings=model_settings,
        )

        agent = Agent(**agent_kwargs)

        logger.info(f"Created agent: {agent_name} (model: {agent_def.model})")

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
        on_event: Optional[Callable] = None,
        fallback_on_overflow: bool = False,
        fallback_prompt_builder: Optional[Callable] = None,
        max_turns: int = 10,
        input_text: str = "",
        turn_budget: Optional[TurnBudget] = None,
    ) -> Any:
        """Run an agent and return its final_output directly.

        Single entry point for all agent execution. Three modes controlled by flags:
        - Basic (default): Runner.run() -> final_output
        - Fallback: Runner.run() with overflow recovery -> final_output
        - Streaming: Runner.run_streamed() with event callbacks -> final_output

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
            turn_budget: Optional soft turn budget with self-extension. When provided,
                the SDK's max_turns is set to budget.absolute_max (hard ceiling) and
                the agent perceives budget.effective_max (soft limit) via dynamic
                instructions. The agent can call request_extension() to get more turns.

        Returns:
            Agent's final_output (dataclass, dict, or string)
        """
        # When turn budget is active, override max_turns with absolute ceiling
        sdk_max_turns = turn_budget.absolute_max if turn_budget else max_turns

        if turn_budget:
            turn_budget.reset()
            context._turn_budget = turn_budget

        if streaming:
            return await self._execute_streamed(
                agent_name, context, session, on_event, sdk_max_turns, input_text,
                turn_budget=turn_budget,
            )
        elif fallback_on_overflow:
            return await self._execute_with_fallback(
                agent_name, context, session, sdk_max_turns, input_text,
                fallback_prompt_builder,
            )
        else:
            return await self._execute_basic(
                agent_name, context, session, sdk_max_turns, input_text,
                turn_budget=turn_budget,
            )

    async def _execute_basic(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        max_turns: int,
        input_text: str,
        turn_budget: Optional[TurnBudget] = None,
    ) -> Any:
        """Run agent via Runner.run() and return final_output."""
        agent = await self.create_agent(agent_name=agent_name, context=context)

        if turn_budget:
            agent.tools.append(request_extension_tool)
            self._make_instructions_dynamic(agent, turn_budget)

        logger.info(f"Running agent: {agent_name}")

        run_kwargs: Dict[str, Any] = {
            "starting_agent": agent,
            "input": input_text,
            "session": session,
            "context": context,
            "max_turns": max_turns,
        }
        if turn_budget:
            run_kwargs["hooks"] = TurnBudgetHooks(turn_budget)

        result = await Runner.run(**run_kwargs)

        logger.info(f"Agent '{agent_name}' completed successfully")
        return result.final_output

    async def _execute_with_fallback(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        max_turns: int,
        input_text: str,
        fallback_prompt_builder: Optional[Callable],
    ) -> Any:
        """Run agent with automatic fallback on context overflow.

        If the agent hits max_turns or context_length_exceeded, collects
        all gathered tool outputs and makes a single condensed LLM call.
        """
        agent_def = self._get_agent_definition(agent_name)
        agent = await self.create_agent(agent_name=agent_name, context=context)

        collecting = _CollectingSessionWrapper(session)
        logger.info(f"Running agent with fallback: {agent_name}")

        try:
            run_result = await Runner.run(
                starting_agent=agent,
                input=input_text,
                session=collecting,
                context=context,
                max_turns=max_turns,
            )
            logger.info(f"Agent '{agent_name}' completed successfully")
            return run_result.final_output

        except Exception as err:
            err_str = str(err)
            is_recoverable = (
                "Max turns" in err_str
                or "context_length_exceeded" in err_str
            )
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

            from openai import AsyncOpenAI
            from agents.models._openai_shared import get_default_openai_key

            api_key = get_default_openai_key()
            client = AsyncOpenAI(api_key=api_key)

            output_type = self._resolve_output_type(agent_def.output_dataclass)
            use_json = output_type and output_type != str

            messages: list[Dict[str, str]] = []
            if use_json:
                messages.append({
                    "role": "system",
                    "content": "You must respond with valid JSON.",
                })
                prompt += "\n\nReturn your response as JSON."
            messages.append({"role": "user", "content": prompt})

            kwargs: Dict[str, Any] = {
                "model": agent_def.model or "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.3,
            }
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if not use_json:
                return content

            data = json.loads(content)
            if output_type and output_type != str:
                inner_type = getattr(output_type, "output_type", output_type)
                return inner_type(**data)

            return data

    async def _execute_streamed(
        self,
        agent_name: str,
        context: Any,
        session: AgentSession,
        on_event: Optional[Callable],
        max_turns: int,
        input_text: str,
        turn_budget: Optional[TurnBudget] = None,
    ) -> Any:
        """Run agent with token-level streaming via Runner.run_streamed().

        Adds user message to session, streams events via on_event callback,
        and returns final_output.
        """
        agent = await self.create_agent(agent_name=agent_name, context=context)

        if turn_budget:
            agent.tools.append(request_extension_tool)
            self._make_instructions_dynamic(agent, turn_budget)

        if input_text:
            await session.add_items([{"role": "user", "content": input_text}])
        history = await session.get_items()

        logger.info(f"Running agent (streamed): {agent_name}")

        run_kwargs: Dict[str, Any] = {
            "starting_agent": agent,
            "input": history,
            "context": context,
            "max_turns": max_turns,
        }
        if turn_budget:
            run_kwargs["hooks"] = TurnBudgetHooks(turn_budget)

        result = Runner.run_streamed(**run_kwargs)

        tools_called: List[str] = []

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
                    name = (
                        getattr(item, "name", None)
                        or getattr(raw, "name", None)
                        or "unknown"
                    )
                    tools_called.append(name)
                    if on_event:
                        on_event({
                            "event": "tool_call",
                            "data": {
                                "tool": name,
                                "message": f"Calling {name.replace('_', ' ')}...",
                            },
                        })
                elif item.type == "tool_call_output_item":
                    if on_event:
                        on_event({
                            "event": "tool_output",
                            "data": {"output": str(item.output)[:500]},
                        })
                elif item.type == "message_output_item":
                    if on_event:
                        on_event({
                            "event": "message_output",
                            "data": {"text": ItemHelpers.text_message_output(item)},
                        })

            elif event.type == "agent_updated_stream_event":
                if on_event:
                    on_event({
                        "event": "agent_updated",
                        "data": {"agent": event.new_agent.name},
                    })

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

        if on_event:
            on_event({
                "event": "answer",
                "data": {"response": response, "tools_called": tools_called, "usage": usage},
            })

        logger.info(f"Agent '{agent_name}' (streamed) completed, {len(tools_called)} tool calls")
        return response

    # ------------------------------------------------------------------ #
    # run_agent() — backward-compatible entry point
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_usage_dict(usage: Usage) -> Dict[str, Any]:
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
    def _aggregate_usage(result: Any) -> Dict[str, Any]:
        """Aggregate token usage from a non-streaming RunResult."""
        usage = Usage()
        for response in result.raw_responses:
            usage.add(response.usage)
        return BaseAgentRunner._build_usage_dict(usage)

    async def run_agent(
        self,
        agent_name: str,
        session: AgentSession,
        context: Any,
        input_message: str = "",
    ) -> Dict[str, Any]:
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

    def _get_agent_definition(self, agent_name: str):
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
                f"Agent '{agent_name}' not found in registry. "
                f"Available agents: {available}"
            )
        return agent_def

    def _build_instructions(self, agent_def, ctx_wrapper: RunContextWrapper) -> str:
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
        return instructions

    async def _build_tools(self, tool_names: list, context: Any) -> list:
        """Build agent tools list, handling regular tools and agents-as-tools.

        Args:
            tool_names: List of tool names from agent definition
            context: Context for agent-as-tool creation

        Returns:
            List of configured tool functions
        """
        agent_tools = []

        for tool_name in tool_names:
            if tool_name in self.tool_map:
                agent_tools.append(self.tool_map[tool_name])
            elif tool_name in self.agent_registry._agents:
                tool_agent = await self.create_agent(
                    agent_name=tool_name,
                    context=context,
                )
                agent_def = self.agent_registry._agents[tool_name]
                as_tool_kwargs: Dict[str, Any] = {
                    "tool_name": tool_name,
                    "tool_description": agent_def.description,
                    "failure_error_function": structured_tool_error,
                }
                if agent_def.as_tool_parameters is not None:
                    as_tool_kwargs["parameters"] = agent_def.as_tool_parameters
                if agent_def.as_tool_max_turns is not None:
                    as_tool_kwargs["max_turns"] = agent_def.as_tool_max_turns
                agent_tools.append(tool_agent.as_tool(**as_tool_kwargs))
            else:
                logger.warning(f"Tool '{tool_name}' not found in tool or agent registry")

        return agent_tools

    def _build_hosted_tools(self, hosted_tools: list) -> list:
        """Build hosted tools list (e.g., WebSearchTool, FileSearchTool).

        Hosted tools are OpenAI SDK tools that run on LLM servers alongside
        the AI models. Each entry can be a callable (factory) or a direct
        tool instance.

        Args:
            hosted_tools: List of callables or tool instances

        Returns:
            List of hosted tool instances
        """
        tools = []

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

    def _build_guardrails(self, guardrail_names: list) -> list:
        """Build agent guardrails list.

        Args:
            guardrail_names: List of guardrail names from agent definition

        Returns:
            List of configured guardrail functions
        """
        agent_guardrails = []

        for guardrail_name in guardrail_names:
            if guardrail_name in self.guardrail_map:
                agent_guardrails.append(self.guardrail_map[guardrail_name])
            else:
                logger.warning(f"Guardrail '{guardrail_name}' not found in registry")

        return agent_guardrails

    async def _build_handoffs(self, handoff_names: list, context: Any) -> list:
        """Build agent handoffs list.

        Args:
            handoff_names: List of handoff agent names from agent definition
            context: Context for handoff agent creation

        Returns:
            List of configured handoff agent instances
        """
        handoffs = []

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

    def _resolve_output_type(self, output_dataclass):
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
                return getattr(output_models, output_dataclass)
            except AttributeError:
                logger.warning(f"Output dataclass '{output_dataclass}' not found")
                return str

        return output_dataclass

    def _build_model_settings(self, agent_def, ctx_wrapper: RunContextWrapper):
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
            return agent_def.model_settings_fn(ctx_wrapper)
        except Exception as e:
            logger.error(f"Error building model settings: {e}")
            return None

    def _build_agent_kwargs(
        self,
        agent_def,
        instructions: str,
        tools: list,
        guardrails: list,
        handoffs: list,
        output_type,
        model_settings,
    ) -> dict:
        """Build agent constructor kwargs.

        Args:
            agent_def: Agent definition
            instructions: Processed instructions
            tools: Configured tools list
            guardrails: Configured guardrails list
            handoffs: Configured handoffs list
            output_type: Resolved output type
            model_settings: Model settings or None

        Returns:
            Dictionary of agent constructor arguments
        """
        agent_kwargs = {
            "name": agent_def.name,
            "instructions": instructions,
            "tools": tools,
            "output_guardrails": guardrails if guardrails else [],
            "model": agent_def.model,
            "output_type": output_type,
        }

        if handoffs:
            agent_kwargs["handoffs"] = handoffs

        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings

        return agent_kwargs

    # ------------------------------------------------------------------ #
    # Turn budget helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_instructions_dynamic(agent: Agent, budget: TurnBudget) -> None:
        """Replace static instructions with a callable that appends budget status.

        The SDK evaluates callable instructions before each LLM call, so the
        budget section updates dynamically as turns are consumed.
        """
        base_instructions = agent.instructions

        if callable(base_instructions):
            original_fn = base_instructions

            def dynamic_with_budget(ctx_wrapper, agent_obj):
                base = original_fn(ctx_wrapper, agent_obj)
                section = budget.build_instruction_section()
                return f"{base}\n\n{section}" if section else base

            agent.instructions = dynamic_with_budget
        else:
            static_text = base_instructions or ""

            def dynamic_from_static(ctx_wrapper, agent_obj):
                section = budget.build_instruction_section()
                return f"{static_text}\n\n{section}" if section else static_text

            agent.instructions = dynamic_from_static

    # ------------------------------------------------------------------ #
    # Fallback prompt builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_fallback_prompt_builder(
        instructions: str,
        raw_items: List[Dict[str, Any]],
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
            "\n\n---\n\n".join(tool_outputs) if tool_outputs
            else "(no tool outputs collected)"
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
        self.raw_items: List[Dict[str, Any]] = []

    @property
    def session_id(self) -> str:
        return self._real.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._real.session_id = value

    @property
    def session_settings(self):
        return getattr(self._real, "session_settings", None)

    @session_settings.setter
    def session_settings(self, value) -> None:
        self._real.session_settings = value

    async def get_items(self, limit: Optional[int] = None) -> list:
        return await self._real.get_items(limit)

    async def add_items(self, items: list) -> None:
        self.raw_items.extend(items)
        await self._real.add_items(items)

    async def pop_item(self):
        return await self._real.pop_item()

    async def clear_session(self) -> None:
        self.raw_items.clear()
        await self._real.clear_session()
