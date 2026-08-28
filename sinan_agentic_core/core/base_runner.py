"""Base runner for agent execution.

Provides the core execution engine for running agents. Single entry point:
execute() with flags for streaming, fallback_on_overflow, etc.

Also retains run_agent() for backward compatibility.
"""

import copy
import inspect
import logging
from collections.abc import Callable, Sequence
from typing import Any

from agents import (
    Agent,
    ItemHelpers,
    ModelBehaviorError,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    RunErrorHandlers,
    RunHooks,
    Runner,
)
from agents.items import TResponseInputItem
from openai.types.responses import ResponseTextDeltaEvent

from ..llm import resolve_openai_client
from ..models import outputs as output_models
from ..models.context import AgentContext
from ..registry import get_agent_registry, get_guardrail_registry, get_tool_registry
from ..registry.agent_registry import AgentDefinition
from ..registry.guardrail_registry import (
    GuardrailCategory,
    ResolvedGuardrails,
    attach_tool_input_guardrails,
)
from ..session import AgentSession, ConversationHistory
from ..utils.turn_budget_context import set_turn_budget
from .capabilities import Capability
from .errors import structured_tool_error
from .model_retry import build_model_retry_settings
from .model_settings import (
    PROMPT_CACHE_KEY_FIELD,
    apply_declared_model_settings,
    apply_prompt_cache_key,
)
from .output_recovery import (
    build_output_schema,
    invalid_final_output_handlers,
    salvage_structured_output,
)
from .reasoning import (
    reasoning_event,
    reasoning_stream_event,
    reasoning_summary_texts,
    run_reasoning_texts,
)
from .run_config import build_model_input_filter, tool_input_pre_approval
from .run_errors import FALLBACK_RECOVERABLE_KINDS, classify_run_error
from .stream_preview import tool_output_preview
from .tool_error_recovery import ToolErrorRecovery, build_tool_error_recovery
from .turn_budget import TurnBudget
from .turn_budget_tool import request_extension_tool
from .usage import aggregate_usage, completion_usage_record, last_input_tokens

logger = logging.getLogger(__name__)

# How much of each collected tool output the default fallback prompt carries into
# the condensed rescue call. That call only happens because the run already ran
# out of context or turns, so replaying every gathered output whole would
# reproduce the failure it is rescuing — the head of each one is what the model
# needs to still produce a final answer.
#
# Deliberately not ``stream_preview.TOOL_OUTPUT_PREVIEW_CHARS``: that bound
# shortens what a stream consumer is *shown* and never reaches a model, so it is
# free to be much smaller. This one shapes a model's input. The two move for
# different reasons and are named separately so neither drags the other with it.
# A caller needing a different bound passes its own ``fallback_prompt_builder``.
FALLBACK_PROMPT_TOOL_OUTPUT_CHARS = 2000


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

        # The runner's own record of the last run's token usage. The runner
        # outlives every run it drives, so this record is never handed to a
        # consumer as-is: each hand-out (a streamed answer event, a capability's
        # on_fallback_end) gets its own deep copy. A shallow copy would leave the
        # nested *_tokens_details mappings shared.
        self.last_usage: dict[str, Any] | None = None

        # What the model said about its own thinking on the last run, one entry
        # per summary part. The streamed path delivers this as events while the
        # run is happening; the non-streaming paths hand back only
        # ``final_output``, so this is where their reasoning survives. Rebuilt
        # from scratch by every run, and empty unless the caller asked for a
        # summary via ``model_settings``.
        self.last_reasoning: list[str] = []

        self.tool_map = self.tool_registry.get_all_functions()

        guardrail_names = self.guardrail_registry.list_names()
        logger.debug(f"Loaded {len(self.tool_map)} tools: {list(self.tool_map.keys())}")
        logger.debug(f"Loaded {len(guardrail_names)} guardrails: {guardrail_names}")

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
        prompt_cache_key: str | None = None,
    ) -> Agent:
        """Create an agent instance with proper tools and configuration.

        Args:
            agent_name: Name of registered agent to create
            context: Context for dynamic instruction generation
            model_override: If set, replaces the agent definition's model.
            model_settings_override: If set, replaces the computed model settings.
            capabilities: Cloned per-run capabilities to expose tools from.
                If None, falls back to ``agent_def.capabilities`` (no clone).
            prompt_cache_key: Prompt-cache shard this agent's model calls route
                to. Left off, the provider shards by whatever the SDK generates.

        Returns:
            Configured Agent instance

        Raises:
            ValueError: If agent not found in registry
        """
        agent_def = self._get_agent_definition(agent_name)
        ctx_wrapper = RunContextWrapper(context)

        instructions = await self._build_instructions(agent_def, ctx_wrapper)
        agent_tools = await self._build_tools(agent_def.tools, context, prompt_cache_key)
        hosted = self._build_hosted_tools(agent_def.hosted_tools)
        agent_tools.extend(hosted)
        for cap in capabilities or agent_def.capabilities:
            agent_tools.extend(cap.tools())
        agent_guardrails = self.guardrail_registry.resolve(agent_def.guardrails)
        agent_tools = attach_tool_input_guardrails(
            agent_tools, agent_guardrails.tool_input_guardrails
        )
        handoffs = await self._build_handoffs(agent_def.handoffs, context, prompt_cache_key)
        output_type = self._resolve_output_type(agent_def.output_dataclass)

        model_settings: ModelSettings | None
        if model_settings_override is not None:
            model_settings = model_settings_override
        else:
            model_settings = self._build_model_settings(agent_def, ctx_wrapper)

        model_settings = apply_declared_model_settings(
            model_settings,
            model_retry=agent_def.model_retry,
            model_timeout=agent_def.model_timeout,
        )
        model_settings = apply_prompt_cache_key(model_settings, prompt_cache_key)

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
        prompt_cache_key: str | None = None,
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
            prompt_cache_key: Prompt-cache shard every model call of this run
                routes to, the rescue call the overflow fallback makes included.
                Sibling runs that share a leading prompt span reuse the
                provider's cached prefix only when they share a key. Left off,
                the provider shards by whatever the SDK generates — which is
                nothing at all for an Azure client.

        Returns:
            Agent's final_output (dataclass, dict, or string)
        """
        agent_def = self._get_agent_definition(agent_name)

        # A boolean is a declaration, not a capability — translate it the way the
        # YAML paths do rather than constructing one here.
        if isinstance(error_recovery, bool):
            error_recovery = build_tool_error_recovery(error_recovery, self.tool_registry)

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
                set_turn_budget(context, cap)
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
                prompt_cache_key=prompt_cache_key,
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
                prompt_cache_key=prompt_cache_key,
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
                prompt_cache_key=prompt_cache_key,
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
        prompt_cache_key: str | None = None,
    ) -> Any:
        """Run agent via Runner.run() and return final_output."""
        capabilities = capabilities or []
        agent_def = self._get_agent_definition(agent_name)
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
            model_override=model_override,
            model_settings_override=model_settings_override,
            capabilities=capabilities,
            prompt_cache_key=prompt_cache_key,
        )

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

        run_config = self._build_run_config(agent_def, capabilities)
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        error_handlers = self._build_error_handlers(agent_def)
        if error_handlers is not None:
            run_kwargs["error_handlers"] = error_handlers

        result = await Runner.run(**run_kwargs)

        self.last_usage = aggregate_usage(result)
        self.last_reasoning = run_reasoning_texts(result)
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
        prompt_cache_key: str | None = None,
    ) -> Any:
        """Run agent with automatic fallback on context overflow.

        When the run fails with a kind in ``FALLBACK_RECOVERABLE_KINDS`` --
        ``MAX_TURNS`` or ``CONTEXT_OVERFLOW`` -- collects all gathered tool
        outputs and makes a single condensed LLM call. Every other failure,
        a refusal included, propagates untouched.

        The prompt builder and each capability's ``on_fallback_start`` own the
        collected items they receive, the same way ``on_fallback_end`` owns its
        usage record: every hand-out is a deep copy of the collector's list, so
        an edit through one cannot reach the collector or another hand-out.
        """
        capabilities = capabilities or []
        agent_def = self._get_agent_definition(agent_name)
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
            model_override=model_override,
            model_settings_override=model_settings_override,
            capabilities=capabilities,
            prompt_cache_key=prompt_cache_key,
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

        run_config = self._build_run_config(agent_def, capabilities)
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        error_handlers = self._build_error_handlers(agent_def)
        if error_handlers is not None:
            run_kwargs["error_handlers"] = error_handlers

        try:
            run_result = await Runner.run(**run_kwargs)
            self.last_usage = aggregate_usage(run_result)
            self.last_reasoning = run_reasoning_texts(run_result)
            logger.info(f"Agent '{agent_name}' completed successfully")
            return run_result.final_output

        except Exception as err:
            kind = classify_run_error(err)
            if kind not in FALLBACK_RECOVERABLE_KINDS:
                raise

            logger.warning(
                "Agent '%s' hit %s: %s. Falling back to summarize-and-extract.",
                agent_name,
                kind.value,
                err,
            )

            ctx_wrapper = RunContextWrapper(context)
            instructions = await self._build_instructions(agent_def, ctx_wrapper)

            # NOTE: a declared tool_output_trim does not reach this prompt. The
            # SDK applies the filter through RunConfig.call_model_input_filter,
            # read only inside Runner.run (agents.run_internal.turn_preparation,
            # openai-agents==0.21.1), which this branch bypasses. The filter also
            # keys its window off the last N *user* messages, and a rescue prompt
            # replays a single one — so running it here would trim nothing. The
            # prompt builder caps each output instead.
            builder = fallback_prompt_builder or self._default_fallback_prompt_builder
            prompt = builder(instructions, copy.deepcopy(collecting.raw_items), agent_def)

            for cap in capabilities:
                cap.on_fallback_start(ctx_wrapper, prompt, copy.deepcopy(collecting.raw_items))

            client = resolve_openai_client()

            # NOTE: retry policies and backoff are runner-managed — the SDK reads
            # them off the resolved ModelSettings inside Runner.run
            # (agents.run_internal.model_retry, openai-agents==0.21.1), which this
            # branch bypasses. Only the declared attempt count carries over, via
            # the OpenAI client's own retry, so a rescue call is not left on the
            # client default while every other branch honors the agent's budget.
            # The declared per-attempt timeout carries over the same way: the SDK
            # enforces it in the module above, but the client's own per-request
            # timeout bounds this call to the same number of seconds, so the
            # rescue cannot be the one model call left free to hang.
            client_options: dict[str, Any] = {}
            retry_settings = build_model_retry_settings(agent_def.model_retry)
            if retry_settings is not None and retry_settings.max_retries is not None:
                client_options["max_retries"] = retry_settings.max_retries
            if agent_def.model_timeout is not None:
                client_options["timeout"] = agent_def.model_timeout
            if client_options:
                client = client.with_options(**client_options)

            # Reuse or build the SDK's AgentOutputSchema so the fallback
            # LLM sees the identical JSON schema (including the "response"
            # wrapper for dataclass output types) and we can parse correctly.
            # Non-strict: the schema is sent to the provider as a
            # response_format, which rejects strict-mode constructs.
            output_schema = build_output_schema(
                self._resolve_output_type(agent_def.output_dataclass),
                strict_json_schema=False,
            )

            messages: list[dict[str, str]] = []
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": model_override or agent_def.model or "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.3,
            }
            # NOTE: the key reaches every other branch through
            # ModelSettings.extra_args, which the SDK's model adapters splat into
            # the request. This branch bypasses those adapters, so it names the
            # provider parameter directly — chat.completions.create accepts
            # prompt_cache_key (openai 3.3.1) — rather than leaving the rescue as
            # the one model call of the run routed to a different cache shard.
            if prompt_cache_key is not None:
                kwargs[PROMPT_CACHE_KEY_FIELD] = prompt_cache_key
            if output_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output",
                        "schema": output_schema.json_schema(),
                    },
                }

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            # Capture fallback usage from the direct LLM call
            fallback_usage: dict[str, Any] | None = None
            if response.usage:
                fallback_usage = completion_usage_record(response.usage)
                self.last_usage = fallback_usage

            for cap in capabilities:
                cap.on_fallback_end(ctx_wrapper, content, copy.deepcopy(fallback_usage))

            if output_schema is None:
                return content

            # Use the SDK's schema to parse — handles "response" unwrapping.
            # This branch bypasses Runner.run, so the SDK's invalid_final_output
            # handler cannot reach it; apply the same salvage directly.
            try:
                return output_schema.validate_json(content or "")
            except ModelBehaviorError:
                if not agent_def.invalid_output_recovery:
                    raise
                salvaged = salvage_structured_output(content, output_schema)
                if salvaged is None:
                    raise
                logger.info(
                    "Recovered structured output for agent '%s' from a malformed "
                    "fallback response",
                    agent_name,
                )
                return salvaged

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
        prompt_cache_key: str | None = None,
    ) -> Any:
        """Run agent with token-level streaming via Runner.run_streamed().

        Adds user message to session, streams events via on_event callback,
        and returns final_output.

        Events delivered to ``on_event``::

            {"event": "text_delta",         "data": {"delta": "..."}}
            {"event": "reasoning_part_added", "data": {"index": 0}}
            {"event": "reasoning_delta",    "data": {"delta": "...", "index": 0}}
            {"event": "reasoning_part_done", "data": {"index": 0, "text": "..."}}
            {"event": "reasoning",          "data": {"summary": ["...", "..."]}}
            {"event": "tool_call",          "data": {"tool": "...", "message": "..."}}
            {"event": "tool_output",        "data": {"output": "..."}}
            {"event": "message_output",     "data": {"text": "..."}}
            {"event": "agent_updated",      "data": {"agent": "..."}}
            {"event": "answer",             "data": {...}}

        The reasoning events carry what the model says about its own thinking,
        and only appear when the caller asked for it — ``model_settings`` reaches
        the agent untouched, so ``Reasoning(summary="auto")`` on a reasoning
        model is what turns them on. A summary arrives in parts, one thought per
        part: ``index`` is the part a delta belongs to, and the
        ``reasoning_part_added`` / ``reasoning_part_done`` pair brackets it, so a
        consumer rendering a step log knows where one thought ends and the next
        begins. The terminal ``reasoning`` event repeats every finished part —
        the same guarantee ``answer`` gives for the response text, so a consumer
        that dropped a fragment still ends up with exactly what was said.

        The ``answer`` event owns its ``usage`` payload: it is a deep copy of the
        record kept as ``self.last_usage``, so editing it does not rewrite the
        runner's own record and a later reader of ``last_usage`` does not see
        whatever a consumer wrote into a delivered event. It owns its
        ``tools_called`` list the same way — copied out of the accumulator this
        run appends to, so the event stays a fixed record of what ran. Only the
        container is copied; the entries are plain tool names.
        """
        capabilities = capabilities or []
        agent_def = self._get_agent_definition(agent_name)
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
            model_override=model_override,
            model_settings_override=model_settings_override,
            capabilities=capabilities,
            prompt_cache_key=prompt_cache_key,
        )

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

        run_config = self._build_run_config(agent_def, capabilities)
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        error_handlers = self._build_error_handlers(agent_def)
        if error_handlers is not None:
            run_kwargs["error_handlers"] = error_handlers

        result = Runner.run_streamed(**run_kwargs)

        tools_called: list[str] = []
        reasoning_said: list[str] = []

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if on_event and event.data.delta:
                        on_event({"event": "text_delta", "data": {"delta": event.data.delta}})
                else:
                    reasoning = reasoning_stream_event(event.data)
                    if on_event and reasoning:
                        on_event(reasoning)

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
                                "data": {"output": tool_output_preview(item.output)},
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
                elif item.type == "reasoning_item":
                    summary = reasoning_summary_texts(item)
                    reasoning_said.extend(summary)
                    if on_event and summary:
                        on_event(reasoning_event(summary))

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

        usage = aggregate_usage(result)
        # Last response's input_tokens = actual context window usage (not sum)
        usage["last_input_tokens"] = last_input_tokens(result)
        self.last_usage = usage
        self.last_reasoning = reasoning_said

        if on_event:
            on_event(
                {
                    "event": "answer",
                    "data": {
                        "response": response,
                        "tools_called": list(tools_called),
                        "usage": copy.deepcopy(usage),
                    },
                }
            )

        logger.info(f"Agent '{agent_name}' (streamed) completed, {len(tools_called)} tool calls")
        return response

    # ------------------------------------------------------------------ #
    # run_agent() — backward-compatible entry point
    # ------------------------------------------------------------------ #

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
        agent_def = self._get_agent_definition(agent_name)
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context,
        )

        logger.info(f"Running agent: {agent_name}")

        run_kwargs: dict[str, Any] = {
            "starting_agent": agent,
            "input": input_message,
            "session": session,
            "context": context,
        }

        run_config = self._build_run_config(agent_def)
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        error_handlers = self._build_error_handlers(agent_def)
        if error_handlers is not None:
            run_kwargs["error_handlers"] = error_handlers

        result = await Runner.run(**run_kwargs)

        logger.info(f"Agent '{agent_name}' completed successfully")

        return {
            "output": result.final_output if hasattr(result, "final_output") else result,
            "usage": aggregate_usage(result),
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

    async def _build_instructions(self, agent_def: Any, ctx_wrapper: RunContextWrapper[Any]) -> str:
        """Build agent instructions, handling both static and dynamic.

        Dynamic instructions follow the SDK's await-by-result rule: the callable
        is invoked once and its result awaited when awaitable, so an ``async def``
        builder — or a builder object with an async ``__call__`` — resolves to its
        text instead of leaking a coroutine repr into the system prompt.

        Args:
            agent_def: Agent definition with instructions
            ctx_wrapper: Context wrapper for dynamic instruction generation

        Returns:
            Processed instructions string
        """
        instructions = agent_def.instructions
        if callable(instructions):
            result = instructions(ctx_wrapper, agent_def)
            instructions = await result if inspect.isawaitable(result) else result
        return str(instructions)

    async def _build_tools(
        self, tool_names: list[str], context: Any, prompt_cache_key: str | None = None
    ) -> list[Any]:
        """Build agent tools list, handling regular tools and agents-as-tools.

        Args:
            tool_names: List of tool names from agent definition
            context: Context for agent-as-tool creation
            prompt_cache_key: Prompt-cache shard the run routes to. A sub-agent
                invoked as a tool is part of the same run, so it routes to the
                same shard.

        Returns:
            List of configured tool functions
        """
        agent_tools: list[Any] = []

        for tool_name in tool_names:
            if tool_name in self.tool_map:
                agent_tools.append(self.tool_map[tool_name])
                continue

            agent_def = self.agent_registry.get(tool_name)
            if agent_def is None:
                logger.warning(f"Tool '{tool_name}' not found in tool or agent registry")
                continue

            tool_agent = await self.create_agent(
                agent_name=tool_name,
                context=context,
                prompt_cache_key=prompt_cache_key,
            )
            as_tool_kwargs: dict[str, Any] = {
                "tool_name": tool_name,
                "tool_description": agent_def.description,
                "failure_error_function": structured_tool_error,
            }
            if agent_def.as_tool_parameters is not None:
                as_tool_kwargs["parameters"] = agent_def.as_tool_parameters

            # The sub-agent's capabilities steer its model calls through its run
            # config, so they are assembled before the config that carries them.
            sub_caps: list[Capability] = []
            budget = agent_def.as_tool_turn_budget
            if budget:
                budget.reset()
                sub_caps.append(budget)
                tool_agent.tools.append(request_extension_tool)
                as_tool_kwargs["hooks"] = _CompositeHooks(sub_caps)
                as_tool_kwargs["max_turns"] = budget.absolute_max
            elif agent_def.as_tool_max_turns is not None:
                as_tool_kwargs["max_turns"] = agent_def.as_tool_max_turns

            sub_run_config = self._build_run_config(agent_def, sub_caps)
            if sub_run_config is not None:
                as_tool_kwargs["run_config"] = sub_run_config

            # NOTE: no error_handlers here — Agent.as_tool() accepts
            # run_config but has no error_handlers parameter, so a
            # sub-agent's invalid structured output still raises. Wire it
            # the moment the SDK exposes it.

            agent_tools.append(tool_agent.as_tool(**as_tool_kwargs))

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

    def _build_run_config(
        self,
        agent_def: Any,
        capabilities: Sequence[Capability] = (),
    ) -> RunConfig | None:
        """Build the SDK run config for an agent, or None when defaults apply.

        Declaring a tool-input guardrail opts the agent into running those
        guardrails *before* the SDK emits a pending-approval interruption, so a
        rejected call never reaches the tool or a human approver.

        Declaring ``tool_output_trim`` installs the SDK's tool-output filter,
        which shrinks oversized outputs from older turns before each model call.

        Running with capabilities installs steering, which appends their
        instruction fragments to the input of each model call. Both want the one
        ``call_model_input_filter`` slot, so :func:`build_model_input_filter`
        decides what it holds rather than this method choosing between them.

        Args:
            agent_def: Agent definition whose guardrails and trim policy decide
                the config
            capabilities: The run's capabilities, already cloned and reset.
                Empty — as on ``run_agent()``, which runs none — steers nothing.

        Returns:
            Configured RunConfig, or None when no setting differs from the default
        """
        config_kwargs: dict[str, Any] = {}

        if self.guardrail_registry.has_category(agent_def.guardrails, GuardrailCategory.TOOL_INPUT):
            config_kwargs["tool_execution"] = tool_input_pre_approval()

        model_input_filter = build_model_input_filter(agent_def.tool_output_trim, capabilities)
        if model_input_filter is not None:
            config_kwargs["call_model_input_filter"] = model_input_filter

        if not config_kwargs:
            return None

        return RunConfig(**config_kwargs)

    def _build_error_handlers(self, agent_def: Any) -> RunErrorHandlers[Any] | None:
        """Build the SDK run error handlers for an agent, or None when defaults apply.

        Agents keep ``invalid_output_recovery`` on by default: a final message
        whose payload is valid but wrapped in prose or a code fence is
        re-parsed instead of failing the run. Recovery only ever returns a
        payload the agent's own output schema accepts, so an agent that must
        fail loudly on any malformed output sets the flag to ``False``.

        Args:
            agent_def: Agent definition whose recovery flag decides the handlers

        Returns:
            Configured handlers, or None when the agent opts out
        """
        if not agent_def.invalid_output_recovery:
            return None

        return invalid_final_output_handlers()

    async def _build_handoffs(
        self, handoff_names: list[str], context: Any, prompt_cache_key: str | None = None
    ) -> list[Any]:
        """Build agent handoffs list.

        Args:
            handoff_names: List of handoff agent names from agent definition
            context: Context for handoff agent creation
            prompt_cache_key: Prompt-cache shard the run routes to. A handed-off
                agent continues the same run, so it routes to the same shard.

        Returns:
            List of configured handoff agent instances
        """
        handoffs: list[Any] = []

        for handoff_name in handoff_names:
            if self.agent_registry.get(handoff_name) is None:
                logger.warning(f"Handoff agent '{handoff_name}' not found in registry")
                continue

            handoff_agent = await self.create_agent(
                agent_name=handoff_name,
                context=context,
                prompt_cache_key=prompt_cache_key,
            )
            handoffs.append(handoff_agent)

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
        guardrails: ResolvedGuardrails,
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
            guardrails: Guardrails bucketed by category. Tool-input guardrails
                are already attached to *tools*, so only the run-level slots
                are wired here.
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
            "input_guardrails": guardrails.input_guardrails,
            "output_guardrails": guardrails.output_guardrails,
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
        Each output is cut to :data:`FALLBACK_PROMPT_TOOL_OUTPUT_CHARS` so a
        prompt built from an overflowed run does not overflow in turn.

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
                tool_outputs.append(str(output)[:FALLBACK_PROMPT_TOOL_OUTPUT_CHARS])

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

        # The collector's own record of everything the failed loop gathered. It
        # outlives each read, so it is never handed to a caller as-is: each
        # hand-out on the recovery branch (the fallback prompt builder, a
        # capability's on_fallback_start) gets its own deep copy. A shallow copy
        # would leave the item dicts — and the values nested inside them —
        # shared.
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


class _CompositeHooks(RunHooks):
    """Adapt a list of capabilities into a single SDK ``RunHooks`` instance.

    Each SDK lifecycle event is dispatched in registration order to every
    capability's matching method. ``on_tool_start`` extracts ``tool_arguments``
    from the SDK's ``ToolContext`` so the abstracted Capability signature
    receives them directly.
    """

    def __init__(self, capabilities: list[Capability]) -> None:
        """
        Args:
            capabilities: Capabilities to dispatch every lifecycle event to, in
                registration order. The list is copied, so a caller that keeps
                its own list and edits it after construction cannot change what
                a live bundle dispatches to, and no two bundles built from one
                list share a dispatch set. The capabilities inside it are
                deliberately not copied: a turn budget's remaining turns and a
                recovery tracker's error state are live per-run state the caller
                reads back after the run, so the elements stay shared and only
                the container is detached.
        """
        self._capabilities = list(capabilities)

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
