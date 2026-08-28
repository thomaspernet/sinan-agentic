"""Chat functions for API endpoints.

Three flavours of chat, from simplest to most granular:

- ``chat()``            — non-streaming, returns a dict
- ``chat_with_hooks()`` — yields SSE-style dicts with tool-call notifications
- ``chat_streamed()``   — yields token-level deltas via ``Runner.run_streamed()``

All three handle session history, error handling, and return structured
events so your API layer stays thin.  They also apply the run-level SDK
settings the resolved agent needs — an agent whose tools carry tool-input
guardrails runs them ahead of any human-approval interruption, an agent whose
definition declares ``tool_output_trim`` has its bulky older tool outputs
shrunk before each model call, and an agent with a structured ``output_type``
recovers a final message whose payload is valid but wrapped in prose or a
code fence instead of failing the run.

All three also accept ``prompt_cache_key``, which pins the provider's
prompt-cache shard the turn routes to. Turns of one conversation reuse a cached
prompt prefix only when they share a key, and the SDK generates one only for an
official OpenAI client — so a consumer on Azure, or one grouping turns its own
way, names it here.

A failure never reaches the caller as an exception, so each of the three
reports *why* it failed: the caught error is classified into a
``RunErrorKind`` and travels as ``error_kind`` beside the rendered message.
An API layer branches on that kind — retry a ``max_turns``, surface a
``model_refusal`` — instead of matching the message text. A guardrail tripwire
carries a ``guardrail`` entry too, naming the check that stopped the run and
listing every guardrail that finished; all three functions report the same set,
which openai-agents populates uniformly from 0.19.2 on.

Each function accepts either ``agent_name`` (resolved via the registry)
or a pre-built ``agent`` instance.  Use the latter when you need
features that ``create_agent_from_registry`` does not support, such as
dynamic instructions, structured output, or handoffs.

Usage:
    from sinan_agentic_core.services.chat import chat, chat_with_hooks, chat_streamed
    from sinan_agentic_core import AgentSession

    session = AgentSession(session_id="user-123")

    # Simple — agent resolved from registry
    result = await chat("Hello!", agent_name="my_agent", session=session)

    # Streaming tokens — pre-built agent
    async for event in chat_streamed("Hello!", agent=my_agent, session=session, context=ctx):
        print(event)
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from agents import Agent, ItemHelpers, Runner
from openai.types.responses import ResponseTextDeltaEvent

from ..core.model_settings import apply_prompt_cache_key
from ..core.output_recovery import build_error_handlers
from ..core.reasoning import (
    reasoning_event,
    reasoning_stream_event,
    reasoning_summary_texts,
    run_reasoning_texts,
)
from ..core.run_config import build_run_config
from ..core.run_errors import run_error_payload
from ..core.stream_preview import tool_output_preview
from ..core.usage import aggregate_usage
from ..registry.agent_factory import create_agent_from_registry
from ..session import AgentSession
from .hooks import StreamingRunHooks

logger = logging.getLogger(__name__)


def _resolve_agent(
    agent: Agent | None,
    agent_name: str | None,
    model_override: str | None = None,
    prompt_cache_key: str | None = None,
) -> Agent:
    """Return a ready-to-run Agent, either pre-built or from the registry.

    A prompt cache key is applied after the agent is resolved, so it reaches a
    pre-built agent and a registry-resolved one on the same terms. The caller's
    own agent is never mutated — the key lands on a clone.

    Args:
        agent: Pre-built Agent instance (takes priority).
        agent_name: Registry name, used when *agent* is ``None``.
        model_override: Override model (only applies to registry lookup).
        prompt_cache_key: Prompt-cache shard this agent's model calls route to.

    Returns:
        An ``Agent`` instance.

    Raises:
        ValueError: If neither *agent* nor *agent_name* is provided.
    """
    if agent is not None:
        resolved = agent
    elif agent_name is not None:
        resolved = create_agent_from_registry(agent_name, model_override)
    else:
        raise ValueError("Provide either 'agent' (pre-built) or 'agent_name' (registry lookup)")

    if prompt_cache_key is None:
        return resolved
    return resolved.clone(
        model_settings=apply_prompt_cache_key(resolved.model_settings, prompt_cache_key)
    )


async def chat(
    message: str,
    agent_name: str | None = None,
    session: AgentSession | None = None,
    context: Any = None,
    model_override: str | None = None,
    agent: Agent | None = None,
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    """Run a single chat turn (non-streaming).

    Args:
        message: User message text.
        agent_name: Name of a registered agent (ignored when *agent* is set).
        session: ``AgentSession`` that tracks conversation history.
        context: Optional context object passed to agent instructions and tools
            via ``RunContextWrapper``. Can be any dataclass or object.
        model_override: Use a different model than the agent definition.
        agent: Pre-built ``Agent`` instance. When provided, *agent_name* and
            *model_override* are ignored.
        prompt_cache_key: Prompt-cache shard this turn's model calls route to.
            Turns of one conversation reuse the provider's cached prompt prefix
            only when they share a key.

    Returns:
        ``{"success": True, "response": str, "session_id": str, "tools_called": list,
        "usage": dict, "reasoning": list[str]}`` or
        ``{"success": False, "error": str, "error_kind": str, "session_id": str}``
        on failure, where ``error_kind`` is a ``RunErrorKind`` value naming why
        the run failed. A guardrail tripwire adds a ``guardrail`` entry naming
        the check that rejected the run.

        ``reasoning`` carries what the model said about its own thinking, one
        entry per thought. It is empty unless the caller asked for it — a
        reasoning model with ``Reasoning(summary="auto")`` in its
        ``model_settings`` — and empty on a turn the model had nothing to say
        about, which is normal rather than a loss.
    """
    if session is None:
        raise ValueError("'session' is required")
    try:
        resolved = _resolve_agent(agent, agent_name, model_override, prompt_cache_key)

        await session.add_items([{"role": "user", "content": message}])
        history = await session.get_items()

        run_kwargs: dict[str, Any] = {"starting_agent": resolved, "input": history}
        if context is not None:
            run_kwargs["context"] = context

        run_config = build_run_config(resolved)
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        error_handlers = build_error_handlers(resolved)
        if error_handlers is not None:
            run_kwargs["error_handlers"] = error_handlers

        result = await Runner.run(**run_kwargs)
        response = result.final_output

        await session.add_items([{"role": "assistant", "content": response}])

        return {
            "success": True,
            "response": response,
            "session_id": session.session_id,
            "tools_called": [],
            "usage": aggregate_usage(result),
            "reasoning": run_reasoning_texts(result),
        }
    except Exception as e:
        payload = run_error_payload(e)
        logger.error("Chat error (%s): %s", payload["error_kind"], e, exc_info=True)
        return {"success": False, **payload, "session_id": session.session_id}


async def chat_with_hooks(
    message: str,
    agent_name: str | None = None,
    session: AgentSession | None = None,
    context: Any = None,
    tool_friendly_names: dict[str, str] | None = None,
    model_override: str | None = None,
    agent: Agent | None = None,
    prompt_cache_key: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Chat with real-time tool-call notifications via ``RunHooks``.

    Uses ``Runner.run()`` internally — the agent runs to completion, but
    ``StreamingRunHooks`` pushes events to an ``asyncio.Queue`` so you can
    stream progress to the client while it executes.

    Args:
        message: User message text.
        agent_name: Name of a registered agent (ignored when *agent* is set).
        session: ``AgentSession`` for conversation history.
        context: Optional context object passed to agent instructions and tools.
        tool_friendly_names: Optional ``tool_name → display name`` mapping.
        model_override: Use a different model than the agent definition.
        agent: Pre-built ``Agent`` instance.
        prompt_cache_key: Prompt-cache shard this turn's model calls route to.
            Turns of one conversation reuse the provider's cached prompt prefix
            only when they share a key.

    Yields:
        Event dicts with an ``"event"`` key and a ``"data"`` dict::

            {"event": "thinking",   "data": {"message": "..."}}
            {"event": "tool_start", "data": {"tool": "...", ...}}
            {"event": "tool_end",   "data": {"tool": "...", ...}}
            {"event": "finalizing", "data": {"message": "..."}}
            {"event": "reasoning",  "data": {"summary": ["...", "..."]}}
            {"event": "answer",     "data": {"response": "...", "tools_called": [...]}}
            {"event": "error",      "data": {"error": "...", "error_kind": "..."}}

        ``error_kind`` is a ``RunErrorKind`` value naming why the run failed; a
        guardrail tripwire adds a ``guardrail`` entry naming the check that
        rejected the run.

        The ``reasoning`` event carries what the model said about its own
        thinking, one entry per thought. Because this path runs the agent to
        completion rather than streaming its tokens, the whole summary arrives
        at once, just before the answer — ``chat_streamed()`` is the path that
        delivers it as it is written. It is skipped entirely when the caller did
        not ask for a summary, or on a turn the model had nothing to say about.

        The ``answer`` payload owns its ``tools_called`` list: it is copied out
        of the hooks accumulator rather than aliased to it, so the payload is a
        fixed record of this run — the same guarantee ``chat()`` already gives,
        which yields a list it owns.  Only the container is copied; the entries
        are plain tool names.
    """
    if session is None:
        raise ValueError("'session' is required")
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    hooks = StreamingRunHooks(queue, tool_friendly_names)

    try:
        yield {"event": "thinking", "data": {"message": "Analyzing your question..."}}

        resolved = _resolve_agent(agent, agent_name, model_override, prompt_cache_key)
        await session.add_items([{"role": "user", "content": message}])
        history = await session.get_items()

        # Run agent with hooks in the background
        async def _run() -> Any:
            run_kwargs: dict[str, Any] = {
                "starting_agent": resolved,
                "input": history,
                "hooks": hooks,
            }
            if context is not None:
                run_kwargs["context"] = context

            run_config = build_run_config(resolved)
            if run_config is not None:
                run_kwargs["run_config"] = run_config

            error_handlers = build_error_handlers(resolved)
            if error_handlers is not None:
                run_kwargs["error_handlers"] = error_handlers

            return await Runner.run(**run_kwargs)

        task = asyncio.create_task(_run())

        # Forward hook events as they arrive
        while not task.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

        # Drain any remaining events
        while not queue.empty():
            yield await queue.get()

        yield {"event": "finalizing", "data": {"message": "Generating response..."}}

        result = await task
        response = result.final_output
        await session.add_items([{"role": "assistant", "content": response}])

        summary = run_reasoning_texts(result)
        if summary:
            yield reasoning_event(summary)

        yield {
            "event": "answer",
            "data": {
                "response": response,
                "tools_called": list(hooks.tools_called),
                "usage": aggregate_usage(result),
            },
        }
    except Exception as e:
        payload = run_error_payload(e)
        logger.error("Chat hooks error (%s): %s", payload["error_kind"], e, exc_info=True)
        yield {"event": "error", "data": payload}


async def chat_streamed(
    message: str,
    agent_name: str | None = None,
    session: AgentSession | None = None,
    context: Any = None,
    model_override: str | None = None,
    agent: Agent | None = None,
    prompt_cache_key: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Chat with token-level streaming via ``Runner.run_streamed()``.

    Yields events for every text delta, tool invocation, tool output,
    agent handoff, and the final answer.  This is the most granular
    streaming option — ideal for rendering responses character-by-character.

    Args:
        message: User message text.
        agent_name: Name of a registered agent (ignored when *agent* is set).
        session: ``AgentSession`` for conversation history.
        context: Optional context object passed to agent instructions and tools.
        model_override: Use a different model than the agent definition.
        agent: Pre-built ``Agent`` instance. When provided, *agent_name* and
            *model_override* are ignored.
        prompt_cache_key: Prompt-cache shard this turn's model calls route to.
            Turns of one conversation reuse the provider's cached prompt prefix
            only when they share a key.

    Yields:
        Event dicts::

            {"event": "text_delta",      "data": {"delta": "..."}}
            {"event": "reasoning_part_added", "data": {"index": 0}}
            {"event": "reasoning_delta", "data": {"delta": "...", "index": 0}}
            {"event": "reasoning_part_done",  "data": {"index": 0, "text": "..."}}
            {"event": "reasoning",       "data": {"summary": ["...", "..."]}}
            {"event": "tool_call",       "data": {"tool": "...", "message": "..."}}
            {"event": "tool_output",     "data": {"output": "..."}}
            {"event": "message_output",  "data": {"text": "..."}}
            {"event": "agent_updated",   "data": {"agent": "..."}}
            {"event": "answer",          "data": {"response": "...", "tools_called": [...]}}
            {"event": "error",           "data": {"error": "...", "error_kind": "..."}}

        ``error_kind`` is a ``RunErrorKind`` value naming why the run failed; a
        guardrail tripwire adds a ``guardrail`` entry naming the check that
        rejected the run.

        The ``answer`` payload owns its ``tools_called`` list: it is copied out
        of the accumulator this run appends to rather than aliased to it, so the
        payload is a fixed record of this run — the same guarantee
        ``chat_with_hooks()`` gives.  Only the container is copied; the entries
        are plain tool names.

        The reasoning events carry what the model says about its own thinking.
        They appear only when the caller asked for it — a reasoning model with
        ``Reasoning(summary="auto")`` in its ``model_settings`` — and a turn the
        model had nothing to say about yields none, which is normal rather than
        a dropped event. A summary arrives one thought per part: ``index`` names
        the part a delta belongs to, and the ``reasoning_part_added`` /
        ``reasoning_part_done`` pair brackets it, so a consumer rendering a step
        log knows where one thought ends and the next begins. The terminal
        ``reasoning`` event repeats every finished part — the same guarantee
        ``answer`` gives for the response text.
    """
    if session is None:
        raise ValueError("'session' is required")
    try:
        resolved = _resolve_agent(agent, agent_name, model_override, prompt_cache_key)
        await session.add_items([{"role": "user", "content": message}])
        history = await session.get_items()

        run_kwargs: dict[str, Any] = {"starting_agent": resolved, "input": history}
        if context is not None:
            run_kwargs["context"] = context

        run_config = build_run_config(resolved)
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        error_handlers = build_error_handlers(resolved)
        if error_handlers is not None:
            run_kwargs["error_handlers"] = error_handlers

        result = Runner.run_streamed(**run_kwargs)

        tools_called: list[str] = []

        async for event in result.stream_events():
            # Token-level text deltas, and the model's own account of its thinking
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    yield {"event": "text_delta", "data": {"delta": event.data.delta}}
                else:
                    reasoning = reasoning_stream_event(event.data)
                    if reasoning:
                        yield reasoning

            # Higher-level run-item events
            elif event.type == "run_item_stream_event":
                item = event.item

                if item.type == "tool_call_item":
                    raw = getattr(item, "raw_item", None)
                    name = getattr(item, "name", None) or getattr(raw, "name", None) or "unknown"
                    tools_called.append(name)
                    yield {
                        "event": "tool_call",
                        "data": {
                            "tool": name,
                            "message": f"Calling {name.replace('_', ' ')}...",
                        },
                    }
                elif item.type == "tool_call_output_item":
                    yield {
                        "event": "tool_output",
                        "data": {"output": tool_output_preview(item.output)},
                    }
                elif item.type == "message_output_item":
                    yield {
                        "event": "message_output",
                        "data": {"text": ItemHelpers.text_message_output(item)},
                    }
                elif item.type == "reasoning_item":
                    summary = reasoning_summary_texts(item)
                    if summary:
                        yield reasoning_event(summary)

            # Agent handoff
            elif event.type == "agent_updated_stream_event":
                yield {
                    "event": "agent_updated",
                    "data": {"agent": event.new_agent.name},
                }

        # Final answer
        response = result.final_output
        await session.add_items([{"role": "assistant", "content": response}])

        yield {
            "event": "answer",
            "data": {
                "response": response,
                "tools_called": list(tools_called),
                "usage": aggregate_usage(result),
            },
        }
    except Exception as e:
        payload = run_error_payload(e)
        logger.error("Streaming error (%s): %s", payload["error_kind"], e, exc_info=True)
        yield {"event": "error", "data": payload}
