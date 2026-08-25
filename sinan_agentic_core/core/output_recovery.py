"""Recovery for malformed structured output.

When an agent declares an ``output_type`` and the model's final message does
not validate against it, the SDK raises ``ModelBehaviorError`` and the run is
lost. The SDK's ``invalid_final_output`` run error handler is the extension
point for recovering instead: it receives the failed run data and may return a
replacement final output, which the SDK re-validates against the same schema
before it flows through the normal final-output hooks, guardrails, streaming
events, and session persistence. Returning ``None`` preserves the raise.

:func:`recover_invalid_final_output` is this framework's handler. It never
fabricates a value — it only re-parses text the model already produced. The
common failure is a well-formed payload the SDK cannot parse because the model
wrapped it: a ``` fence, a "Here is the JSON:" preamble, trailing commentary.
``AgentOutputSchema.validate_json`` hands the whole message to Pydantic
verbatim, so those runs raise today even though the payload is valid. This
module locates the embedded payload and validates it against the agent's own
schema. Anything that does not validate is not recovered.

``ModelBehaviorError`` reaches a caller by two routes and this handler sits on
only one of them. The malformed-final-output route runs the whole turn first,
so the failed run data exists and salvage has something to re-parse. The other
route is a Responses turn the provider ended in a terminal ``failed`` /
``incomplete`` state: openai-agents raises that from inside ``get_response``,
before any final output is validated, so no ``invalid_final_output`` handler is
consulted and the error propagates to the caller as-is. Both arrive classified
as ``RunErrorKind.MODEL_BEHAVIOR``; only the first is recoverable here.

The scan that locates the payload, :func:`iter_payload_candidates`, is the
shared half: anything that has to read a payload out of a model message —
recovery here, session persistence in ``session/agent_session.py`` — reads it
the same way rather than handing the whole message to a parser.

Handlers belong to the *run*, not the agent, so building an agent is not enough
to get recovery: they are passed to ``Runner.run()`` / ``Runner.run_streamed()``.
``BaseAgentRunner`` decides from the agent *definition*, whose
``invalid_output_recovery`` flag it has on every execution branch. Call sites
that only hold a built ``Agent`` — the chat service, and consumers driving
``Runner`` themselves — call :func:`build_error_handlers` instead, so a
pre-built agent gets recovery on the same terms as a registry-resolved one: the
output type comes off the agent, and the ``invalid_output_recovery`` flag off
the definition registered under the agent's name, since the flag has no slot on
``Agent``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    ModelBehaviorError,
    RunErrorHandlers,
    RunItem,
)
from agents.agent_output import AgentOutputSchema, AgentOutputSchemaBase
from agents.run_error_handlers import RunErrorHandlerInput

from ..registry.agent_registry import resolve_agent_definition

logger = logging.getLogger(__name__)

# JSON's two container openers, mapped to the closer that balances them.
_CONTAINER_DELIMITERS = {"{": "}", "[": "]"}


def build_output_schema(
    output_type: Any,
    strict_json_schema: bool = True,
) -> AgentOutputSchemaBase | None:
    """Resolve an agent output type into the SDK schema that validates it.

    Mirrors the SDK's own resolution so a schema built here accepts exactly
    what the run accepted.

    Args:
        output_type: Value of ``Agent.output_type`` — a type, an
            ``AgentOutputSchemaBase``, ``str``, or ``None``.
        strict_json_schema: Passed through to :class:`AgentOutputSchema`.
            ``True`` (the SDK default) validates without type coercion and
            emits a strict JSON schema. Pass ``False`` when the schema is
            being sent to a provider that rejects strict-mode constructs.

    Returns:
        The schema for *output_type*, or ``None`` when the agent produces
        plain text and there is nothing to validate.
    """
    if not _is_structured_output(output_type):
        return None
    if isinstance(output_type, AgentOutputSchemaBase):
        return output_type
    return AgentOutputSchema(output_type, strict_json_schema=strict_json_schema)


def salvage_structured_output(
    raw_text: str | None,
    output_schema: AgentOutputSchemaBase,
) -> Any | None:
    """Parse the first payload in *raw_text* that satisfies *output_schema*.

    Tries the whole message first, then every balanced ``{...}`` / ``[...]``
    span inside it, so a payload wrapped in a code fence or surrounded by
    prose still parses. Candidates are validated, never coerced or patched —
    a message with no valid payload yields ``None`` rather than a guess.

    Args:
        raw_text: The model's final message text.
        output_schema: Schema the payload must satisfy.

    Returns:
        The validated output object, or ``None`` when nothing in *raw_text*
        satisfies the schema.
    """
    if not raw_text:
        return None

    for candidate in iter_payload_candidates(raw_text.strip()):
        try:
            return output_schema.validate_json(candidate)
        except ModelBehaviorError:
            continue
    return None


def iter_payload_candidates(text: str) -> Iterator[str]:
    """Yield *text* itself, then each balanced JSON container found inside it.

    The whole message comes first, so text that is already a bare payload is
    read exactly as it was written. The balanced spans that follow are what
    survive a model wrapping its payload in a ``` fence, a preamble, or
    trailing commentary. Delimiters inside JSON strings do not open or close
    a span, so a payload whose values contain braces still ends where it
    should.

    Args:
        text: Message text to scan. Callers strip it first when a padded
            duplicate of the first candidate is not worth parsing twice.

    Yields:
        Candidate payloads, outermost first. Nothing at all for empty text.
    """
    if not text:
        return
    yield text
    for span in _iter_balanced_spans(text):
        if span != text:
            yield span


def recover_invalid_final_output(handler_input: RunErrorHandlerInput[Any]) -> Any | None:
    """SDK ``invalid_final_output`` handler — salvage the model's own payload.

    Args:
        handler_input: Failed-run snapshot supplied by the SDK.

    Returns:
        A schema-valid replacement final output, or ``None`` to let the SDK
        raise as it does without a handler.
    """
    agent = handler_input.run_data.last_agent
    output_schema = build_output_schema(getattr(agent, "output_type", None))
    if output_schema is None:
        return None

    salvaged = salvage_structured_output(
        _last_message_text(handler_input.run_data.new_items), output_schema
    )
    if salvaged is None:
        logger.warning(
            "Agent '%s' returned structured output that could not be salvaged: %s",
            agent.name,
            handler_input.error,
        )
        return None

    logger.info("Recovered structured output for agent '%s' from a malformed message", agent.name)
    return salvaged


def build_error_handlers(agent: Agent) -> RunErrorHandlers[Any] | None:
    """Build the SDK run error handlers *agent* needs, or None when defaults apply.

    An agent that answers in plain text has no schema to fail, so there is
    nothing for :func:`recover_invalid_final_output` to salvage and the run
    keeps the SDK defaults. An agent whose definition sets
    ``invalid_output_recovery: false`` keeps them too: it asked for a malformed
    response to fail loudly, and that choice holds here the way it holds under
    ``BaseAgentRunner``.

    Args:
        agent: A built agent, from the registry factory or assembled by hand.
            The recovery flag is looked up by ``agent.name``, so an agent
            assembled by hand picks up the decision declared for that name.

    Returns:
        Configured handlers, or None when the agent declares no output type or
        opts out of recovery.
    """
    if not _is_structured_output(agent.output_type):
        return None

    if _recovery_opted_out(agent):
        return None

    return invalid_final_output_handlers()


def invalid_final_output_handlers() -> RunErrorHandlers[Any]:
    """Build the handler bundle that salvages a malformed structured final output.

    Returns:
        A fresh mapping registering :func:`recover_invalid_final_output` under
        the SDK's ``invalid_final_output`` key. Callers get their own dict
        rather than a shared one.
    """
    handlers: RunErrorHandlers[Any] = {
        "invalid_final_output": recover_invalid_final_output,
    }
    return handlers


def _recovery_opted_out(agent: Agent) -> bool:
    """Whether the definition registered under *agent*'s name turns recovery off.

    The flag is a run-level decision with no slot on ``Agent``, so unlike the
    output type it cannot be read off the built agent — the declaration is
    resolved through
    :func:`~sinan_agentic_core.registry.agent_registry.resolve_agent_definition`,
    the same by-name reader :func:`sinan_agentic_core.core.run_config.build_run_config`
    goes through for a declared trim policy. An agent assembled under a name the
    registry does not know declares nothing, so it keeps recovery.
    """
    agent_def = resolve_agent_definition(agent)

    return agent_def is not None and not agent_def.invalid_output_recovery


def _is_structured_output(output_type: Any) -> bool:
    """Whether *output_type* declares a payload a schema has to validate."""
    return bool(output_type) and output_type is not str


def _last_message_text(items: list[RunItem]) -> str | None:
    """Return the text of the most recent message item, if there is one."""
    for item in reversed(items):
        if isinstance(item, MessageOutputItem):
            return ItemHelpers.extract_text(item.raw_item)
    return None


def _iter_balanced_spans(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` / ``[...]`` span, outermost first."""
    index = 0
    while index < len(text):
        closer = _CONTAINER_DELIMITERS.get(text[index])
        if closer is None:
            index += 1
            continue
        end = _find_matching_delimiter(text, index, text[index], closer)
        if end is None:
            index += 1
            continue
        yield text[index : end + 1]
        index = end + 1


def _find_matching_delimiter(text: str, start: int, opener: str, closer: str) -> int | None:
    """Return the index closing the container at *start*, or ``None`` if unbalanced.

    Delimiters inside JSON strings do not count toward the depth, so a payload
    whose values contain braces still matches at the right place.
    """
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return None
