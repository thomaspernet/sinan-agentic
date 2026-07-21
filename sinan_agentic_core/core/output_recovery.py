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
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from agents import ItemHelpers, MessageOutputItem, ModelBehaviorError, RunItem
from agents.agent_output import AgentOutputSchema, AgentOutputSchemaBase
from agents.run_error_handlers import RunErrorHandlerInput

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
    if not output_type or output_type is str:
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

    for candidate in _iter_payload_candidates(raw_text.strip()):
        try:
            return output_schema.validate_json(candidate)
        except ModelBehaviorError:
            continue
    return None


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


def _last_message_text(items: list[RunItem]) -> str | None:
    """Return the text of the most recent message item, if there is one."""
    for item in reversed(items):
        if isinstance(item, MessageOutputItem):
            return ItemHelpers.extract_text(item.raw_item)
    return None


def _iter_payload_candidates(text: str) -> Iterator[str]:
    """Yield *text* itself, then each balanced JSON container found inside it."""
    if not text:
        return
    yield text
    for span in _iter_balanced_spans(text):
        if span != text:
            yield span


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
