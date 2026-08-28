"""Capability steering — fragments delivered at the tail of the model input.

A capability contributes an instruction fragment through
:meth:`Capability.instructions`, and those fragments are volatile by design: a
turn budget counts down, a recovery tracker gains an entry after each failed
tool call. Merged into the system prompt — which a provider serializes first —
a fragment that changes between two calls of one run moves the very front of
the request, so the whole conversation behind it falls out of the provider's
cacheable prefix and a run making eight model calls pays full price for its
history eight times.

This module delivers the same fragments at the other end of the request. The
SDK's ``RunConfig.call_model_input_filter`` runs immediately before each model
call and may edit what is sent, so the fragments ride as one trailing input
item: the system prompt stays the string resolved once at agent creation, the
history in front of the item stays a stable prefix, and the steering lands
where a model follows it most reliably. The item is ephemeral — built per call,
appended to what the SDK is about to send, and never written to the session.

The same filter carries the drift guard. A prefix is only cacheable while it
stays byte-identical, and this package is not the only thing that can move it:
a consumer's own instruction callable, another filter, or the SDK itself can
resolve a different string on a later call. The filter remembers what the
previous call resolved to and warns when a call differs, so instability is
reported rather than silently paid for.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

from agents import RunContextWrapper
from agents.items import TResponseInputItem
from agents.run_config import CallModelData, ModelInputData

from .base import Capability

logger = logging.getLogger(__name__)

# Role the steering item is sent under. Deliberately not ``user``: a trailing
# user message reads as a fresh question the model should answer, and it would
# also shift the tool-output trimmer's untouched window, which is measured in
# user messages. ``system`` survives both the Responses and chat-completions
# item converters (``agents.models.chatcmpl_converter``, openai-agents 0.21.1).
STEERING_ITEM_ROLE = "system"

# What separates two fragments. A blank line, exactly as when the fragments were
# joined into the system prompt, so moving them leaves their wording untouched.
FRAGMENT_SEPARATOR = "\n\n"


class CapabilitySteering:
    """Append the run's capability fragments to each model call's input.

    Implements the SDK's ``CallModelInputFilter`` protocol, so an instance is
    passed straight to ``RunConfig.call_model_input_filter``. State is per run:
    build one per run rather than sharing an instance, or the drift guard reads
    one run's instructions against another's.
    """

    def __init__(self, capabilities: Sequence[Capability]) -> None:
        """
        Args:
            capabilities: The run's capabilities, in the order their fragments
                are joined. The sequence is copied, so a caller that keeps its
                own list and edits it after construction cannot change what a
                live filter reads. The capabilities inside it are deliberately
                not copied: a budget's remaining turns and a tracker's error
                state are the live per-run state each fragment reports.
        """
        self._capabilities = list(capabilities)
        self._previous_instructions: str | None = None
        self._seen_a_call = False

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        """Return the model input with this call's steering item appended.

        Existing items are forwarded by identity and never rebuilt, so a run
        under a server-side conversation can still match every item it is about
        to send against the input it has pending
        (``agents.run_internal.oai_conversation``, openai-agents 0.21.1).
        """
        model_data = data.model_data
        self._warn_on_instruction_drift(model_data.instructions)

        steering = self._steering_text(RunContextWrapper[Any](data.context))
        if steering is None:
            return model_data

        item = cast(
            TResponseInputItem,
            {"role": STEERING_ITEM_ROLE, "content": steering},
        )
        return ModelInputData(
            input=[*model_data.input, item],
            instructions=model_data.instructions,
        )

    def _steering_text(self, ctx: RunContextWrapper[Any]) -> str | None:
        """Join what the capabilities say this call, or None when they say nothing.

        The wrapper is built here rather than handed down: the SDK's filter
        payload carries the run's context object but not the run's wrapper
        around it (``agents.run_config.CallModelData``), so ``ctx.context`` is
        the object the caller passed to ``execute()`` while ``ctx.usage`` starts
        empty rather than reporting the run so far.
        """
        fragments = [fragment for cap in self._capabilities if (fragment := cap.instructions(ctx))]
        if not fragments:
            return None
        return FRAGMENT_SEPARATOR.join(fragments)

    def _warn_on_instruction_drift(self, instructions: str | None) -> None:
        """Warn when this call's resolved instructions differ from the last call's."""
        if self._seen_a_call and instructions != self._previous_instructions:
            logger.warning(
                "Resolved instructions changed between two model calls of one run "
                "(previous: %s, current: %s). The provider's cached prefix is lost "
                "from this call on.",
                _instructions_shape(self._previous_instructions),
                _instructions_shape(instructions),
            )
        self._seen_a_call = True
        self._previous_instructions = instructions


def build_capability_steering(
    capabilities: Sequence[Capability],
) -> CapabilitySteering | None:
    """Build the steering filter for *capabilities*, or None when there is none.

    Args:
        capabilities: The run's capabilities. Empty means nothing to steer with,
            which is the common case — most agents declare none.

    Returns:
        The filter to install, or None. Each call builds a fresh filter, so no
        two runs share one's drift state.
    """
    if not capabilities:
        return None

    return CapabilitySteering(capabilities)


def _instructions_shape(instructions: str | None) -> str:
    """Describe resolved instructions for the drift warning.

    The prompt itself is never logged — it is large and carries whatever the
    consumer put in it. Absent instructions and empty instructions are reported
    as the different states they are.
    """
    if instructions is None:
        return "absent"

    return f"{len(instructions)} chars"
