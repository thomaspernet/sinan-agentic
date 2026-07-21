"""Declarative opt-in trimming of oversized tool outputs before the model call.

Long-running agents accumulate bulky tool outputs — search results, file dumps,
error payloads — that stay in the conversation long after they stop being
relevant. Left alone they grow the input until the run dies on
``context_length_exceeded`` and :meth:`BaseAgentRunner._execute_with_fallback`
has to rescue it with a second, condensed model call.

The SDK ships a filter that prevents that: it replaces oversized tool outputs
from older turns with a short preview *before* each model call, leaving recent
turns at full fidelity. Reaching it means importing ``agents.extensions`` and
constructing the filter by hand at every agent.

This module carries the same choice as data — how many turns to keep intact, the
size above which an output is trimmed, how much of it to preserve, and which
tools are eligible. ``BaseAgentRunner`` translates it into the SDK filter inside
``_build_run_config``, so trimming stays declarative in ``agents.yaml`` and
reaches every execution branch that runs through the SDK.

Trimming is off unless declared: it removes content the model would otherwise
have seen, so no agent gets it implicitly.
"""

from __future__ import annotations

from typing import Any

from agents.extensions import ToolOutputTrimmer
from pydantic import BaseModel, Field


class ToolOutputTrimConfig(BaseModel):
    """Opt-in tool-output trimming for one agent.

    Every field is optional; the SDK fills an unset one with its own default.
    Declaring the key with no fields (``tool_output_trim: {}``) therefore means
    "trim with SDK defaults".

    Declared in ``agents.yaml``::

        agents:
          researcher:
            model: gpt-4o-mini
            description: Reads papers
            tool_output_trim:
              recent_turns: 3
              max_output_chars: 4000
              preview_chars: 500
              trimmable_tools: [web_search]

    Attributes:
        recent_turns: How many recent user turns stay untouched. Items at or
            after the Nth-from-last user message are never trimmed. The window
            is measured in *user messages*, so nothing trims until the
            conversation holds more than this many of them — a single question
            answered by any number of tool calls is left whole at every setting.
        max_output_chars: Outputs longer than this are candidates for trimming.
        preview_chars: How much of the original output survives as a preview.
            The SDK keeps the original whenever the replacement would not be
            shorter, so a preview at or above ``max_output_chars`` still shrinks
            far larger outputs — it only spares the ones near the cap.
        trimmable_tools: Tool names whose outputs may be trimmed. Unset means
            every tool is eligible.
    """

    recent_turns: int | None = Field(default=None, ge=1)
    max_output_chars: int | None = Field(default=None, ge=1)
    preview_chars: int | None = Field(default=None, ge=0)
    # An empty list would make no tool eligible, so the filter would run on every
    # model call and never trim anything. Unset — not empty — means "all tools".
    trimmable_tools: list[str] | None = Field(default=None, min_length=1)

    def build(self) -> ToolOutputTrimmer:
        """Translate this config into the SDK's ``ToolOutputTrimmer``.

        Unset fields are omitted so the SDK's own defaults apply rather than a
        second set of defaults maintained here.
        """
        fields: dict[str, Any] = self.model_dump(exclude_none=True)
        return ToolOutputTrimmer(**fields)
