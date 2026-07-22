"""Previews of tool output carried by streamed run events.

A streamed run emits a ``tool_output`` event as soon as a tool returns, so a UI
can show what a tool produced without waiting for the final answer. The whole
output does not belong on that wire: a search or file-dump tool can return tens
of thousands of characters, and the event is a progress notice, not the payload
the model reasons over. Only the head of it travels.

How much travels is one decision, and both streaming paths make it — the
runner's ``BaseAgentRunner._execute_streamed`` and the chat service's
``chat_streamed``. :data:`TOOL_OUTPUT_PREVIEW_CHARS` holds it once and
:func:`tool_output_preview` applies it, so the two paths cannot drift into
showing different amounts of the same tool's output.

This is unrelated to ``tool_output_trim``: that policy shrinks bulky outputs
*before a model call* and changes what the model sees, and it is declared
per-agent in ``agents.yaml``. This preview only shortens what a stream consumer
is shown, and never affects the run.
"""

from __future__ import annotations

# Long enough to read a short tool result whole, short enough that a file dump
# does not flood the event stream. Structural to the event shape, not a policy
# knob — a consumer that wants the full output reads the run's final result.
TOOL_OUTPUT_PREVIEW_CHARS = 500


def tool_output_preview(output: object) -> str:
    """Render *output* as the preview a streamed ``tool_output`` event carries.

    A tool returns whatever its signature declares — a string, a dict, a
    dataclass — so the value is rendered before it is cut.

    Args:
        output: The value the tool returned, taken off the SDK's
            ``tool_call_output_item``.

    Returns:
        The rendered output, truncated to :data:`TOOL_OUTPUT_PREVIEW_CHARS`.
        Anything shorter is returned whole.
    """
    return str(output)[:TOOL_OUTPUT_PREVIEW_CHARS]
