"""Standard output models for agent responses.

Provides reusable dataclasses for:
- ToolOutput: Base return type for all tools
- ChatResponse: Standard chat response with tool tracking
"""

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolOutput:
    """Standard output format for tool functions.

    Use as a base class or directly for consistent tool returns.

    The result owns its ``data`` and ``metadata`` mappings: it is a fixed record
    of what the tool produced, so both are copied on the way in and copied again
    on the way out of :meth:`to_dict` rather than aliased to whoever supplied
    them. A subclass that defines its own ``__post_init__`` must call
    ``super().__post_init__()``, or its instances go back to aliasing the
    caller's mappings. That call copies the two mappings declared here and
    nothing else — the base class cannot reach a field it does not declare — so
    a subclass that adds a collection of its own must copy that one itself.

    Attributes:
        success: Whether the tool executed successfully
        data: The actual data returned
        error: Error message if success is False
        metadata: Additional context
    """

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Take ownership of the caller's mappings.

        ``default_factory`` only covers the omitted-argument case, so a tool
        that hands over a mapping it keeps — the row it is still building, a
        metadata dict reused across calls — leaves that mapping aliased into
        the result. Copying here means a later edit by the tool cannot change
        a result already returned, and two results built from one mapping do
        not share it.

        The copies are deep. Both are ``dict[str, Any]`` holding whatever the
        tool produced, so a value can itself be a list or a nested object;
        detaching only the outer mapping would leave an edit inside such a
        value reaching the result.
        """
        self.data = copy.deepcopy(self.data)
        self.metadata = copy.deepcopy(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Render the result as a payload, with its own copies of the mappings.

        A consumer that edits ``payload["data"]``, or a metadata value merged
        into the payload — at the top level or inside a nested value — must not
        reach back into the result it read.
        """
        result: dict[str, Any] = {"success": self.success}
        if self.data is not None:
            result["data"] = copy.deepcopy(self.data)
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result.update(copy.deepcopy(self.metadata))
        return result


@dataclass
class ChatResponse:
    """Standard response format for chat interactions.

    The response owns its ``tools_called`` list and ``usage`` mapping: it is a
    fixed record of one chat turn, so both are copied on the way in and copied
    again on the way out of :meth:`to_dict` rather than aliased to whoever
    supplied them.

    Attributes:
        success: Whether the chat completed successfully
        response: The assistant's text response
        session_id: Session ID for conversation continuity
        tools_called: List of tool names that were invoked
        error: Error message if success is False
        usage: Token usage dict with requests, input_tokens, output_tokens,
            total_tokens, input_tokens_details, output_tokens_details
    """

    success: bool
    response: str = ""
    session_id: str = "default"
    tools_called: list[str] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Take ownership of the caller's collections.

        The natural construction is from live run state — the hooks bundle's
        tool list and the runner's last usage record — both of which outlive
        the turn: the list keeps growing while the run continues, and the
        runner overwrites its usage record on the next turn. Copying here pins
        both to the turn this response describes, and means a consumer editing
        ``response.tools_called`` cannot reach back into the hooks bundle.

        ``tools_called`` holds plain tool names, so detaching the list is
        enough. ``usage`` nests ``input_tokens_details`` and
        ``output_tokens_details``, so it is copied all the way down — a shallow
        copy would leave an edit to a cached- or reasoning-token count reaching
        through.
        """
        self.tools_called = list(self.tools_called)
        self.usage = copy.deepcopy(self.usage)

    def to_dict(self) -> dict[str, Any]:
        """Render the response as a payload, with its own copies of the collections.

        A consumer that edits ``payload["tools_called"]`` or ``payload["usage"]``
        — including inside a nested usage detail — must not reach back into the
        response it read.
        """
        result: dict[str, Any] = {
            "success": self.success,
            "response": self.response,
            "session_id": self.session_id,
        }
        if self.tools_called:
            result["tools_called"] = list(self.tools_called)
        if self.error:
            result["error"] = self.error
        if self.usage:
            result["usage"] = copy.deepcopy(self.usage)
        return result


__all__ = [
    "ToolOutput",
    "ChatResponse",
]
