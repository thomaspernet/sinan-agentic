"""Generic Agent Session - Custom session implementation for OpenAI Agent SDK.

This is a generic conversation history manager that works with the OpenAI Agents SDK.
"""

import copy
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from agents.items import TResponseInputItem
from agents.memory.session import SessionABC

from ..core.output_recovery import iter_payload_candidates

if TYPE_CHECKING:
    from ..core.capabilities import Capability
    from .sqlite_store import SQLiteSessionStore

# Key the SDK's AgentOutputSchema wraps a non-dict output type under
# (``agents.agent_output._WRAPPER_DICT_KEY``, openai-agents==0.18.3).
_STRUCTURED_OUTPUT_WRAPPER_KEY = "response"


class ConversationHistory:
    """Generic conversation history container.

    Stores messages in a simple list format compatible with OpenAI SDK.
    Extend this for your specific storage needs (database persistence, etc.).
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def to_list_dict(self) -> list[dict[str, Any]]:
        """Return a snapshot of the messages as plain dictionaries.

        The copy goes all the way down: a message is a dict nesting further
        mutable values — a structured-output ``content`` is itself a list of
        dicts — so copying only the outer list would hand back the history's
        own messages and let a reader rewrite what it replays.

        Returns:
            A detached list of message dicts.
        """
        return copy.deepcopy(self.messages)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to history.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            **kwargs: Additional fields (name, tool_call_id, etc.)
        """
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


class AgentSession(SessionABC):
    """Generic session implementation for OpenAI Agents SDK.

    Manages conversation history for multi-agent workflows with automatic
    summarization when history becomes too long.

    A session owns its history and its metadata, in both directions: what it
    is given is copied on the way in, and what it hands back is copied on the
    way out. Neither a caller that keeps its seed nor a consumer that edits a
    reading can change what the session replays on the next turn. To change a
    session, call its own methods (``add_items``, ``clear_session``,
    ``set_metadata``) rather than editing what a read handed back.

    Usage:
        session = AgentSession(session_id="unique-id")

        # Add messages manually
        await session.add_items([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ])

        # Get history
        history = await session.get_items()
    """

    def __init__(
        self,
        session_id: str,
        initial_history: ConversationHistory | None = None,
        max_items: int = 50,
    ) -> None:
        """Initialize session.

        The seed history is copied, so appending to this session never reaches
        the caller's object and two sessions seeded from one history do not
        share a message list. Each message is a dict that nests further mutable
        values — a structured-output ``content`` arrives as a list of dicts —
        so the copy goes all the way down; copying only the message list would
        leave those nested edit paths open.

        Args:
            session_id: Unique identifier for this session
            initial_history: Optional existing conversation history, taken as a
                snapshot rather than adopted
            max_items: Maximum messages before triggering summarization
        """
        self.session_id = session_id
        self.history = (
            ConversationHistory() if initial_history is None else copy.deepcopy(initial_history)
        )
        self.max_items = max_items
        # Arbitrary session metadata. Values are snapshots: set_metadata copies
        # on the way in and get_metadata copies on the way out.
        self.metadata: dict[str, Any] = {}

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Retrieve conversation history as list of dicts.

        The items are a snapshot taken by ``ConversationHistory.to_list_dict``,
        so editing one does not reach the session.

        Args:
            limit: Optional limit on number of items to return (from end)

        Returns:
            List of message dicts compatible with Agent SDK
        """
        items = self.history.to_list_dict()
        if limit:
            return cast(list[TResponseInputItem], items[-limit:])
        return cast(list[TResponseInputItem], items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Add new messages to conversation history.

        Args:
            items: List of message dicts to add
        """
        for raw_item in items:
            # Items arrive as the SDK's TResponseInputItem union of TypedDicts;
            # we treat them as plain dicts internally for storage.
            item = cast(dict[str, Any], raw_item)

            # Skip empty messages
            content = item.get("content", "")
            if not content or (isinstance(content, str) and not content.strip()):
                continue

            # Clean up SDK's structured output format
            # SDK returns: [{'text': '{"response":...}', 'type': 'output_text', ...}]
            # We want: just the JSON string
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    content = content[0]["text"]

            # For assistant messages with structured output, extract response
            if item.get("role") == "assistant" and isinstance(content, str):
                content = _unwrap_structured_output(content)

            # Create message dict
            msg: dict[str, Any] = {"role": item.get("role"), "content": str(content)}

            # Add optional fields
            if "name" in item:
                msg["name"] = item["name"]
            if "tool_call_id" in item:
                msg["tool_call_id"] = item["tool_call_id"]

            self.history.messages.append(msg)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent message.

        The message is handed over rather than shared — it leaves the history
        in the same call, so there is nothing left for a copy to detach.

        Returns:
            Last message as dict, or None if empty
        """
        if self.history.messages:
            return cast(TResponseInputItem, self.history.messages.pop())
        return None

    async def clear_session(self) -> None:
        """Clear all messages from history."""
        self.history.messages.clear()

    def get_history(self) -> ConversationHistory:
        """Get a snapshot of the full ConversationHistory object.

        The copy is taken on the object itself, not on its message list, for
        the same two reasons the constructor deep-copies its seed: a caller's
        ``ConversationHistory`` subclass survives the round trip, and the
        values nested inside a message are detached along with the list.

        Returns:
            A detached ConversationHistory for debugging/persistence
        """
        return copy.deepcopy(self.history)

    def needs_summarization(self) -> bool:
        """Check if session has exceeded max items threshold.

        Returns:
            True if session should be summarized
        """
        return len(self.history.messages) > self.max_items

    def get_message_count(self) -> int:
        """Get current number of messages in session.

        Returns:
            Number of messages
        """
        return len(self.history.messages)

    def set_metadata(self, key: str, value: Any) -> None:
        """Store a snapshot of *value* under *key*.

        Metadata is a record of what was stored, not a live view of the
        caller's object, so the value is copied on the way in. The store is
        declared ``dict[str, Any]``, so a value can nest containers of its own
        and the copy goes all the way down; a stored value must therefore be
        deep-copyable, which makes this the wrong place for a live handle.

        Args:
            key: Metadata key
            value: Metadata value, taken as a snapshot
        """
        self.metadata[key] = copy.deepcopy(value)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a snapshot of session metadata.

        A stored value is copied on the way out, so two readers of one key do
        not share an object and neither can write back into the session.
        *default* is the caller's own value handed straight back — the session
        never retained it, so there is nothing to detach.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            A copy of the stored value, or *default* when the key is unset
        """
        if key not in self.metadata:
            return default
        return copy.deepcopy(self.metadata[key])

    # -----------------------------------------------------------------
    # Capability snapshot/rehydrate
    # -----------------------------------------------------------------

    def rehydrate_capabilities(
        self,
        store: "SQLiteSessionStore",
        capabilities: Iterable["Capability"],
    ) -> None:
        """Restore each capability from its persisted snapshot, if any.

        For every capability that produces a snapshot key (i.e.
        ``to_snapshot()`` returns a dict), look up the stored blob and call
        ``from_snapshot()``. Capabilities without a snapshot row are left
        untouched, and capabilities that opt out of persistence (default
        ``to_snapshot`` returns ``None``) are silently skipped.

        Args:
            store: Backing SQLite store that owns the snapshot table.
            capabilities: Capabilities attached to this session in the
                same order they will be cloned by the runner.
        """
        snapshots = store.load_all_capability_snapshots(self.session_id)
        if not snapshots:
            return
        for capability in capabilities:
            if capability.to_snapshot() is None:
                continue
            key = _capability_key(capability)
            data = snapshots.get(key)
            if data is None:
                continue
            capability.from_snapshot(data)

    def persist_capabilities(
        self,
        store: "SQLiteSessionStore",
        capabilities: Iterable["Capability"],
    ) -> None:
        """Persist a snapshot for each capability that opts in.

        Capabilities whose ``to_snapshot()`` returns ``None`` are treated
        as stateless and skipped.
        """
        for capability in capabilities:
            data = capability.to_snapshot()
            if data is None:
                continue
            store.save_capability_snapshot(
                self.session_id,
                _capability_key(capability),
                data,
            )


def _unwrap_structured_output(content: str) -> str:
    """Return the payload the SDK wrapped under its structured-output key.

    An agent with a non-dict ``output_dataclass`` answers with the value
    nested under a wrapper key; history should carry the answer, not the
    envelope. The message is not always a bare payload — a model that fences
    its JSON or writes a preamble around it produces text no parser accepts
    whole — so the wrapped payload is located by the same balanced-span scan
    that structured-output recovery uses, rather than by parsing the message
    verbatim and giving up when that fails.

    Args:
        content: Text of an assistant message.

    Returns:
        The wrapped value re-serialized, or *content* unchanged when nothing
        in it is a wrapped payload.
    """
    for candidate in iter_payload_candidates(content.strip()):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict) and _STRUCTURED_OUTPUT_WRAPPER_KEY in parsed:
            return json.dumps(parsed[_STRUCTURED_OUTPUT_WRAPPER_KEY])
    return content


def _capability_key(capability: "Capability") -> str:
    """Stable identifier for a capability instance.

    Uses the fully-qualified class name so two capabilities of the same
    type collapse to one snapshot row per session. Subclasses that need
    per-instance separation can override by setting ``snapshot_key``.
    """
    explicit = getattr(capability, "snapshot_key", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    cls = type(capability)
    return f"{cls.__module__}.{cls.__qualname__}"
