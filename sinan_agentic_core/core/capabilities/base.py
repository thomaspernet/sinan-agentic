"""Capability — pluggable agent behavior primitive.

A ``Capability`` is a single, focused behavior that can be attached to an
agent (turn budgeting, tool-error recovery, memory, audit logging, ...)
without editing the runner. The runtime invokes the lifecycle methods at
the appropriate moments and merges instruction fragments into the system
prompt before each LLM call.

The shape mirrors OpenAI Agents SDK ``sandbox.Capability`` (clone, bind,
instructions) so future interop is cheap, but this class does not depend
on any sandbox internals.

Subclasses override only the methods they need; every method has a safe
default. Capabilities are stateful and must not leak across runs — the
runtime calls ``clone()`` before each execution and ``reset()`` at the
start of each ``execute()`` call.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Optional

from agents import RunContextWrapper, Tool


class Capability:
    """Base class for pluggable agent behaviors.

    Default implementations are no-ops so that a trivial subclass type-checks
    under ``mypy --strict`` without overriding anything. Override the methods
    that matter for the behavior you are adding.

    Lifecycle methods mirror the SDK's ``RunHooks`` shape but with simpler
    abstracted signatures. The runtime adapter (``_CompositeHooks``) bridges
    SDK callbacks to these methods.

    Streaming events: assign ``capability.on_event`` to a callback to emit
    custom events during a run. The runtime sets this on cloned capabilities
    when streaming.
    """

    on_event: Optional[Callable[[dict[str, Any]], None]] = None

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        """Contribute a fragment to the system prompt for the next turn.

        Return ``None`` to contribute nothing. The runtime joins fragments
        from all attached capabilities and appends them to the agent's
        base instructions before each LLM call.
        """
        return None

    def on_agent_start(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
    ) -> None:
        """Called when the agent run begins."""
        return None

    def on_agent_end(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        output: Any,
    ) -> None:
        """Called when the agent run ends with a final output."""
        return None

    def on_handoff(
        self,
        ctx: RunContextWrapper[Any],
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        """Called when control hands off from one agent to another."""
        return None

    def on_tool_start(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        args: str,
    ) -> None:
        """Called immediately before a tool executes."""
        return None

    def on_tool_end(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        result: str,
    ) -> None:
        """Called immediately after a tool executes."""
        return None

    def on_llm_start(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        system_prompt: Optional[str],
        input_items: Any,
    ) -> None:
        """Called immediately before the model is invoked for a turn."""
        return None

    def on_llm_end(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        response: Any,
    ) -> None:
        """Called immediately after the model returns a response."""
        return None

    def reset(self) -> None:
        """Reset mutable state. Called at the start of each ``execute()`` call."""
        return None

    def clone(self) -> Capability:
        """Return an independent per-run copy.

        The default implementation does a deep copy, which is correct for
        capabilities whose state is plain Python data. Override when the
        capability holds non-copyable references (open connections, locks)
        and needs custom copy semantics.
        """
        clone = copy.deepcopy(self)
        clone.on_event = None
        return clone

    def to_snapshot(self) -> dict[str, Any] | None:
        """Return a JSON-serializable snapshot of this capability's mutable state.

        Return ``None`` (the default) to opt out of persistence: the runtime
        treats the capability as stateless and skips writing a snapshot row.
        Override to expose counters, queues, or any state that must survive
        a process restart so the next session can resume mid-run.

        The returned dict must round-trip through ``json.dumps`` /
        ``json.loads`` — keep values to plain types (str, int, float, bool,
        list, dict, None).
        """
        return None

    def from_snapshot(self, data: dict[str, Any]) -> None:
        """Rehydrate mutable state from a snapshot produced by ``to_snapshot``.

        The default is a no-op so subclasses without persistence stay
        unaffected. Implementations should be tolerant of missing keys (the
        snapshot may have been written by an older version of the
        capability) and never raise on partial input.
        """
        return None

    def tools(self) -> list[Tool]:
        """Return tools this capability exposes to the agent.

        Reserved for future use. The default returns an empty list.
        """
        return []
