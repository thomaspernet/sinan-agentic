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
from typing import Any

from agents import RunContextWrapper, Tool


class Capability:
    """Base class for pluggable agent behaviors.

    Default implementations are no-ops so that a trivial subclass type-checks
    under ``mypy --strict`` without overriding anything. Override the methods
    that matter for the behavior you are adding.
    """

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        """Contribute a fragment to the system prompt for the next turn.

        Return ``None`` to contribute nothing. The runtime joins fragments
        from all attached capabilities and appends them to the agent's
        base instructions before each LLM call.
        """
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
        return copy.deepcopy(self)

    def tools(self) -> list[Tool]:
        """Return tools this capability exposes to the agent.

        Reserved for future use. The default returns an empty list.
        """
        return []
