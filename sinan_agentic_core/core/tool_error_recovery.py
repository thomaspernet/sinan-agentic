"""Tool error recovery — intelligent error tracking with dynamic guidance.

When a tool returns an error, agents often retry with identical parameters,
wasting turns in a loop. ToolErrorRecovery solves this by:

1. Tracking tool errors and the arguments that caused them (via RunHooks)
2. Detecting repeated identical calls (same tool + same args)
3. Injecting progressive recovery guidance into the agent's instructions
   before each LLM call (via dynamic instructions)

This follows the same pattern as TurnBudget:
- State is tracked via RunHooks (on_tool_start / on_tool_end)
- Guidance is injected via dynamic instructions (callable)
- The agent reads the guidance and decides what to do (leverages LLM intelligence)

Works for all tool types: custom @function_tool, agent-as-tool, and MCP tools.

Usage:
    recovery = ToolErrorRecovery(tool_registry=registry)
    hooks = ToolErrorRecoveryHooks(recovery)
    # Compose with other hooks, wire into dynamic instructions
    # See BaseAgentRunner for integration details.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from agents import RunHooks

logger = logging.getLogger(__name__)


@dataclass
class ToolErrorEntry:
    """Tracks a tool error with its context.

    Attributes:
        tool_name: Name of the tool that errored.
        error: The error message from the tool response.
        args_hash: Hash of the arguments that caused the error.
        call_count: How many times this tool errored (any args).
        identical_count: How many times the exact same args were retried.
        recovery_hint: Static hint from tool registration (if any).
        last_args_summary: Short summary of the arguments for instruction text.
    """

    tool_name: str
    error: str
    args_hash: str
    call_count: int = 1
    identical_count: int = 1
    recovery_hint: str = ""
    last_args_summary: str = ""


class ToolErrorRecovery:
    """Track tool errors and generate recovery guidance for agents.

    Detects error patterns (repeated calls, identical arguments) and builds
    an instruction section that tells the agent what went wrong and how to
    recover. The instruction section is injected dynamically before each
    LLM call, so the agent always has up-to-date error awareness.

    Attributes:
        max_identical_before_stop: After this many identical failures, the
            instruction section tells the agent to stop retrying entirely.
    """

    def __init__(
        self,
        tool_registry: Any = None,
        mcp_hints: dict[str, str] | None = None,
        max_identical_before_stop: int = 3,
    ) -> None:
        """Initialize the recovery tracker.

        Args:
            tool_registry: ToolRegistry instance for looking up recovery_hint.
                If None, only mcp_hints are used for hint lookup.
            mcp_hints: Mapping of MCP tool names to recovery hint strings.
                Use this for tools you don't control (external MCP servers).
            max_identical_before_stop: After N identical failures, instruct
                the agent to stop retrying. Default: 3.
        """
        self._registry = tool_registry
        self._mcp_hints = mcp_hints or {}
        self._errors: dict[str, ToolErrorEntry] = {}  # key: tool_name
        self._last_args: dict[str, str] = {}  # tool_name -> last args_hash
        self.max_identical_before_stop = max_identical_before_stop

    # Status values that indicate a tool failure (used by _extract_error).
    _FAILURE_STATUSES = frozenset({"failed", "validation_error", "error"})

    def record_tool_result(
        self,
        tool_name: str,
        result: str,
        arguments: str = "",
    ) -> None:
        """Record a tool result, tracking errors and clearing on success.

        Called from ToolErrorRecoveryHooks.on_tool_end(). Parses the result
        to detect errors and tracks:
        - Total error count per tool
        - Identical-argument retry count
        - Recovery hints from the registry

        Error detection supports two JSON shapes:
        - ``{"error": "message"}`` — explicit error key
        - ``{"status": "failed", "message": "..."}`` — status-based failure

        On success, clears the tracked error for this tool.

        Args:
            tool_name: Name of the tool that was called.
            result: The tool's return value (usually a JSON string).
            arguments: Raw JSON string of the arguments passed to the tool.
        """
        try:
            data = json.loads(result) if isinstance(result, str) else {}
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(data, dict):
            return

        error_msg = self._extract_error(data)
        if error_msg is None:
            # Success — clear tracked error for this tool
            self._errors.pop(tool_name, None)
            self._last_args.pop(tool_name, None)
            return
        args_hash = self._hash_args(arguments)
        hint = self._get_hint(tool_name)
        args_summary = self._summarize_args(arguments)

        existing = self._errors.get(tool_name)
        prev_hash = self._last_args.get(tool_name)

        if existing:
            existing.call_count += 1
            existing.error = error_msg
            existing.last_args_summary = args_summary
            if args_hash == prev_hash:
                existing.identical_count += 1
            else:
                existing.identical_count = 1
            existing.args_hash = args_hash
        else:
            self._errors[tool_name] = ToolErrorEntry(
                tool_name=tool_name,
                error=error_msg,
                args_hash=args_hash,
                recovery_hint=hint,
                last_args_summary=args_summary,
            )

        self._last_args[tool_name] = args_hash

        logger.info(
            "Tool error tracked: %s (count=%d, identical=%d) — %s",
            tool_name,
            self._errors[tool_name].call_count,
            self._errors[tool_name].identical_count,
            error_msg[:100],
        )

    def build_instruction_section(self) -> str:
        """Build dynamic instruction text about recent tool errors.

        Returns guidance appropriate to the current state:
        - First failure: show error + recovery hint
        - Repeated failure (different args): suggest alternative approach
        - Repeated failure (same args): escalate — tell agent to stop

        Returns:
            Instruction text to append to the agent's system prompt,
            or empty string if no errors are tracked.
        """
        if not self._errors:
            return ""

        lines = ["## Tool Error Recovery"]

        for entry in self._errors.values():
            if entry.identical_count >= self.max_identical_before_stop:
                lines.append(self._section_stop(entry))
            elif entry.identical_count >= 2:
                lines.append(self._section_repeated(entry))
            else:
                lines.append(self._section_first(entry))

        lines.append(
            "\nGeneral rule: Never retry a tool call with identical parameters. "
            "If a tool fails, read the error and choose a different approach."
        )
        return "\n".join(lines)

    @property
    def has_errors(self) -> bool:
        """True if any tool errors are currently tracked."""
        return bool(self._errors)

    @property
    def has_critical_errors(self) -> bool:
        """True if any tool has hit the identical-retry stop threshold."""
        return any(
            e.identical_count >= self.max_identical_before_stop
            for e in self._errors.values()
        )

    def get_error_summary(self) -> dict[str, Any]:
        """Return a summary dict of all tracked errors (for logging/events)."""
        return {
            name: {
                "error": entry.error,
                "call_count": entry.call_count,
                "identical_count": entry.identical_count,
            }
            for name, entry in self._errors.items()
        }

    def reset(self) -> None:
        """Clear all tracked errors. Called at the start of each execution."""
        self._errors.clear()
        self._last_args.clear()

    # ------------------------------------------------------------------ #
    # Instruction section builders (progressive escalation)
    # ------------------------------------------------------------------ #

    def _section_first(self, entry: ToolErrorEntry) -> str:
        """First failure: show error and hint."""
        args_info = f" (called with: {entry.last_args_summary})" if entry.last_args_summary else ""
        line = f"- {entry.tool_name} returned error: \"{entry.error}\"{args_info}"
        if entry.recovery_hint:
            line += f"\n  Recovery hint: {entry.recovery_hint}"
        return line

    def _section_repeated(self, entry: ToolErrorEntry) -> str:
        """Repeated failure with same args: warn strongly."""
        line = (
            f"- {entry.tool_name} has FAILED {entry.identical_count} TIMES "
            f"with identical arguments. Error: \"{entry.error}\"\n"
            f"  You are repeating the same failing call. "
            f"Change your parameters or use a different tool/approach."
        )
        if entry.recovery_hint:
            line += f"\n  Available alternatives: {entry.recovery_hint}"
        return line

    def _section_stop(self, entry: ToolErrorEntry) -> str:
        """Too many identical retries: tell agent to stop."""
        return (
            f"- STOP: {entry.tool_name} has failed {entry.identical_count} times "
            f"with the same arguments. Do NOT call this tool again.\n"
            f"  Error was: \"{entry.error}\"\n"
            f"  Move on to your next task, try a completely different approach, "
            f"or return your partial results."
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_error(cls, data: dict[str, Any]) -> str | None:
        """Extract an error message from a tool result dict.

        Supports two common shapes:
        - ``{"error": "message"}`` — explicit error key (highest priority)
        - ``{"status": "failed"|"validation_error"|"error", "message": "..."}``
          — status-based failure

        Returns the error message string, or None if the result is a success.
        """
        if "error" in data:
            return str(data["error"])
        status = data.get("status")
        if isinstance(status, str) and status in cls._FAILURE_STATUSES:
            return str(data.get("message", f"Tool returned status: {status}"))
        return None

    def _get_hint(self, tool_name: str) -> str:
        """Look up recovery hint from registry or MCP config."""
        if self._registry:
            tool_def = self._registry.get_tool(tool_name)
            if tool_def and getattr(tool_def, "recovery_hint", ""):
                return tool_def.recovery_hint
        return self._mcp_hints.get(tool_name, "")

    @staticmethod
    def _hash_args(arguments: str) -> str:
        """Create a stable hash of tool arguments for dedup detection."""
        if not arguments:
            return "empty"
        try:
            # Normalize JSON key ordering for stable hashing
            parsed = json.loads(arguments)
            normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            normalized = arguments
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    @staticmethod
    def _summarize_args(arguments: str) -> str:
        """Create a short human-readable summary of tool arguments."""
        if not arguments:
            return ""
        try:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                return str(parsed)[:80]
            # Show non-empty values only, truncated
            parts = []
            for k, v in parsed.items():
                if v is None or v == "" or v == []:
                    parts.append(f"{k}=<empty>")
                else:
                    val_str = str(v)
                    if len(val_str) > 40:
                        val_str = val_str[:37] + "..."
                    parts.append(f"{k}={val_str}")
            return ", ".join(parts[:5])  # max 5 params shown
        except (json.JSONDecodeError, TypeError):
            return arguments[:80]


class ToolErrorRecoveryHooks(RunHooks):
    """RunHooks subclass that tracks tool errors via on_tool_end.

    Captures tool name, arguments, and result after each tool call.
    The ToolErrorRecovery instance analyzes the result and tracks errors.
    Dynamic instructions read recovery.build_instruction_section() to
    inject guidance before the next LLM call.

    Optionally accepts an on_event callback for streaming error events.
    """

    def __init__(
        self,
        recovery: ToolErrorRecovery,
        on_event: Optional[Callable] = None,
    ) -> None:
        self.recovery = recovery
        self.on_event = on_event
        self._pending_args: dict[str, str] = {}  # tool_name -> arguments

    async def on_tool_start(
        self,
        context: Any,
        agent: Any,
        tool: Any,
    ) -> None:
        """Capture tool arguments before execution.

        The SDK passes a ToolContext (subclass of RunContextWrapper) that
        has tool_arguments. We store them here for use in on_tool_end.
        """
        tool_name = getattr(tool, "name", str(tool))
        # ToolContext has .tool_arguments (raw JSON string)
        arguments = getattr(context, "tool_arguments", "")
        self._pending_args[tool_name] = arguments

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: str,
    ) -> None:
        """Record tool result for error tracking."""
        tool_name = getattr(tool, "name", str(tool))
        arguments = self._pending_args.pop(tool_name, "")

        self.recovery.record_tool_result(tool_name, result, arguments)

        if self.on_event and self.recovery.has_errors:
            self.on_event({
                "event": "tool_error_recovery",
                "data": self.recovery.get_error_summary(),
            })
