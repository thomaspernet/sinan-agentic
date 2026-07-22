"""Tool error recovery — intelligent error tracking with dynamic guidance.

When a tool returns an error, agents often retry with identical parameters,
wasting turns in a loop. ToolErrorRecovery solves this by:

1. Tracking tool errors and the arguments that caused them (Capability hooks)
2. Detecting repeated identical calls (same tool + same args)
3. Injecting progressive recovery guidance into the agent's instructions
   before each LLM call (via Capability.instructions)

ToolErrorRecovery is a Capability — wire it via
``AgentDefinition.capabilities`` or pass to ``execute()`` directly.

Works for all tool types: custom @function_tool, agent-as-tool, and MCP tools.

``ToolErrorRecoveryConfig`` carries the same choice declaratively. Three paths
read a declaration — the ``error_recovery:`` shorthand on an agent entry, an
``error_recovery`` entry in the explicit ``capabilities:`` list, and the
``error_recovery=`` kwarg on ``execute()`` — so the translation lives here as
``ToolErrorRecoveryConfig.build()``, with ``build_tool_error_recovery`` adding
the "off when disabled" rule the two boolean paths need. No path constructs
``ToolErrorRecovery`` itself, so the tool-registry default is decided once.

Usage:
    recovery = ToolErrorRecovery(tool_registry=registry)
    agent_def = AgentDefinition(..., capabilities=[recovery])
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from agents import RunContextWrapper, Tool
from pydantic import BaseModel, ConfigDict

from .capabilities import Capability

logger = logging.getLogger(__name__)

# Identical failures tolerated before the guidance tells the agent to stop
# retrying the tool outright.
DEFAULT_MAX_IDENTICAL_BEFORE_STOP = 3


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


class ToolErrorRecovery(Capability):
    """Track tool errors and generate recovery guidance for agents.

    Detects error patterns (repeated calls, identical arguments) and builds
    an instruction section that tells the agent what went wrong and how to
    recover. The instruction section is injected dynamically before each
    LLM call, so the agent always has up-to-date error awareness.

    Attributes:
        mcp_hints: Recovery hints for MCP tool names the registry does not know.
        max_identical_before_stop: After this many identical failures, the
            instruction section tells the agent to stop retrying entirely.

    Every field of :class:`ToolErrorRecoveryConfig` is stored as an attribute of
    the same name, which is what lets :meth:`clone` rebuild a copy from the
    declaration wholesale instead of naming each field by hand.
    """

    def __init__(
        self,
        tool_registry: Any = None,
        mcp_hints: dict[str, str] | None = None,
        max_identical_before_stop: int = DEFAULT_MAX_IDENTICAL_BEFORE_STOP,
    ) -> None:
        """Initialize the recovery tracker.

        Args:
            tool_registry: ToolRegistry instance for looking up recovery_hint.
                If None, only mcp_hints are used for hint lookup.
            mcp_hints: Mapping of MCP tool names to recovery hint strings.
                Use this for tools you don't control (external MCP servers).
                Copied, so the caller's dict is not aliased into the tracker
                and no two trackers built from it end up sharing one mapping.
            max_identical_before_stop: After N identical failures, instruct
                the agent to stop retrying. Defaults to
                :data:`DEFAULT_MAX_IDENTICAL_BEFORE_STOP`.
        """
        self._registry = tool_registry
        self._errors: dict[str, ToolErrorEntry] = {}  # key: tool_name
        self._last_args: dict[str, str] = {}  # tool_name -> last args_hash
        self._pending_args: dict[str, str] = {}  # tool_name -> in-flight args
        self.mcp_hints = dict(mcp_hints or {})
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

        Called from on_tool_end(). Parses the result
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
            e.identical_count >= self.max_identical_before_stop for e in self._errors.values()
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
        self._pending_args.clear()

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        """Capability hook — return current recovery guidance, or None when clean."""
        return self.build_instruction_section() or None

    def on_tool_start(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        args: str,
    ) -> None:
        """Capability hook — capture tool arguments before execution."""
        tool_name = getattr(tool, "name", str(tool))
        self._pending_args[tool_name] = args

    def on_tool_end(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        result: str,
    ) -> None:
        """Capability hook — record the tool result and emit any recovery event."""
        tool_name = getattr(tool, "name", str(tool))
        arguments = self._pending_args.pop(tool_name, "")
        self.record_tool_result(tool_name, result, arguments)
        if self.on_event and self.has_errors:
            self.on_event(
                {
                    "event": "tool_error_recovery",
                    "data": self.get_error_summary(),
                }
            )

    def clone(self) -> "ToolErrorRecovery":
        """Return a fresh ``ToolErrorRecovery`` with the same configuration and zeroed state.

        The declared fields are read back off :class:`ToolErrorRecoveryConfig`
        and forwarded wholesale, mirroring :meth:`ToolErrorRecoveryConfig.build`
        in the other direction, so a field added to that model survives cloning
        without a second edit here. ``tool_registry`` is passed separately for
        the same reason it is not a config field: it is a live object rather
        than something a declaration can carry.
        """
        declared = {name: getattr(self, name) for name in ToolErrorRecoveryConfig.model_fields}
        return ToolErrorRecovery(tool_registry=self._registry, **declared)

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize tracked errors and last-args map so retries survive restarts."""
        return {
            "errors": {
                name: {
                    "tool_name": entry.tool_name,
                    "error": entry.error,
                    "args_hash": entry.args_hash,
                    "call_count": entry.call_count,
                    "identical_count": entry.identical_count,
                    "recovery_hint": entry.recovery_hint,
                    "last_args_summary": entry.last_args_summary,
                }
                for name, entry in self._errors.items()
            },
            "last_args": dict(self._last_args),
        }

    def from_snapshot(self, data: dict[str, Any]) -> None:
        """Restore tracked errors from a previous snapshot.

        ``_pending_args`` is intentionally left empty: it tracks in-flight
        tool calls that did not survive the process boundary.
        """
        raw_errors = data.get("errors", {})
        self._errors = {}
        if isinstance(raw_errors, dict):
            for name, payload in raw_errors.items():
                if not isinstance(payload, dict):
                    continue
                self._errors[str(name)] = ToolErrorEntry(
                    tool_name=str(payload.get("tool_name", name)),
                    error=str(payload.get("error", "")),
                    args_hash=str(payload.get("args_hash", "")),
                    call_count=int(payload.get("call_count", 1)),
                    identical_count=int(payload.get("identical_count", 1)),
                    recovery_hint=str(payload.get("recovery_hint", "")),
                    last_args_summary=str(payload.get("last_args_summary", "")),
                )
        raw_last = data.get("last_args", {})
        self._last_args = (
            {str(k): str(v) for k, v in raw_last.items()} if isinstance(raw_last, dict) else {}
        )
        self._pending_args = {}

    # ------------------------------------------------------------------ #
    # Instruction section builders (progressive escalation)
    # ------------------------------------------------------------------ #

    def _section_first(self, entry: ToolErrorEntry) -> str:
        """First failure: show error and hint."""
        args_info = f" (called with: {entry.last_args_summary})" if entry.last_args_summary else ""
        line = f'- {entry.tool_name} returned error: "{entry.error}"{args_info}'
        if entry.recovery_hint:
            line += f"\n  Recovery hint: {entry.recovery_hint}"
        return line

    def _section_repeated(self, entry: ToolErrorEntry) -> str:
        """Repeated failure with same args: warn strongly."""
        line = (
            f"- {entry.tool_name} has FAILED {entry.identical_count} TIMES "
            f'with identical arguments. Error: "{entry.error}"\n'
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
            f'  Error was: "{entry.error}"\n'
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
                hint: str = tool_def.recovery_hint
                return hint
        return self.mcp_hints.get(tool_name, "")

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


class ToolErrorRecoveryConfig(BaseModel):
    """Opt-in tool error recovery for one agent.

    Every field is optional, so declaring the capability with no config opts in
    with the defaults below. The tool registry is not a field here: it is a live
    object rather than something YAML can express, so it is passed to
    :meth:`build` by whichever path resolved the declaration.

    Declared in ``agents.yaml``::

        agents:
          research_agent:
            model: gpt-4o
            description: Deep research
            error_recovery: true      # shorthand — defaults, nothing to tune
            capabilities:
              - name: error_recovery  # explicit — same capability, tuned
                config:
                  max_identical_before_stop: 5
                  mcp_hints:
                    mcp_arxiv_search: Query must be at least 2 characters.
    """

    # An unknown key is a typo or a field that belongs elsewhere
    # (``tool_registry`` is supplied by the caller, not declared), and this model
    # is the only gate the declaration passes through — so reject it rather than
    # drop it silently.
    model_config = ConfigDict(extra="forbid")

    mcp_hints: dict[str, str] = {}
    max_identical_before_stop: int = DEFAULT_MAX_IDENTICAL_BEFORE_STOP

    def build(self, tool_registry: Any = None) -> ToolErrorRecovery:
        """Translate this config into a runtime :class:`ToolErrorRecovery`.

        Fields are forwarded wholesale rather than named one by one, so a field
        added to this model reaches the built capability without editing this
        method.

        Args:
            tool_registry: Registry consulted for each tool's ``recovery_hint``.
                Falls back to the process-wide registry, so a declaration that
                names no registry still resolves hints.
        """
        # Imported here rather than at module scope: ``registry`` imports this
        # module, so a module-level import would close the cycle.
        from ..registry.tool_registry import get_tool_registry

        return ToolErrorRecovery(
            **self.model_dump(),
            tool_registry=tool_registry if tool_registry is not None else get_tool_registry(),
        )


def build_tool_error_recovery(
    enabled: bool,
    tool_registry: Any = None,
) -> ToolErrorRecovery | None:
    """Translate a declared error-recovery flag into the capability a run installs.

    Callers whose declaration is a boolean go through this rather than repeating
    the check — the ``error_recovery:`` shorthand on an agent entry, and the
    ``error_recovery=`` kwarg on ``execute()`` — so the "off when disabled" rule
    lives here instead of once per caller. The explicit ``capabilities:`` list
    has no disabled case (the entry *is* the declaration) and calls
    :meth:`ToolErrorRecoveryConfig.build` directly.

    Args:
        enabled: Whether the agent declared recovery on.
        tool_registry: Registry consulted for each tool's ``recovery_hint``.
            Falls back to the process-wide registry.

    Returns:
        The configured capability, or None when recovery is off. Each call builds
        a fresh capability, so no two agents share one set of tracked errors.
    """
    if not enabled:
        return None

    return ToolErrorRecoveryConfig().build(tool_registry)
