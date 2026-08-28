"""Turn budget system for soft turn management with self-extension.

Provides a flexible turn budget that agents can self-manage:
- Default budget configured per agent (soft limit the agent perceives)
- SDK max_turns set to absolute ceiling (hard safety net)
- Agent gets warned when running low on turns
- Agent can call request_extension() to approve more turns for itself

TurnBudget is a Capability: the runtime calls ``on_llm_start`` to count
turns and ``instructions`` to inject budget awareness before each LLM call.

``TurnBudgetConfig`` carries the same choice declaratively in ``agents.yaml``.
Two paths read that declaration — the ``turn_budget:`` shorthand on an agent
entry and a ``turn_budget`` entry in the explicit ``capabilities:`` list — so
the translation lives here as ``TurnBudgetConfig.build()``, with
``build_turn_budget`` adding the "off unless declared" rule the shorthand path
needs. Neither path unpacks the config field by field, so a field added to the
model reaches the built budget without a second edit. Both field lists read
their defaults from the same module constants, so what an agent gets by
declaring nothing and what it gets by declaring an empty block is one value
rather than two copies free to drift.

Usage:
    budget = TurnBudget(default_turns=10)
    # Wire via AgentDefinition.capabilities=[budget] or pass to execute().
    # SDK gets max_turns = budget.absolute_max (hard ceiling)
    # Agent perceives budget.default_turns (soft limit)
    # At turn 8, agent sees "2 turns remaining" in instructions
    # Agent calls request_extension() -> soft limit extends by extension_size
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agents import RunContextWrapper, Tool
from pydantic import BaseModel, ConfigDict

from .capabilities import Capability
from .turn_budget_tool import request_extension_tool

logger = logging.getLogger(__name__)

# Soft-budget defaults, named once and read by both the runtime ``TurnBudget``
# below and the ``TurnBudgetConfig`` that declares it, so the two field lists
# cannot drift to different numbers.
DEFAULT_TURNS = 10
DEFAULT_REMINDER_AT = 2
DEFAULT_MAX_EXTENSIONS = 3
DEFAULT_EXTENSION_SIZE = 5

# The hard SDK ceiling an agent runs under when it declares no ``max_turns``.
# High enough that the soft budget, not this, is what the agent negotiates with.
# Not one of the four above: it is not a ``TurnBudgetConfig`` field, because in
# YAML the ceiling is the agent's own ``max_turns``.
DEFAULT_ABSOLUTE_MAX_TURNS = 25


@dataclass
class TurnBudget(Capability):
    """Soft turn budget with self-extension capability.

    The agent perceives `effective_max` as its budget. The SDK's hard
    `max_turns` is set to `absolute_max` — a safety ceiling the agent
    never knows about.

    Attributes:
        default_turns: Initial turn budget the agent perceives.
        reminder_at: Warn agent when this many turns remain.
        max_extensions: How many times the agent can self-extend.
        extension_size: Turns added per extension.
        absolute_max: Hard ceiling passed to SDK (never exceeded).
    """

    default_turns: int = DEFAULT_TURNS
    reminder_at: int = DEFAULT_REMINDER_AT
    max_extensions: int = DEFAULT_MAX_EXTENSIONS
    extension_size: int = DEFAULT_EXTENSION_SIZE
    absolute_max: int = DEFAULT_ABSOLUTE_MAX_TURNS

    # Mutable state — tracked during execution
    turns_used: int = field(default=0, init=False)
    extensions_used: int = field(default=0, init=False)
    extension_reasons: list[str] = field(default_factory=list, init=False)

    @property
    def effective_max(self) -> int:
        """Current perceived budget (default + extensions granted)."""
        return self.default_turns + (self.extensions_used * self.extension_size)

    @property
    def remaining(self) -> int:
        """Turns remaining in the current soft budget."""
        return max(0, self.effective_max - self.turns_used)

    @property
    def is_warning(self) -> bool:
        """True when remaining turns <= reminder_at threshold."""
        return self.remaining <= self.reminder_at and self.remaining > 0

    @property
    def is_exhausted(self) -> bool:
        """True when soft budget is fully used."""
        return self.remaining <= 0

    @property
    def can_extend(self) -> bool:
        """True if the agent can still request extensions."""
        if self.extensions_used >= self.max_extensions:
            return False
        projected = self.effective_max + self.extension_size
        return projected <= self.absolute_max

    def request_extension(self, reason: str) -> tuple[bool, str]:
        """Request additional turns. Returns (success, message)."""
        if not self.can_extend:
            if self.extensions_used >= self.max_extensions:
                return (
                    False,
                    f"Extension denied: maximum extensions ({self.max_extensions}) reached.",
                )
            return (
                False,
                f"Extension denied: would exceed absolute maximum ({self.absolute_max} turns).",
            )

        self.extensions_used += 1
        self.extension_reasons.append(reason)
        logger.info(
            "Turn budget extended: +%d turns (extension %d/%d, reason: %s)",
            self.extension_size,
            self.extensions_used,
            self.max_extensions,
            reason,
        )
        return (
            True,
            f"Extension approved. You now have {self.remaining} turns remaining (budget: {self.effective_max}).",
        )

    def record_turn(self) -> None:
        """Record that a turn was used."""
        self.turns_used += 1
        logger.debug(
            "Turn %d/%d used (absolute ceiling: %d)",
            self.turns_used,
            self.effective_max,
            self.absolute_max,
        )

    def build_instruction_section(self) -> str:
        """Build the turn budget instruction text for the agent.

        Returns budget awareness text appropriate to the current state:
        initial planning, normal status, warning, or exhausted.
        """
        if self.turns_used == 0:
            return self._section_initial()
        if self.is_exhausted:
            return self._section_exhausted()
        if self.is_warning:
            return self._section_warning()
        return f"Turn budget: {self.remaining} of {self.effective_max} turns remaining."

    def _section_initial(self) -> str:
        return (
            f"Turn budget: You have {self.effective_max} turns for this task. "
            f"Plan your work accordingly."
        )

    def _section_exhausted(self) -> str:
        base = f"Turn budget EXHAUSTED ({self.turns_used}/{self.effective_max} used). "
        if self.can_extend:
            return base + (
                "You must call request_extension with a reason to continue, "
                "or wrap up immediately with whatever results you have."
            )
        return base + "No extensions remaining. Wrap up NOW with whatever results you have."

    def _section_warning(self) -> str:
        base = f"Turn budget: {self.remaining} turn(s) remaining out of {self.effective_max}. "
        if self.can_extend:
            return base + (
                "Either wrap up with a complete response, or call request_extension "
                "if the task genuinely needs more work."
            )
        return base + "No extensions available. Wrap up now with a complete response."

    def reset(self) -> None:
        """Reset mutable state for reuse."""
        self.turns_used = 0
        self.extensions_used = 0
        self.extension_reasons.clear()

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        """Capability hook — return the current budget steering fragment."""
        return self.build_instruction_section()

    def on_llm_start(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        system_prompt: str | None,
        input_items: Any,
    ) -> None:
        """Capability hook — count this turn and emit a streaming event."""
        self.record_turn()
        if self.on_event:
            self.on_event(
                {
                    "event": "turn_budget",
                    "data": {
                        "turns_used": self.turns_used,
                        "effective_max": self.effective_max,
                        "remaining": self.remaining,
                        "extensions_used": self.extensions_used,
                        "is_warning": self.is_warning,
                    },
                }
            )

    def tools(self) -> list[Tool]:
        """Expose request_extension so the agent can self-extend its budget."""
        return [request_extension_tool]

    def clone(self) -> "TurnBudget":
        """Return a fresh ``TurnBudget`` with the same configuration and zeroed counters."""
        return TurnBudget(
            default_turns=self.default_turns,
            reminder_at=self.reminder_at,
            max_extensions=self.max_extensions,
            extension_size=self.extension_size,
            absolute_max=self.absolute_max,
        )

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize counters so a future session can resume mid-budget."""
        return {
            "turns_used": self.turns_used,
            "extensions_used": self.extensions_used,
            "extension_reasons": list(self.extension_reasons),
        }

    def from_snapshot(self, data: dict[str, Any]) -> None:
        """Restore counters from a previous snapshot.

        Tolerates missing keys so older snapshots stay readable.
        """
        self.turns_used = int(data.get("turns_used", 0))
        self.extensions_used = int(data.get("extensions_used", 0))
        reasons = data.get("extension_reasons", [])
        self.extension_reasons = [str(r) for r in reasons] if isinstance(reasons, list) else []


class TurnBudgetConfig(BaseModel):
    """Opt-in soft turn budget for one agent.

    Every field is optional, so declaring the key with no fields
    (``turn_budget: {}``) opts in with the defaults below — only a missing key
    means the agent opts out.

    The hard ceiling is not a field here: it is the agent's own ``max_turns``,
    passed to :meth:`build` by whichever path resolved the declaration.

    Declared in ``agents.yaml``::

        agents:
          research_agent:
            model: gpt-4o
            description: Deep research
            max_turns: 30          # hard ceiling handed to the SDK
            turn_budget:
              default_turns: 15    # soft budget the agent perceives
              reminder_at: 3
              max_extensions: 2
              extension_size: 5
    """

    # An unknown key is a typo or a field that belongs elsewhere (``absolute_max``
    # is the agent's ``max_turns``), and this model is the only gate both
    # declaration paths pass through — so reject it rather than drop it silently.
    model_config = ConfigDict(extra="forbid")

    default_turns: int = DEFAULT_TURNS
    reminder_at: int = DEFAULT_REMINDER_AT
    max_extensions: int = DEFAULT_MAX_EXTENSIONS
    extension_size: int = DEFAULT_EXTENSION_SIZE

    def build(self, absolute_max: int | None = None) -> TurnBudget:
        """Translate this config into a runtime :class:`TurnBudget`.

        Fields are forwarded wholesale rather than named one by one, so a field
        added to this model reaches the built budget without editing this method.

        Args:
            absolute_max: The agent's hard turn ceiling. Falls back to
                :data:`DEFAULT_ABSOLUTE_MAX_TURNS` when the agent declares none.
        """
        return TurnBudget(
            **self.model_dump(),
            absolute_max=absolute_max or DEFAULT_ABSOLUTE_MAX_TURNS,
        )


def build_turn_budget(
    turn_budget: TurnBudgetConfig | None,
    absolute_max: int | None = None,
) -> TurnBudget | None:
    """Translate a declared turn budget into the capability a run installs.

    Callers that resolve a budget which may be absent go through this rather
    than repeating the check — today ``AgentYamlEntry.build_turn_budget()``, for
    the optional ``turn_budget:`` shorthand — so the "off unless declared" rule
    lives here instead of once per caller. The explicit ``capabilities:`` list
    has no absent case (the entry *is* the declaration) and calls
    :meth:`TurnBudgetConfig.build` directly.

    Args:
        turn_budget: The agent's declared budget, or None when it opts out.
        absolute_max: The agent's hard turn ceiling. Falls back to
            :data:`DEFAULT_ABSOLUTE_MAX_TURNS` when the agent declares none.

    Returns:
        The configured budget, or None when nothing is declared. Each call builds
        a fresh budget, so no two agents share one set of counters.
    """
    if turn_budget is None:
        return None

    return turn_budget.build(absolute_max)
