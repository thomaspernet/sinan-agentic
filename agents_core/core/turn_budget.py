"""Turn budget system for soft turn management with self-extension.

Provides a flexible turn budget that agents can self-manage:
- Default budget configured per agent (soft limit the agent perceives)
- SDK max_turns set to absolute ceiling (hard safety net)
- Agent gets warned when running low on turns
- Agent can call request_extension() to approve more turns for itself

The TurnBudgetHooks class tracks turns via on_llm_start and injects
budget awareness into the agent's instructions dynamically.

Usage:
    budget = TurnBudget(default_turns=10)
    # SDK gets max_turns = budget.absolute_max (hard ceiling)
    # Agent perceives budget.default_turns (soft limit)
    # At turn 8, agent sees "2 turns remaining" in instructions
    # Agent calls request_extension() -> soft limit extends by extension_size
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from agents import RunHooks

logger = logging.getLogger(__name__)


@dataclass
class TurnBudget:
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

    default_turns: int = 10
    reminder_at: int = 2
    max_extensions: int = 3
    extension_size: int = 5
    absolute_max: int = 25

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
                return False, f"Extension denied: maximum extensions ({self.max_extensions}) reached."
            return False, f"Extension denied: would exceed absolute maximum ({self.absolute_max} turns)."

        self.extensions_used += 1
        self.extension_reasons.append(reason)
        logger.info(
            "Turn budget extended: +%d turns (extension %d/%d, reason: %s)",
            self.extension_size,
            self.extensions_used,
            self.max_extensions,
            reason,
        )
        return True, f"Extension approved. You now have {self.remaining} turns remaining (budget: {self.effective_max})."

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


class TurnBudgetHooks(RunHooks):
    """RunHooks subclass that tracks turns via on_llm_start.

    Each LLM invocation = 1 turn. The hook increments the budget's
    turn counter before each call. The dynamic instructions callable
    reads the budget state to inject awareness text.
    """

    def __init__(self, budget: TurnBudget) -> None:
        self.budget = budget

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: Optional[str],
        input_items: Any,
    ) -> None:
        """Increment turn counter before each LLM call."""
        self.budget.record_turn()
