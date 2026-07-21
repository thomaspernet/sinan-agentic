"""Public accessors for the active turn budget carried on the run context.

The runner carries the active ``TurnBudget`` on the run context so that
instruction builders and the ``request_extension`` tool can locate it without
knowing the storage detail. Every write and read goes through these accessors,
so the context attribute name is defined here and nowhere else.

These accessors operate on the raw (unwrapped) run context. Callers holding a
``RunContextWrapper`` unwrap it first (see ``utils.tool_helpers.unwrap_context``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.turn_budget import TurnBudget

_TURN_BUDGET_ATTR = "turn_budget"


def set_turn_budget(context: Any, budget: TurnBudget) -> None:
    """Carry the active turn budget on the run context."""
    setattr(context, _TURN_BUDGET_ATTR, budget)


def get_turn_budget(context: Any) -> TurnBudget | None:
    """Return the active turn budget carried on the run context, or None."""
    return getattr(context, _TURN_BUDGET_ATTR, None)
