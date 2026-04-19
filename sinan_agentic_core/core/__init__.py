"""Core agent execution components.

This module contains fundamental components for agent execution:
- BaseAgentRunner: Abstract base class for all agent runners
- TurnBudget: Soft turn budget with self-extension
- TurnBudgetHooks: RunHooks subclass for turn tracking
- ToolErrorRecovery: Intelligent tool error tracking with progressive guidance
- ToolErrorRecoveryHooks: RunHooks subclass for tool error tracking
"""

from .base_runner import BaseAgentRunner
from .errors import structured_tool_error
from .tool_error_recovery import ToolErrorRecovery, ToolErrorRecoveryHooks
from .turn_budget import TurnBudget, TurnBudgetHooks

__all__ = [
    "BaseAgentRunner",
    "structured_tool_error",
    "ToolErrorRecovery",
    "ToolErrorRecoveryHooks",
    "TurnBudget",
    "TurnBudgetHooks",
]
