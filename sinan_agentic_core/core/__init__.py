"""Core agent execution components.

This module contains fundamental components for agent execution:
- BaseAgentRunner: Abstract base class for all agent runners
- Capability: Pluggable agent behavior primitive
- TurnBudget: Soft turn budget with self-extension (Capability)
- ToolErrorRecovery: Intelligent tool error tracking (Capability)
- ToolTracer: Observation-only tool-call tracer (Capability)
- recover_invalid_final_output: SDK handler that salvages malformed structured output
"""

from .base_runner import BaseAgentRunner
from .capabilities import Capability
from .errors import structured_tool_error
from .output_recovery import recover_invalid_final_output
from .tool_error_recovery import ToolErrorRecovery
from .tool_tracer import ToolTracer
from .turn_budget import TurnBudget

__all__ = [
    "BaseAgentRunner",
    "Capability",
    "recover_invalid_final_output",
    "structured_tool_error",
    "ToolErrorRecovery",
    "ToolTracer",
    "TurnBudget",
]
