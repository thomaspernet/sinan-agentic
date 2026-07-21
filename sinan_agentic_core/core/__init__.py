"""Core agent execution components.

This module contains fundamental components for agent execution:
- BaseAgentRunner: Abstract base class for all agent runners
- Capability: Pluggable agent behavior primitive
- TurnBudget: Soft turn budget with self-extension (Capability)
- ToolErrorRecovery: Intelligent tool error tracking (Capability)
- ToolTracer: Observation-only tool-call tracer (Capability)
- recover_invalid_final_output: SDK handler that salvages malformed structured output
- ModelRetryConfig: Declarative opt-in retry policy for model calls
- ToolOutputTrimConfig: Declarative opt-in trimming of oversized tool outputs
- RunErrorKind / classify_run_error: Typed classification of agent-run failures
"""

from .base_runner import BaseAgentRunner
from .capabilities import Capability
from .errors import structured_tool_error
from .model_retry import ModelRetryConfig, RetryBackoffConfig, RetryTrigger
from .output_recovery import recover_invalid_final_output
from .run_errors import RunErrorKind, classify_run_error
from .tool_error_recovery import ToolErrorRecovery
from .tool_output_trim import ToolOutputTrimConfig
from .tool_tracer import ToolTracer
from .turn_budget import TurnBudget

__all__ = [
    "BaseAgentRunner",
    "Capability",
    "classify_run_error",
    "ModelRetryConfig",
    "recover_invalid_final_output",
    "RetryBackoffConfig",
    "RetryTrigger",
    "RunErrorKind",
    "structured_tool_error",
    "ToolErrorRecovery",
    "ToolOutputTrimConfig",
    "ToolTracer",
    "TurnBudget",
]
