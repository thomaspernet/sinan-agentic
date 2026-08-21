"""Generic Agent Orchestrator - Code orchestration pattern.

This orchestrator manages multi-agent workflows using code orchestration.
Adapt this template for your specific use case.
"""

import logging
from collections.abc import Callable
from typing import Any

from .core import BaseAgentRunner
from .core.run_errors import run_error_payload

logger = logging.getLogger(__name__)


class AgentOrchestrator(BaseAgentRunner):
    """Generic orchestrator using code orchestration pattern.

    This class demonstrates the pattern of:
    1. Initialize session and context
    2. Run agents in sequence (or based on routing logic)
    3. Accumulate results in context
    4. Return final output

    Extends BaseAgentRunner to reuse agent creation and execution logic.

    Usage:
        orchestrator = AgentOrchestrator()
        result = await orchestrator.run_workflow(
            user_query="Analyze sales data",
            context_data={"database_connector": db}
        )
    """

    def __init__(self) -> None:
        # Initialize base class (loads registries and builds mappings)
        super().__init__()

    async def run_workflow(
        self,
        user_query: str,
        context_data: dict[str, Any],
        session_id: str | None = None,
        initial_history: list[Any] | None = None,
        event_callback: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run the orchestrator workflow.

        Args:
            user_query: User's input query
            context_data: Initial context data (database connector, filters, etc.)
            session_id: Optional session ID for conversation continuity
            initial_history: Optional conversation history
            event_callback: Optional callback for streaming events

        Returns:
            ``{"success": True, "result": ..., "usage": ..., "session_id": ...}``
            on success, or ``{"success": False, "error": str, "error_kind": str,
            "session_id": ...}`` on failure. ``error_kind`` is a
            ``RunErrorKind`` value naming why the run failed; a guardrail
            tripwire adds a ``guardrail`` entry naming the check that rejected
            the run.
        """
        # 1. Setup session and context using base class methods
        session = self.setup_session(session_id=session_id, initial_history=initial_history)
        context = self.setup_context(**context_data)

        # Add user query to session
        await session.add_items([{"role": "user", "content": user_query}])

        try:
            # Example: Run a single agent using base class method
            result = await self.run_agent(
                agent_name="your_agent_name", session=session, context=context
            )

            return {
                "success": True,
                "result": result["output"],
                "usage": result["usage"],
                "session_id": session.session_id,
            }

        except Exception as e:
            payload = run_error_payload(e)
            logger.error("Orchestration failed (%s): %s", payload["error_kind"], e, exc_info=True)
            return {"success": False, **payload, "session_id": session.session_id}
