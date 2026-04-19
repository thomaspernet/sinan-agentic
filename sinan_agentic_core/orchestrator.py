"""Generic Agent Orchestrator - Code orchestration pattern.

This orchestrator manages multi-agent workflows using code orchestration.
Adapt this template for your specific use case.
"""

import logging
from typing import Optional, Dict, Any

from .core import BaseAgentRunner
from .services.events import StreamingHelper

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

    def __init__(self):
        # Initialize base class (loads registries and builds mappings)
        super().__init__()

    async def run_workflow(
        self,
        user_query: str,
        context_data: Dict[str, Any],
        session_id: Optional[str] = None,
        initial_history: Optional[list] = None,
        event_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Run the orchestrator workflow.

        Args:
            user_query: User's input query
            context_data: Initial context data (database connector, filters, etc.)
            session_id: Optional session ID for conversation continuity
            initial_history: Optional conversation history
            event_callback: Optional callback for streaming events

        Returns:
            Dict with workflow results
        """
        # 1. Setup session and context using base class methods
        session = self.setup_session(session_id=session_id, initial_history=initial_history)
        context = self.setup_context(**context_data)
        streaming = StreamingHelper(event_callback)

        # Add user query to session
        await session.add_items([{
            "role": "user",
            "content": user_query
        }])
        
        try:
            # Example: Run a single agent using base class method
            result = await self.run_agent(
                agent_name="your_agent_name",
                session=session,
                context=context
            )
            
            return {
                "success": True,
                "result": result["output"],
                "usage": result["usage"],
                "session_id": session.session_id,
            }
        
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session.session_id
            }

