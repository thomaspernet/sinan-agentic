"""Tests for AgentOrchestrator."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sinan_agentic_core.registry.agent_registry import AgentRegistry
from sinan_agentic_core.registry.guardrail_registry import GuardrailRegistry
from sinan_agentic_core.registry.tool_registry import ToolRegistry


@pytest.fixture
def orchestrator():
    """Build an AgentOrchestrator with empty registries."""
    with (
        patch("sinan_agentic_core.core.base_runner.get_agent_registry", return_value=AgentRegistry()),
        patch("sinan_agentic_core.core.base_runner.get_tool_registry", return_value=ToolRegistry()),
        patch(
            "sinan_agentic_core.core.base_runner.get_guardrail_registry",
            return_value=GuardrailRegistry(),
        ),
    ):
        from sinan_agentic_core.orchestrator import AgentOrchestrator

        return AgentOrchestrator()


class TestOrchestratorRunWorkflow:
    async def test_success(self, orchestrator):
        usage = {
            "requests": 1,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

        with patch.object(orchestrator, "run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"output": "orchestrated answer", "usage": usage}

            result = await orchestrator.run_workflow(
                user_query="test query",
                context_data={"database_connector": Mock()},
                session_id="orch-session",
            )

        assert result["success"] is True
        assert result["result"] == "orchestrated answer"
        assert result["usage"] == usage
        assert result["session_id"] == "orch-session"

    async def test_error(self, orchestrator):
        with patch.object(orchestrator, "run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("Agent exploded")

            result = await orchestrator.run_workflow(
                user_query="bad query",
                context_data={"database_connector": Mock()},
            )

        assert result["success"] is False
        assert "Agent exploded" in result["error"]
