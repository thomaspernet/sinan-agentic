"""Shared fixtures for sinan_agentic_core tests."""

from unittest.mock import Mock

import pytest
from agents import Usage

from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.session.agent_session import AgentSession, ConversationHistory


@pytest.fixture
def session():
    """Create a fresh AgentSession."""
    return AgentSession(session_id="test-session")


@pytest.fixture
def context():
    """Create a fresh AgentContext with a mock connector."""
    return AgentContext(database_connector=Mock())


@pytest.fixture
def conversation_history():
    """Create a ConversationHistory with sample messages."""
    h = ConversationHistory()
    h.add_message("user", "Hello")
    h.add_message("assistant", "Hi there!")
    return h


@pytest.fixture
def sample_usage():
    """Create a real Usage object from the SDK."""
    return Usage(
        requests=1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )


@pytest.fixture
def mock_run_result(sample_usage):
    """Create a mock RunResult with raw_responses containing real Usage."""
    response = Mock()
    response.usage = sample_usage

    result = Mock()
    result.raw_responses = [response]
    result.final_output = "Test response"
    return result
