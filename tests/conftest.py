"""Shared fixtures and helpers for sinan_agentic_core tests."""

from dataclasses import fields
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from unittest.mock import Mock

import httpx
import pytest
from agents import Usage
from openai import BadRequestError

from sinan_agentic_core.core.run_errors import CONTEXT_OVERFLOW_ERROR_CODE
from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.session.agent_session import AgentSession, ConversationHistory


def make_context_overflow_error(
    message: str = "This model's maximum context length is 8192 tokens.",
) -> BadRequestError:
    """Build the provider error a real context overflow raises.

    Context overflow has no SDK exception class -- it surfaces as the
    provider's HTTP 400 with ``context_length_exceeded`` in the error body's
    ``code`` field. Tests classify against that shape, not against a message.
    """
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return BadRequestError(
        message,
        response=httpx.Response(400, request=request),
        body={
            "message": message,
            "type": "invalid_request_error",
            "code": CONTEXT_OVERFLOW_ERROR_CODE,
        },
    )


def _declared_members(annotation: object) -> tuple[object, ...]:
    """The members of an optional annotation, or the annotation itself when it is not one."""
    if get_origin(annotation) in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def collection_field_names(cls: type) -> list[str]:
    """Field names declared as a ``dict[...]``/``list[...]``, optional or not.

    Selecting on the declared type excludes a non-collection field
    (``database_connector: Any``, ``model: str``) without naming it, and — unlike
    ``default_factory is list`` — still sees a collection field that defaults to
    ``None`` rather than to an empty container. A drift guard built on it
    therefore does not need editing when either kind of field is added.
    """
    hints = get_type_hints(cls)
    return [
        f.name
        for f in fields(cls)
        if any(get_origin(m) in (dict, list) for m in _declared_members(hints[f.name]))
    ]


def edit_every_level(seeded: dict[str, Any] | list[Any]) -> None:
    """Edit the container and the values nested in it — a shallow copy detaches only the outer."""
    for value in seeded.values() if isinstance(seeded, dict) else seeded:
        if isinstance(value, list):
            value.append("added_late")
        elif isinstance(value, dict):
            value["added_late"] = True

    if isinstance(seeded, dict):
        seeded["added_late"] = True
    else:
        seeded.append({"added_late": True})


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
