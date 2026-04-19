"""Tests for sinan_agentic_core.utils.tool_helpers."""

import json

from sinan_agentic_core.utils.tool_helpers import tool_error, tool_response, unwrap_context


class _FakeWrapper:
    """Simulates OpenAI Agents SDK RunContextWrapper."""

    def __init__(self, context: object) -> None:
        self.context = context


class _FakeContext:
    """Simulates AgentContext."""

    name = "test_context"


def test_unwrap_context_from_wrapper() -> None:
    inner = _FakeContext()
    wrapper = _FakeWrapper(inner)
    assert unwrap_context(wrapper) is inner


def test_unwrap_context_passthrough() -> None:
    ctx = _FakeContext()
    assert unwrap_context(ctx) is ctx


def test_tool_response_dict() -> None:
    result = tool_response({"key": "value", "count": 3})
    parsed = json.loads(result)
    assert parsed == {"key": "value", "count": 3}


def test_tool_response_list() -> None:
    result = tool_response([1, 2, 3])
    assert json.loads(result) == [1, 2, 3]


def test_tool_response_handles_non_serializable() -> None:
    from datetime import datetime

    dt = datetime(2026, 1, 1, 12, 0, 0)
    result = tool_response({"ts": dt})
    parsed = json.loads(result)
    assert "2026" in parsed["ts"]


def test_tool_error_simple() -> None:
    result = tool_error("Something went wrong")
    parsed = json.loads(result)
    assert parsed == {"error": "Something went wrong"}


def test_tool_error_with_extras() -> None:
    result = tool_error("Not found", document_uuid="abc-123", papers=[])
    parsed = json.loads(result)
    assert parsed["error"] == "Not found"
    assert parsed["document_uuid"] == "abc-123"
    assert parsed["papers"] == []


def test_imports_from_package() -> None:
    """Verify top-level imports work."""
    from sinan_agentic_core import tool_error, tool_response, unwrap_context

    assert callable(unwrap_context)
    assert callable(tool_response)
    assert callable(tool_error)
