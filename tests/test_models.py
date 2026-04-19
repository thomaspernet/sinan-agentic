"""Tests for output models and context."""

from unittest.mock import Mock

from sinan_agentic_core.models.outputs import ChatResponse, ToolOutput
from sinan_agentic_core.models.context import AgentContext


# -- ToolOutput ---------------------------------------------------------------


class TestToolOutput:
    def test_success_to_dict(self):
        out = ToolOutput(success=True, data={"key": "val"})
        d = out.to_dict()
        assert d["success"] is True
        assert d["data"] == {"key": "val"}
        assert "error" not in d

    def test_error_to_dict(self):
        out = ToolOutput(success=False, error="boom")
        d = out.to_dict()
        assert d["success"] is False
        assert d["error"] == "boom"
        assert "data" not in d

    def test_metadata_merged_into_dict(self):
        out = ToolOutput(success=True, metadata={"source": "db", "latency": 42})
        d = out.to_dict()
        assert d["source"] == "db"
        assert d["latency"] == 42

    def test_empty_metadata_not_in_dict(self):
        out = ToolOutput(success=True)
        d = out.to_dict()
        assert "metadata" not in d


# -- ChatResponse -------------------------------------------------------------


class TestChatResponse:
    def test_success_to_dict(self):
        r = ChatResponse(success=True, response="hello", session_id="s1")
        d = r.to_dict()
        assert d == {"success": True, "response": "hello", "session_id": "s1"}

    def test_with_tools_called(self):
        r = ChatResponse(success=True, tools_called=["tool_a"])
        d = r.to_dict()
        assert d["tools_called"] == ["tool_a"]

    def test_error_to_dict(self):
        r = ChatResponse(success=False, error="fail")
        d = r.to_dict()
        assert d["error"] == "fail"

    def test_usage_included_when_present(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        r = ChatResponse(success=True, usage=usage)
        d = r.to_dict()
        assert d["usage"] == usage

    def test_usage_omitted_when_none(self):
        r = ChatResponse(success=True)
        d = r.to_dict()
        assert "usage" not in d

    def test_default_values(self):
        r = ChatResponse(success=True)
        assert r.response == ""
        assert r.session_id == "default"
        assert r.tools_called == []
        assert r.error is None
        assert r.usage is None


# -- AgentContext --------------------------------------------------------------


class TestAgentContext:
    def test_has_data_false_when_empty(self, context):
        assert context.has_data is False

    def test_has_data_true_after_adding(self, context):
        context.query_results.append({"id": 1})
        assert context.has_data is True

    def test_add_query_result(self, context):
        context.add_query_result({"data": [{"name": "Alice"}, {"name": "Bob"}]})
        assert len(context.query_results) == 2
        assert context.query_results[0]["name"] == "Alice"

    def test_add_query_result_ignores_non_list_data(self, context):
        context.add_query_result({"data": "not a list"})
        assert len(context.query_results) == 0

    def test_add_query_result_ignores_missing_data_key(self, context):
        context.add_query_result({"other": [1, 2]})
        assert len(context.query_results) == 0

    def test_clear_results(self, context):
        context.query_results.append({"id": 1})
        context.discovered_data["key"] = ["val"]
        context.clear_results()
        assert context.query_results == []
        assert context.discovered_data == {}

    def test_add_discovered_item(self, context):
        context.add_discovered_item("tags", "python")
        context.add_discovered_item("tags", "async")
        assert context.get_discovered_items("tags") == ["python", "async"]

    def test_add_discovered_item_overwrites_non_list(self, context):
        context.discovered_data["key"] = "scalar"
        context.add_discovered_item("key", "new_val")
        assert context.get_discovered_items("key") == "new_val"

    def test_get_discovered_items_missing_key(self, context):
        assert context.get_discovered_items("nonexistent") is None

    def test_schema_default(self, context):
        assert context.schema == ""
