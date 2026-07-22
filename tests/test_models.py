"""Tests for output models and context."""

from unittest.mock import Mock

from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.models.outputs import ChatResponse, ToolOutput

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


class TestAgentContextCopiesTheCallersCollections:
    """The context owns what it was seeded with, so a run never writes into a caller's data."""

    def test_the_seeded_collections_are_readable(self):
        ctx = AgentContext(
            database_connector=Mock(),
            schema_data={"tables": ["users"]},
            query_results=[{"id": 1}],
            filters={"status": ["active"]},
            discovered_data={"tags": ["python"]},
        )

        assert ctx.schema_data == {"tables": ["users"]}
        assert ctx.query_results == [{"id": 1}]
        assert ctx.filters == {"status": ["active"]}
        assert ctx.discovered_data == {"tags": ["python"]}

    def test_a_result_added_by_the_caller_later_is_not_visible(self):
        seeded = [{"id": 1}]
        ctx = AgentContext(database_connector=Mock(), query_results=seeded)

        seeded.append({"id": 2})

        assert ctx.query_results == [{"id": 1}]

    def test_an_edit_inside_a_seeded_result_row_is_not_visible(self):
        """Rows nest inside the list, so copying only the list would not detach them."""
        seeded = [{"id": 1}]
        ctx = AgentContext(database_connector=Mock(), query_results=seeded)

        seeded[0]["id"] = 99

        assert ctx.query_results[0]["id"] == 1

    def test_results_added_by_the_run_do_not_reach_the_callers_list(self):
        seeded = [{"id": 1}]
        ctx = AgentContext(database_connector=Mock(), query_results=seeded)

        ctx.add_query_result({"data": [{"id": 2}]})

        assert seeded == [{"id": 1}]

    def test_an_item_discovered_by_the_run_does_not_reach_the_callers_dict(self):
        seeded = {"tags": ["python"]}
        ctx = AgentContext(database_connector=Mock(), discovered_data=seeded)

        ctx.add_discovered_item("authors", "ada")

        assert seeded == {"tags": ["python"]}

    def test_a_discovery_appended_by_the_run_does_not_reach_the_callers_nested_list(self):
        """add_discovered_item appends into the list under the key, one level deeper again."""
        seeded_tags = ["python"]
        ctx = AgentContext(database_connector=Mock(), discovered_data={"tags": seeded_tags})

        ctx.add_discovered_item("tags", "async")

        assert ctx.get_discovered_items("tags") == ["python", "async"]
        assert seeded_tags == ["python"]

    def test_a_filter_added_by_the_caller_later_is_not_visible(self):
        seeded = {"status": ["active"]}
        ctx = AgentContext(database_connector=Mock(), filters=seeded)

        seeded["owner"] = "someone"

        assert ctx.filters == {"status": ["active"]}

    def test_an_edit_inside_a_nested_filter_value_is_not_visible(self):
        seeded = {"status": ["active"]}
        ctx = AgentContext(database_connector=Mock(), filters=seeded)

        seeded["status"].append("archived")

        assert ctx.filters == {"status": ["active"]}

    def test_an_edit_inside_the_seeded_schema_data_is_not_visible(self):
        seeded = {"tables": {"users": ["id", "name"]}}
        ctx = AgentContext(database_connector=Mock(), schema_data=seeded)

        seeded["tables"]["users"].append("email")

        assert ctx.schema_data == {"tables": {"users": ["id", "name"]}}

    def test_an_absent_collection_stays_none(self):
        ctx = AgentContext(database_connector=Mock())

        assert ctx.schema_data is None
        assert ctx.filters is None

    def test_two_contexts_seeded_from_one_collection_do_not_share_it(self):
        seeded = [{"id": 1}]
        first = AgentContext(database_connector=Mock(), query_results=seeded)
        second = AgentContext(database_connector=Mock(), query_results=seeded)

        first.add_query_result({"data": [{"id": 2}]})

        assert second.query_results == [{"id": 1}]

    def test_the_database_connector_is_not_copied(self):
        """The connector is a live handle the run uses, not data the context accumulates into."""
        connector = Mock()

        ctx = AgentContext(database_connector=connector)

        assert ctx.database_connector is connector
