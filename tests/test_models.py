"""Tests for output models and context."""

import copy
from typing import Any
from unittest.mock import Mock

from sinan_agentic_core.models.context import AgentContext
from sinan_agentic_core.models.outputs import ChatResponse, ToolOutput
from tests.conftest import collection_field_names, edit_every_level

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


class TestToolOutputOwnsItsMappings:
    """The result is a fixed record of what the tool produced, at both ends."""

    def test_the_supplied_mappings_are_readable(self):
        out = ToolOutput(
            success=True,
            data={"rows": [{"id": 1}]},
            metadata={"source": "db"},
        )

        assert out.data == {"rows": [{"id": 1}]}
        assert out.metadata == {"source": "db"}

    def test_an_edit_by_the_tool_after_construction_is_not_visible(self):
        data = {"rows": [{"id": 1}]}
        metadata = {"source": "db"}
        out = ToolOutput(success=True, data=data, metadata=metadata)

        data["rows_added_late"] = []
        metadata["latency"] = 42

        assert out.data == {"rows": [{"id": 1}]}
        assert out.metadata == {"source": "db"}

    def test_an_edit_inside_a_nested_value_is_not_visible(self):
        """Values nest inside both mappings, so copying only the outer would not detach them."""
        data = {"rows": [{"id": 1}]}
        metadata = {"trace": {"attempts": 1}}
        out = ToolOutput(success=True, data=data, metadata=metadata)

        data["rows"][0]["id"] = 99
        metadata["trace"]["attempts"] = 99

        assert out.data == {"rows": [{"id": 1}]}
        assert out.metadata == {"trace": {"attempts": 1}}

    def test_two_results_built_from_one_mapping_do_not_share_it(self):
        data = {"rows": [{"id": 1}]}
        first = ToolOutput(success=True, data=data)
        second = ToolOutput(success=True, data=data)

        first.data["rows"].append({"id": 2})

        assert second.data == {"rows": [{"id": 1}]}

    def test_an_absent_data_mapping_stays_none(self):
        out = ToolOutput(success=True)

        assert out.data is None

    def test_a_consumer_edit_to_the_returned_data_does_not_reach_the_result(self):
        out = ToolOutput(success=True, data={"rows": [{"id": 1}]})

        payload = out.to_dict()
        payload["data"]["added_late"] = True
        payload["data"]["rows"][0]["id"] = 99

        assert out.data == {"rows": [{"id": 1}]}

    def test_a_consumer_edit_to_a_merged_metadata_value_does_not_reach_the_result(self):
        out = ToolOutput(success=True, metadata={"trace": {"attempts": 1}})

        payload = out.to_dict()
        payload["trace"]["attempts"] = 99

        assert out.metadata == {"trace": {"attempts": 1}}

    def test_two_payloads_from_one_result_do_not_share_a_data_mapping(self):
        out = ToolOutput(success=True, data={"rows": [{"id": 1}]})

        first = out.to_dict()
        second = out.to_dict()
        first["data"]["rows"].append({"id": 2})

        assert second["data"] == {"rows": [{"id": 1}]}

    def test_every_collection_field_is_detached_from_the_caller(self):
        """A collection field added later without a matching copy fails here, rather than drifting."""
        seeds: dict[str, Any] = {
            "data": {"rows": [{"id": 1}]},
            "metadata": {"trace": {"attempts": 1}},
        }
        assert set(collection_field_names(ToolOutput)) == set(seeds), (
            "ToolOutput gained or lost a collection field — seed it here "
            "and copy it in __post_init__"
        )
        seeded_as_supplied = copy.deepcopy(seeds)

        out = ToolOutput(success=True, **seeds)
        for seeded in seeds.values():
            edit_every_level(seeded)

        for name, as_supplied in seeded_as_supplied.items():
            assert (
                getattr(out, name) == as_supplied
            ), f"{name} is aliased to the caller's collection"


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


class TestChatResponseOwnsItsCollections:
    """The response is a fixed record of one turn, at both ends."""

    def test_the_supplied_collections_are_readable(self):
        r = ChatResponse(
            success=True,
            tools_called=["tool_a"],
            usage={"input_tokens": 100, "input_tokens_details": {"cached_tokens": 10}},
        )

        assert r.tools_called == ["tool_a"]
        assert r.usage == {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 10},
        }

    def test_a_tool_recorded_after_the_turn_is_not_visible(self):
        """The natural source is a live hooks list that keeps growing for the rest of the run."""
        tools_called = ["tool_a"]
        r = ChatResponse(success=True, tools_called=tools_called)

        tools_called.append("tool_b")

        assert r.tools_called == ["tool_a"]

    def test_a_later_write_to_the_usage_record_is_not_visible(self):
        """The natural source is the runner attribute the next turn overwrites."""
        usage = {"input_tokens": 100}
        r = ChatResponse(success=True, usage=usage)

        usage["input_tokens"] = 999

        assert r.usage == {"input_tokens": 100}

    def test_an_edit_inside_the_usage_details_is_not_visible(self):
        """The usage record nests its token details, so copying only the outer misses them."""
        usage = {"input_tokens_details": {"cached_tokens": 10}}
        r = ChatResponse(success=True, usage=usage)

        usage["input_tokens_details"]["cached_tokens"] = 999

        assert r.usage == {"input_tokens_details": {"cached_tokens": 10}}

    def test_two_responses_built_from_one_list_do_not_share_it(self):
        tools_called = ["tool_a"]
        first = ChatResponse(success=True, tools_called=tools_called)
        second = ChatResponse(success=True, tools_called=tools_called)

        first.tools_called.append("tool_b")

        assert second.tools_called == ["tool_a"]

    def test_an_absent_usage_record_stays_none(self):
        r = ChatResponse(success=True)

        assert r.usage is None

    def test_a_consumer_edit_to_the_returned_tools_called_does_not_reach_the_response(self):
        r = ChatResponse(success=True, tools_called=["tool_a"])

        payload = r.to_dict()
        payload["tools_called"].append("tool_b")

        assert r.tools_called == ["tool_a"]

    def test_a_consumer_edit_inside_the_returned_usage_does_not_reach_the_response(self):
        r = ChatResponse(
            success=True,
            usage={"input_tokens": 100, "input_tokens_details": {"cached_tokens": 10}},
        )

        payload = r.to_dict()
        payload["usage"]["input_tokens"] = 999
        payload["usage"]["input_tokens_details"]["cached_tokens"] = 999

        assert r.usage == {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 10},
        }

    def test_two_payloads_from_one_response_do_not_share_a_usage_record(self):
        r = ChatResponse(success=True, usage={"input_tokens_details": {"cached_tokens": 10}})

        first = r.to_dict()
        second = r.to_dict()
        first["usage"]["input_tokens_details"]["cached_tokens"] = 999

        assert second["usage"] == {"input_tokens_details": {"cached_tokens": 10}}

    def test_every_collection_field_is_detached_from_the_caller(self):
        """A collection field added later without a matching copy fails here, rather than drifting."""
        seeds: dict[str, Any] = {
            "tools_called": ["tool_a"],
            "usage": {"input_tokens_details": {"cached_tokens": 10}},
        }
        assert set(collection_field_names(ChatResponse)) == set(seeds), (
            "ChatResponse gained or lost a collection field — seed it here "
            "and copy it in __post_init__"
        )
        seeded_as_supplied = copy.deepcopy(seeds)

        r = ChatResponse(success=True, **seeds)
        for seeded in seeds.values():
            edit_every_level(seeded)

        for name, as_supplied in seeded_as_supplied.items():
            assert getattr(r, name) == as_supplied, f"{name} is aliased to the caller's collection"


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

    def test_every_collection_field_is_detached_from_the_caller(self):
        """A collection field added later without a matching copy fails here, rather than drifting."""
        seeds: dict[str, Any] = {
            "schema_data": {"tables": ["users"]},
            "query_results": [{"id": 1}],
            "filters": {"status": ["active"]},
            "discovered_data": {"tags": ["python"]},
        }
        assert set(collection_field_names(AgentContext)) == set(seeds), (
            "AgentContext gained or lost a collection field — seed it here "
            "and copy it in __post_init__"
        )
        seeded_as_supplied = copy.deepcopy(seeds)

        ctx = AgentContext(database_connector=Mock(), **seeds)
        for seeded in seeds.values():
            edit_every_level(seeded)

        for name, as_supplied in seeded_as_supplied.items():
            assert (
                getattr(ctx, name) == as_supplied
            ), f"{name} is aliased to the caller's collection"


class TestAgentContextOwnsWhatItCollects:
    """The accumulators copy on the way in, so ownership covers a whole run, not just its seed."""

    def test_the_collected_rows_are_readable(self, context):
        context.add_query_result({"data": [{"name": "Alice"}, {"name": "Bob"}]})

        assert context.query_results == [{"name": "Alice"}, {"name": "Bob"}]

    def test_a_later_edit_to_the_callers_payload_does_not_reach_the_context(self, context):
        rows = [{"id": 1, "tags": ["python"]}]

        context.add_query_result({"data": rows})
        edit_every_level(rows)

        assert context.query_results == [{"id": 1, "tags": ["python"]}]

    def test_an_edit_inside_a_collected_row_does_not_reach_the_context(self, context):
        """A row nests further values, so copying only the list would leave them aliased."""
        row = {"id": 1, "tags": ["python"]}

        context.add_query_result({"data": [row]})
        row["tags"].append("async")

        assert context.query_results[0]["tags"] == ["python"]

    def test_one_payload_collected_by_two_contexts_is_not_shared(self):
        rows = [{"id": 1}]
        first = AgentContext(database_connector=Mock())
        second = AgentContext(database_connector=Mock())

        first.add_query_result({"data": rows})
        second.add_query_result({"data": rows})
        first.query_results[0]["id"] = 99

        assert second.query_results == [{"id": 1}]

    def test_the_collected_discovery_is_readable(self, context):
        context.add_discovered_item("tags", {"name": "python"})

        assert context.discovered_data == {"tags": [{"name": "python"}]}

    def test_a_later_edit_to_the_discovered_value_does_not_reach_the_context(self, context):
        discovered = {"name": "python"}

        context.add_discovered_item("tags", discovered)
        discovered["name"] = "async"

        assert context.discovered_data["tags"] == [{"name": "python"}]

    def test_an_edit_inside_a_nested_discovered_value_does_not_reach_the_context(self, context):
        """``value`` is declared Any, so the copy has to reach whatever the run put there."""
        discovered = {"versions": ["3.10"]}

        context.add_discovered_item("tags", discovered)
        discovered["versions"].append("3.13")

        assert context.discovered_data["tags"] == [{"versions": ["3.10"]}]

    def test_one_object_discovered_twice_records_two_independent_entries(self, context):
        discovered = {"name": "python"}

        context.add_discovered_item("tags", discovered)
        context.add_discovered_item("tags", discovered)
        context.discovered_data["tags"][0]["name"] = "async"

        assert context.discovered_data["tags"][1] == {"name": "python"}

    def test_a_discovery_that_overwrites_a_non_list_is_copied_too(self, context):
        """The overwrite branch stores the value directly, so it needs the same copy."""
        context.discovered_data["key"] = "scalar"
        discovered = {"name": "python"}

        context.add_discovered_item("key", discovered)
        discovered["name"] = "async"

        assert context.discovered_data["key"] == {"name": "python"}


class TestAgentContextHandsOutSnapshots:
    """The read boundary copies out, so a consumer of a reading cannot write back into the run."""

    def test_the_collected_discoveries_are_readable(self, context):
        context.add_discovered_item("tags", "python")
        context.add_discovered_item("tags", "async")

        assert context.get_discovered_items("tags") == ["python", "async"]

    def test_a_consumer_appending_to_the_returned_list_does_not_reach_the_context(self, context):
        context.add_discovered_item("tags", "python")

        context.get_discovered_items("tags").append("async")

        assert context.get_discovered_items("tags") == ["python"]

    def test_a_consumer_edit_inside_a_returned_discovery_does_not_reach_the_context(self, context):
        """The list nests the discovered values, so a shallow copy would leave them reachable."""
        context.add_discovered_item("tags", {"name": "python"})

        context.get_discovered_items("tags")[0]["name"] = "async"

        assert context.get_discovered_items("tags") == [{"name": "python"}]

    def test_two_reads_of_one_key_do_not_share_a_list(self, context):
        context.add_discovered_item("tags", {"name": "python"})

        first = context.get_discovered_items("tags")
        second = context.get_discovered_items("tags")
        first[0]["name"] = "async"

        assert second == [{"name": "python"}]

    def test_a_non_list_value_is_handed_out_detached_too(self, context):
        """A value stored under a key that is not a list goes out through the same boundary."""
        context.discovered_data["schema"] = {"tables": ["users"]}

        context.get_discovered_items("schema")["tables"].append("orders")

        assert context.discovered_data["schema"] == {"tables": ["users"]}

    def test_a_key_that_was_never_collected_still_reads_as_none(self, context):
        assert context.get_discovered_items("nonexistent") is None
