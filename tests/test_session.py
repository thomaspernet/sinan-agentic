"""Tests for session management (in-memory and SQLite)."""

from typing import Any

import pytest

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.tool_error_recovery import ToolErrorRecovery
from sinan_agentic_core.core.turn_budget import TurnBudget
from sinan_agentic_core.session.agent_session import (
    _STRUCTURED_OUTPUT_WRAPPER_KEY,
    AgentSession,
    ConversationHistory,
)
from sinan_agentic_core.session.sqlite_store import (
    SESSION_TITLE_CHARS,
    SESSION_TITLE_ELLIPSIS,
    SQLiteSessionStore,
)
from tests.conftest import edit_every_level


class _PersistedHistory(ConversationHistory):
    """History subclass — the class docstring documents subclassing for persistence."""


def _metadata_value() -> dict[str, Any]:
    """A metadata value nesting a further mutable container at every level."""
    return {"tags": ["research"], "limits": {"max_rows": 10}}


# -- ConversationHistory -------------------------------------------------------


class TestConversationHistory:
    def test_add_message(self):
        h = ConversationHistory()
        h.add_message("user", "Hello")
        assert len(h.messages) == 1
        assert h.messages[0] == {"role": "user", "content": "Hello"}

    def test_add_message_with_kwargs(self):
        h = ConversationHistory()
        h.add_message("assistant", "Hi", name="bot")
        assert h.messages[0]["name"] == "bot"

    def test_to_list_dict_returns_copy(self):
        h = ConversationHistory()
        h.add_message("user", "Hi")
        copy = h.to_list_dict()
        copy.append({"role": "system", "content": "injected"})
        assert len(h.messages) == 1  # original unchanged

    def test_to_list_dict_carries_the_stored_messages(self):
        h = ConversationHistory()
        h.add_message("user", "Hi", name="asker")
        assert h.to_list_dict() == [{"role": "user", "content": "Hi", "name": "asker"}]

    def test_clear(self):
        h = ConversationHistory()
        h.add_message("user", "Hi")
        h.clear()
        assert h.messages == []


# -- ConversationHistory hands out snapshots -----------------------------------


class TestConversationHistoryHandsOutSnapshots:
    """``to_list_dict`` detaches the messages, not only the list around them."""

    def test_an_edit_to_a_returned_message_does_not_reach_the_history(self):
        h = ConversationHistory()
        h.add_message("user", "Hi")

        h.to_list_dict()[0]["content"] = "Edited by the reader"

        assert h.messages[0]["content"] == "Hi"

    def test_an_edit_inside_a_returned_content_list_does_not_reach_the_history(self):
        """Structured-output content is a list of dicts, one level deeper again."""
        h = ConversationHistory()
        h.messages.append({"role": "assistant", "content": [{"text": "original"}]})

        h.to_list_dict()[0]["content"][0]["text"] = "edited by the reader"

        assert h.messages[0]["content"][0]["text"] == "original"

    def test_two_readings_do_not_share_a_message(self):
        h = ConversationHistory()
        h.add_message("user", "Hi")

        first = h.to_list_dict()
        second = h.to_list_dict()
        first[0]["content"] = "Edited by the first reader"

        assert second[0]["content"] == "Hi"


# -- AgentSession --------------------------------------------------------------


class TestAgentSession:
    async def test_add_and_get_items(self, session):
        await session.add_items([{"role": "user", "content": "Hello"}])
        items = await session.get_items()
        assert len(items) == 1
        assert items[0]["content"] == "Hello"

    async def test_get_items_with_limit(self, session):
        await session.add_items(
            [
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
                {"role": "user", "content": "msg3"},
            ]
        )
        items = await session.get_items(limit=2)
        assert len(items) == 2
        assert items[0]["content"] == "msg2"

    async def test_skips_empty_content(self, session):
        await session.add_items(
            [
                {"role": "user", "content": ""},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "valid"},
            ]
        )
        items = await session.get_items()
        assert len(items) == 1
        assert items[0]["content"] == "valid"

    async def test_pop_item(self, session):
        await session.add_items([{"role": "user", "content": "first"}])
        popped = await session.pop_item()
        assert popped["content"] == "first"
        assert await session.get_items() == []

    async def test_pop_item_empty(self, session):
        assert await session.pop_item() is None

    async def test_clear_session(self, session):
        await session.add_items([{"role": "user", "content": "msg"}])
        await session.clear_session()
        assert await session.get_items() == []

    def test_get_message_count(self, session):
        session.history.add_message("user", "Hi")
        session.history.add_message("assistant", "Hello")
        assert session.get_message_count() == 2

    def test_needs_summarization(self):
        s = AgentSession(session_id="test", max_items=2)
        s.history.add_message("user", "1")
        s.history.add_message("assistant", "2")
        assert s.needs_summarization() is False
        s.history.add_message("user", "3")
        assert s.needs_summarization() is True

    def test_metadata(self, session):
        session.set_metadata("user_id", "u123")
        assert session.get_metadata("user_id") == "u123"
        assert session.get_metadata("missing", "default") == "default"

    def test_get_history_returns_object(self, session):
        h = session.get_history()
        assert isinstance(h, ConversationHistory)

    async def test_add_items_with_name_and_tool_call_id(self, session):
        await session.add_items(
            [
                {"role": "assistant", "content": "hi", "name": "bot"},
                {"role": "tool", "content": "result", "tool_call_id": "tc_123"},
            ]
        )
        items = await session.get_items()
        assert items[0]["name"] == "bot"
        assert items[1]["tool_call_id"] == "tc_123"

    async def test_structured_output_cleanup(self, session):
        """SDK returns structured output as list of dicts — session should extract text."""
        await session.add_items(
            [
                {
                    "role": "assistant",
                    "content": [{"text": '{"response": "parsed answer"}', "type": "output_text"}],
                }
            ]
        )
        items = await session.get_items()
        assert len(items) == 1
        assert '"parsed answer"' in items[0]["content"]

    async def test_unwraps_fenced_structured_output(self, session):
        """A model that fences its payload still stores the answer, not the envelope."""
        await session.add_items(
            [
                {
                    "role": "assistant",
                    "content": '```json\n{"response": "fenced answer"}\n```',
                }
            ]
        )
        items = await session.get_items()
        assert items[0]["content"] == '"fenced answer"'

    async def test_unwraps_structured_output_behind_a_preamble(self, session):
        await session.add_items(
            [
                {
                    "role": "assistant",
                    "content": 'Here is the result:\n{"response": "prose answer"}',
                }
            ]
        )
        items = await session.get_items()
        assert items[0]["content"] == '"prose answer"'

    async def test_unwraps_structured_output_with_braces_in_the_value(self, session):
        await session.add_items(
            [
                {
                    "role": "assistant",
                    "content": '```json\n{"response": "use {braces} freely"}\n```',
                }
            ]
        )
        items = await session.get_items()
        assert items[0]["content"] == '"use {braces} freely"'

    async def test_json_without_the_wrapper_key_is_stored_as_is(self, session):
        await session.add_items([{"role": "assistant", "content": '{"answer": "yes"}'}])
        items = await session.get_items()
        assert items[0]["content"] == '{"answer": "yes"}'

    async def test_plain_text_is_stored_as_is(self, session):
        await session.add_items([{"role": "assistant", "content": "just a sentence"}])
        items = await session.get_items()
        assert items[0]["content"] == "just a sentence"

    async def test_non_assistant_messages_are_never_unwrapped(self, session):
        await session.add_items([{"role": "user", "content": '{"response": "verbatim"}'}])
        items = await session.get_items()
        assert items[0]["content"] == '{"response": "verbatim"}'

    def test_the_wrapper_key_still_matches_the_one_the_sdk_wraps_under(self):
        """Unwrapping is keyed off an SDK-private constant, so drift must fail loudly.

        ``AgentOutputSchema`` nests a non-dict output type under a key the SDK
        keeps private, so this package mirrors the literal rather than importing
        it into production code. A rename upstream would otherwise leave every
        structured-output message stored as its envelope, with nothing raising.
        """
        from agents.agent_output import _WRAPPER_DICT_KEY

        assert _STRUCTURED_OUTPUT_WRAPPER_KEY == _WRAPPER_DICT_KEY


# -- AgentSession copies the caller's history ----------------------------------


class TestAgentSessionCopiesTheCallersHistory:
    """The session owns its history, so it and the caller never share a message list."""

    async def test_the_seeded_messages_are_readable(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")

        session = AgentSession(session_id="seeded", initial_history=history)

        items = await session.get_items()
        assert [item["content"] for item in items] == ["Hello"]

    async def test_a_message_added_to_the_callers_history_later_is_not_visible(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        session = AgentSession(session_id="seeded", initial_history=history)

        history.add_message("user", "Added late")

        items = await session.get_items()
        assert [item["content"] for item in items] == ["Hello"]

    async def test_an_edit_inside_a_seeded_message_is_not_visible(self):
        """Messages nest inside the list, so copying only the list would not detach them."""
        history = ConversationHistory()
        history.add_message("user", "Hello")
        session = AgentSession(session_id="seeded", initial_history=history)

        history.messages[0]["content"] = "Edited late"

        items = await session.get_items()
        assert items[0]["content"] == "Hello"

    async def test_an_edit_inside_a_nested_content_list_is_not_visible(self):
        """Structured-output content arrives as a list of dicts, one level deeper again."""
        history = ConversationHistory()
        history.messages.append(
            {"role": "assistant", "content": [{"text": "original", "type": "output_text"}]}
        )
        session = AgentSession(session_id="seeded", initial_history=history)

        history.messages[0]["content"][0]["text"] = "edited late"

        items = await session.get_items()
        assert items[0]["content"][0]["text"] == "original"

    async def test_messages_added_to_the_session_do_not_reach_the_callers_history(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        session = AgentSession(session_id="seeded", initial_history=history)

        await session.add_items([{"role": "assistant", "content": "Hi there"}])

        assert [msg["content"] for msg in history.messages] == ["Hello"]

    async def test_two_sessions_seeded_from_one_history_do_not_share_a_message_list(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        first = AgentSession(session_id="first", initial_history=history)
        second = AgentSession(session_id="second", initial_history=history)

        await first.add_items([{"role": "assistant", "content": "only in first"}])

        assert first.get_message_count() == 2
        assert second.get_message_count() == 1


# -- AgentSession hands out history snapshots ----------------------------------


class TestAgentSessionHandsOutHistorySnapshots:
    """The outbound half of the ownership the constructor claims."""

    async def test_an_edit_to_a_returned_item_does_not_reach_the_session(self, session):
        await session.add_items([{"role": "user", "content": "Hello"}])

        (await session.get_items())[0]["content"] = "Edited by the reader"

        assert (await session.get_items())[0]["content"] == "Hello"

    async def test_an_edit_inside_a_returned_items_content_does_not_reach_the_session(self):
        history = ConversationHistory()
        history.messages.append({"role": "assistant", "content": [{"text": "original"}]})
        session = AgentSession(session_id="nested", initial_history=history)

        (await session.get_items())[0]["content"][0]["text"] = "edited by the reader"

        assert (await session.get_items())[0]["content"][0]["text"] == "original"

    async def test_an_edit_to_a_limited_reading_does_not_reach_the_session(self, session):
        """``limit`` slices the snapshot, so the slice is detached with it."""
        await session.add_items(
            [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
        )

        (await session.get_items(limit=1))[0]["content"] = "Edited by the reader"

        assert [item["content"] for item in await session.get_items()] == ["first", "second"]

    async def test_two_readings_do_not_share_an_item(self, session):
        await session.add_items([{"role": "user", "content": "Hello"}])

        first = await session.get_items()
        second = await session.get_items()
        first[0]["content"] = "Edited by the first reader"

        assert second[0]["content"] == "Hello"

    def test_get_history_does_not_hand_out_the_live_history(self, session):
        assert session.get_history() is not session.history

    def test_a_message_added_to_a_returned_history_does_not_reach_the_session(self, session):
        session.get_history().add_message("system", "injected")

        assert session.get_message_count() == 0

    def test_an_edit_inside_a_returned_history_does_not_reach_the_session(self, session):
        session.history.add_message("user", "Hello")

        session.get_history().messages[0]["content"] = "Edited by the reader"

        assert session.history.messages[0]["content"] == "Hello"

    def test_two_readings_do_not_share_a_history(self, session):
        session.history.add_message("user", "Hello")

        first = session.get_history()
        second = session.get_history()
        first.messages[0]["content"] = "Edited by the first reader"

        assert second.messages[0]["content"] == "Hello"

    def test_the_returned_history_carries_the_stored_messages(self, session):
        session.history.add_message("user", "Hello")

        assert session.get_history().messages == [{"role": "user", "content": "Hello"}]

    def test_a_callers_history_subclass_survives_the_round_trip(self):
        """Copying the object keeps a caller's subclass; rebuilding the base type would not."""
        session = AgentSession(session_id="subclassed", initial_history=_PersistedHistory())

        assert isinstance(session.get_history(), _PersistedHistory)


# -- AgentSession owns its metadata --------------------------------------------


class TestAgentSessionOwnsItsMetadata:
    """Metadata is a record of what was stored, not a live view of it."""

    def test_the_stored_value_is_readable(self, session):
        session.set_metadata("filters", _metadata_value())

        assert session.get_metadata("filters") == _metadata_value()

    def test_a_later_edit_by_the_caller_does_not_reach_the_session(self, session):
        value = _metadata_value()
        session.set_metadata("filters", value)

        edit_every_level(value)

        assert session.get_metadata("filters") == _metadata_value()

    def test_two_sessions_given_one_value_do_not_share_it(self):
        value = _metadata_value()
        first = AgentSession(session_id="first")
        second = AgentSession(session_id="second")
        first.set_metadata("filters", value)
        second.set_metadata("filters", value)

        edit_every_level(first.metadata["filters"])

        assert second.get_metadata("filters") == _metadata_value()

    def test_an_edit_to_what_a_reader_got_back_does_not_reach_the_session(self, session):
        session.set_metadata("filters", _metadata_value())

        edit_every_level(session.get_metadata("filters"))

        assert session.get_metadata("filters") == _metadata_value()

    def test_two_readers_of_one_key_do_not_share_a_value(self, session):
        session.set_metadata("filters", _metadata_value())

        first = session.get_metadata("filters")
        second = session.get_metadata("filters")
        edit_every_level(first)

        assert second == _metadata_value()

    def test_a_missing_key_returns_the_callers_own_default(self, session):
        """The default was never retained, so it is handed back rather than copied."""
        default: dict[str, Any] = {}

        assert session.get_metadata("missing", default) is default

    def test_a_stored_none_is_returned_rather_than_the_default(self, session):
        session.set_metadata("filters", None)

        assert session.get_metadata("filters", "fallback") is None


# -- SQLiteSessionStore --------------------------------------------------------


class TestSQLiteSessionStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteSessionStore(str(tmp_path / "test.db"))

    def test_get_or_create_session(self, store):
        s = store.get_or_create_session("s1")
        assert s["id"] == "s1"
        assert s["is_archived"] == 0

    def test_get_or_create_idempotent(self, store):
        s1 = store.get_or_create_session("s1")
        s2 = store.get_or_create_session("s1")
        assert s1["id"] == s2["id"]

    def test_add_and_get_messages(self, store):
        msg_id = store.add_message("s1", "user", "Hello")
        assert isinstance(msg_id, int)
        messages = store.get_messages("s1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_get_conversation_history(self, store):
        store.add_message("s1", "user", "Hi")
        store.add_message("s1", "assistant", "Hello!")
        history = store.get_conversation_history("s1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "Hello!"}

    def test_get_message_count(self, store):
        store.add_message("s1", "user", "a")
        store.add_message("s1", "user", "b")
        assert store.get_message_count("s1") == 2

    def test_archive_session(self, store):
        store.get_or_create_session("s1")
        assert store.archive_session("s1") is True
        # Archived sessions not returned by get_active
        assert len(store.get_active_sessions()) == 0
        assert len(store.get_archived_sessions()) == 1

    def test_archive_nonexistent(self, store):
        assert store.archive_session("missing") is False

    def test_clear_session(self, store):
        store.add_message("s1", "user", "msg")
        assert store.clear_session("s1") is True
        assert store.get_message_count("s1") == 0

    def test_get_active_sessions(self, store):
        store.get_or_create_session("s1")
        store.get_or_create_session("s2")
        active = store.get_active_sessions()
        assert len(active) == 2

    def test_title_set_from_first_user_message(self, store):
        store.add_message("s1", "user", "What is Python?")
        s = store.get_or_create_session("s1")
        assert s["title"] == "What is Python?"

    def test_title_at_the_bound_is_kept_whole(self, store):
        content = "a" * SESSION_TITLE_CHARS
        store.add_message("s1", "user", content)
        assert store.get_or_create_session("s1")["title"] == content

    def test_longer_title_is_cut_to_the_bound_and_marked(self, store):
        content = "a" * (SESSION_TITLE_CHARS + 1)
        store.add_message("s1", "user", content)
        title = store.get_or_create_session("s1")["title"]
        assert title == "a" * SESSION_TITLE_CHARS + SESSION_TITLE_ELLIPSIS
        # The message itself is untouched — only the label is cut.
        assert store.get_messages("s1")[0]["content"] == content

    def test_message_metadata(self, store):
        store.add_message("s1", "user", "msg", metadata={"source": "web"})
        messages = store.get_messages("s1")
        assert messages[0]["metadata"] == {"source": "web"}


# -- Capability snapshot persistence ------------------------------------------


class _NoopRecorder(Capability):
    """Stateless capability — to_snapshot returns None by default."""


class _ManualSnapshot(Capability):
    """Capability that opts in to persistence with a custom snapshot key."""

    snapshot_key = "manual.snapshot"

    def __init__(self) -> None:
        self.tally: int = 0

    def to_snapshot(self):
        return {"tally": self.tally}

    def from_snapshot(self, data):
        self.tally = int(data.get("tally", 0))


class TestSQLiteCapabilitySnapshots:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteSessionStore(str(tmp_path / "snap.db"))

    def test_save_and_load_snapshot(self, store):
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 4})
        assert store.load_capability_snapshot("s1", "TurnBudget") == {"turns_used": 4}

    def test_load_missing_snapshot_returns_none(self, store):
        assert store.load_capability_snapshot("s1", "TurnBudget") is None

    def test_save_overwrites_existing_snapshot(self, store):
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 1})
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 7})
        assert store.load_capability_snapshot("s1", "TurnBudget") == {"turns_used": 7}

    def test_snapshots_are_scoped_per_session(self, store):
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 1})
        store.save_capability_snapshot("s2", "TurnBudget", {"turns_used": 9})
        assert store.load_capability_snapshot("s1", "TurnBudget") == {"turns_used": 1}
        assert store.load_capability_snapshot("s2", "TurnBudget") == {"turns_used": 9}

    def test_load_all_snapshots_returns_keyed_dict(self, store):
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 4})
        store.save_capability_snapshot("s1", "ToolErrorRecovery", {"errors": {}})
        snapshots = store.load_all_capability_snapshots("s1")
        assert snapshots == {
            "TurnBudget": {"turns_used": 4},
            "ToolErrorRecovery": {"errors": {}},
        }

    def test_clear_session_drops_snapshots(self, store):
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 4})
        store.clear_session("s1")
        assert store.load_capability_snapshot("s1", "TurnBudget") is None

    def test_delete_capability_snapshots(self, store):
        store.save_capability_snapshot("s1", "TurnBudget", {"turns_used": 4})
        store.delete_capability_snapshots("s1")
        assert store.load_all_capability_snapshots("s1") == {}


class TestAgentSessionRehydrate:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteSessionStore(str(tmp_path / "hydrate.db"))

    def test_turn_budget_round_trip_through_session_and_store(self, store):
        # AC: configure a TurnBudget to 10 turns, advance counter to 4, snapshot,
        #     rehydrate in a new session, confirm 6 turns remain.
        original_budget = TurnBudget(default_turns=10)
        for _ in range(4):
            original_budget.record_turn()

        original_session = AgentSession(session_id="resume-1")
        original_session.persist_capabilities(store, [original_budget])

        resumed_budget = TurnBudget(default_turns=10)
        resumed_session = AgentSession(session_id="resume-1")
        resumed_session.rehydrate_capabilities(store, [resumed_budget])

        assert resumed_budget.turns_used == 4
        assert resumed_budget.remaining == 6

    def test_tool_error_recovery_round_trip(self, store):
        from unittest.mock import Mock

        from agents import RunContextWrapper

        original = ToolErrorRecovery()
        ctx = RunContextWrapper(context=None)
        tool = Mock()
        tool.name = "search"
        import json as _json

        original.on_tool_start(ctx, tool, _json.dumps({"q": "x"}))
        original.on_tool_end(ctx, tool, _json.dumps({"error": "boom"}))

        AgentSession(session_id="resume-2").persist_capabilities(store, [original])

        resumed = ToolErrorRecovery()
        AgentSession(session_id="resume-2").rehydrate_capabilities(store, [resumed])

        assert resumed.has_errors
        assert resumed._errors["search"].error == "boom"

    def test_stateless_capability_is_skipped(self, store):
        # Stateless capability (default to_snapshot returns None) must not crash
        # and must not write any row.
        cap = _NoopRecorder()
        session = AgentSession(session_id="resume-3")
        session.persist_capabilities(store, [cap])
        assert store.load_all_capability_snapshots("resume-3") == {}

        # Rehydrating with no row stored is also a no-op.
        session.rehydrate_capabilities(store, [cap])  # must not raise

    def test_rehydrate_skips_capabilities_without_stored_row(self, store):
        # If only one capability has a stored snapshot, the others stay
        # untouched at their zeroed defaults.
        budget = TurnBudget(default_turns=10)
        budget.record_turn()
        AgentSession(session_id="mixed").persist_capabilities(store, [budget])

        resumed_budget = TurnBudget(default_turns=10)
        resumed_recovery = ToolErrorRecovery()
        AgentSession(session_id="mixed").rehydrate_capabilities(
            store, [resumed_budget, resumed_recovery]
        )
        assert resumed_budget.turns_used == 1
        assert not resumed_recovery.has_errors

    def test_explicit_snapshot_key_is_honored(self, store):
        cap = _ManualSnapshot()
        cap.tally = 12
        AgentSession(session_id="manual").persist_capabilities(store, [cap])
        snapshots = store.load_all_capability_snapshots("manual")
        assert "manual.snapshot" in snapshots

        resumed = _ManualSnapshot()
        AgentSession(session_id="manual").rehydrate_capabilities(store, [resumed])
        assert resumed.tally == 12
