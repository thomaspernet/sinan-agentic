"""Tests for capability steering (core/capabilities/steering.py)."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from agents import RunContextWrapper

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.capabilities.steering import (
    FRAGMENT_SEPARATOR,
    STEERING_ITEM_ROLE,
    CapabilitySteering,
    build_capability_steering,
)
from tests.conftest import drive_model_input_filter


class Speaking(Capability):
    """A capability whose fragment changes whenever the test moves its state."""

    def __init__(self, text: str | None) -> None:
        self.text = text
        self.contexts: list[Any] = []

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        self.contexts.append(ctx.context)
        return self.text


class TestBuildCapabilitySteering:
    def test_no_capabilities_means_no_filter(self):
        assert build_capability_steering([]) is None

    def test_capabilities_produce_a_filter(self):
        assert isinstance(build_capability_steering([Speaking("hi")]), CapabilitySteering)

    def test_each_call_returns_a_fresh_filter(self):
        """Drift state is per run, so two runs of one agent must not share it."""
        caps = [Speaking("hi")]

        assert build_capability_steering(caps) is not build_capability_steering(caps)

    def test_the_capabilities_themselves_stay_shared_with_the_caller(self):
        """The container is detached; the live per-run state inside it is not."""
        cap = Speaking("hi")
        caps = [cap]

        steering = build_capability_steering(caps)
        caps.append(Speaking("late"))

        steered = drive_model_input_filter(steering)
        assert steered.input[-1]["content"] == "hi"
        assert cap.contexts == [None]


class TestTheSteeringItem:
    def test_the_fragment_is_appended_as_a_trailing_item(self):
        steering = CapabilitySteering([Speaking("stay on task")])

        steered = drive_model_input_filter(
            steering, input_items=[{"role": "user", "content": "hi"}]
        )

        assert steered.input[-1] == {
            "role": STEERING_ITEM_ROLE,
            "content": "stay on task",
        }

    def test_fragments_are_joined_in_registration_order(self):
        steering = CapabilitySteering([Speaking("first"), Speaking("second")])

        steered = drive_model_input_filter(steering)

        assert steered.input[-1]["content"] == f"first{FRAGMENT_SEPARATOR}second"

    def test_a_capability_contributing_nothing_is_skipped(self):
        steering = CapabilitySteering([Speaking(None), Speaking("only me"), Speaking("")])

        steered = drive_model_input_filter(steering)

        assert steered.input[-1]["content"] == "only me"

    def test_nothing_is_appended_when_every_capability_is_quiet(self):
        steering = CapabilitySteering([Speaking(None)])
        items = [{"role": "user", "content": "hi"}]

        steered = drive_model_input_filter(steering, input_items=items)

        assert steered.input == items

    def test_the_resolved_instructions_are_forwarded_untouched(self):
        steering = CapabilitySteering([Speaking("steer")])

        steered = drive_model_input_filter(steering, instructions="You answer questions.")

        assert steered.instructions == "You answer questions."

    def test_absent_instructions_stay_absent(self):
        """``None`` and ``""`` are different states; neither collapses into the other."""
        steering = CapabilitySteering([Speaking("steer")])

        assert drive_model_input_filter(steering, instructions=None).instructions is None
        assert drive_model_input_filter(steering, instructions="").instructions == ""

    def test_the_run_context_reaches_the_capability(self):
        cap = Speaking("steer")
        context = object()

        drive_model_input_filter(CapabilitySteering([cap]), context=context)

        assert cap.contexts == [context]

    def test_the_fragment_is_rebuilt_on_every_call(self):
        cap = Speaking("first")
        steering = CapabilitySteering([cap])

        first = drive_model_input_filter(steering)
        cap.text = "second"
        second = drive_model_input_filter(steering)

        assert first.input[-1]["content"] == "first"
        assert second.input[-1]["content"] == "second"


class TestExistingItemsAreNeverRebuilt:
    """A server-side conversation matches pending items by identity.

    Rebuilding one trips the SDK's reconstructed-item check
    (``agents.run_internal.oai_conversation``), so the filter appends only.
    """

    def test_every_existing_item_is_forwarded_by_identity(self):
        items = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        steered = drive_model_input_filter(
            CapabilitySteering([Speaking("steer")]), input_items=items
        )

        assert [id(item) for item in steered.input[:-1]] == [id(item) for item in items]

    def test_the_caller_s_list_is_not_extended_in_place(self):
        items = [{"role": "user", "content": "hi"}]

        drive_model_input_filter(CapabilitySteering([Speaking("steer")]), input_items=items)

        assert len(items) == 1


class TestTheDriftGuard:
    def test_the_first_call_warns_about_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(CapabilitySteering([Speaking("steer")]), instructions="stable")

        assert caplog.records == []

    def test_a_stable_prompt_across_calls_warns_about_nothing(self, caplog):
        steering = CapabilitySteering([Speaking("steer")])

        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(steering, instructions="stable")
            drive_model_input_filter(steering, instructions="stable")

        assert caplog.records == []

    def test_a_changed_prompt_warns(self, caplog):
        steering = CapabilitySteering([Speaking("steer")])

        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(steering, instructions="first")
            drive_model_input_filter(steering, instructions="second longer")

        assert len(caplog.records) == 1
        assert "changed between two model calls" in caplog.records[0].getMessage()

    def test_the_warning_reports_absent_and_empty_apart(self, caplog):
        steering = CapabilitySteering([Speaking("steer")])

        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(steering, instructions=None)
            drive_model_input_filter(steering, instructions="")

        message = caplog.records[0].getMessage()
        assert "previous: absent" in message
        assert "current: 0 chars" in message

    def test_the_prompt_itself_is_never_logged(self, caplog):
        steering = CapabilitySteering([Speaking("steer")])

        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(steering, instructions="a secret playbook")
            drive_model_input_filter(steering, instructions="a different playbook")

        assert "playbook" not in caplog.records[0].getMessage()

    @pytest.mark.parametrize("quiet_text", [None, "steer"])
    def test_drift_is_guarded_whether_or_not_a_fragment_is_appended(self, caplog, quiet_text):
        """The guard runs before the early return, so a quiet run is watched too."""
        steering = CapabilitySteering([Speaking(quiet_text)])

        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(steering, instructions="first")
            drive_model_input_filter(steering, instructions="second")

        assert len(caplog.records) == 1

    def test_two_runs_do_not_read_each_other_s_instructions(self, caplog):
        caps = [Speaking("steer")]

        with caplog.at_level(logging.WARNING):
            drive_model_input_filter(build_capability_steering(caps), instructions="run one")
            drive_model_input_filter(build_capability_steering(caps), instructions="run two")

        assert caplog.records == []
