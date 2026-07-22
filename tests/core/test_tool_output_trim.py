"""Tests for declarative tool-output trimming (core/tool_output_trim.py)."""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent
from agents.extensions import ToolOutputTrimmer
from agents.run_config import CallModelData, ModelInputData
from pydantic import ValidationError

from sinan_agentic_core.core.tool_output_trim import (
    ToolOutputTrimConfig,
    build_tool_output_trimmer,
)

BULKY_OUTPUT = "x" * 3000


def _conversation(tool_name: str = "web_search", output: str = BULKY_OUTPUT) -> list[Any]:
    """One answered tool call from an older turn, followed by a newer user turn."""
    return [
        {"role": "user", "content": "first question"},
        {"type": "function_call", "call_id": "c1", "name": tool_name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": output},
        {"role": "user", "content": "second question"},
    ]


def _single_turn(calls: int = 20, output: str = BULKY_OUTPUT) -> list[Any]:
    """One question answered by many tool calls — no second user message."""
    items: list[Any] = [{"role": "user", "content": "one question"}]
    for i in range(calls):
        items.append(
            {"type": "function_call", "call_id": f"c{i}", "name": "web_search", "arguments": "{}"}
        )
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": output})
    return items


def _filtered(trimmer: ToolOutputTrimmer, items: list[Any]) -> list[Any]:
    """Run the trimmer the way the SDK does before a model call."""
    data: CallModelData[None] = CallModelData(
        model_data=ModelInputData(input=items, instructions=None),
        agent=Agent(name="agent"),
        context=None,
    )
    return trimmer(data).input


def _tool_output(items: list[Any]) -> str:
    return next(item["output"] for item in items if item.get("type") == "function_call_output")


class TestConfigDefaults:
    def test_declaring_nothing_leaves_every_field_to_the_sdk(self) -> None:
        """``tool_output_trim: {}`` is a valid opt-in that takes SDK defaults."""
        assert ToolOutputTrimConfig().build() == ToolOutputTrimmer()

    def test_unset_tools_make_every_tool_eligible(self) -> None:
        assert ToolOutputTrimConfig().build().trimmable_tools is None


class TestConfigValidation:
    def test_rejects_empty_tool_list(self) -> None:
        """No eligible tool would run the filter on every call and trim nothing."""
        with pytest.raises(ValidationError):
            ToolOutputTrimConfig(trimmable_tools=[])

    def test_rejects_zero_recent_turns(self) -> None:
        with pytest.raises(ValidationError):
            ToolOutputTrimConfig(recent_turns=0)

    def test_rejects_zero_max_output_chars(self) -> None:
        with pytest.raises(ValidationError):
            ToolOutputTrimConfig(max_output_chars=0)

    def test_rejects_negative_preview_chars(self) -> None:
        with pytest.raises(ValidationError):
            ToolOutputTrimConfig(preview_chars=-1)

    def test_accepts_a_preview_at_the_cap(self) -> None:
        """The two sizes are independent — the SDK compares replacement to original."""
        config = ToolOutputTrimConfig(max_output_chars=500, preview_chars=500)

        assert config.preview_chars == 500


class TestBuild:
    def test_declared_fields_map_onto_the_sdk_filter(self) -> None:
        trimmer = ToolOutputTrimConfig(
            recent_turns=3,
            max_output_chars=4000,
            preview_chars=500,
            trimmable_tools=["web_search"],
        ).build()

        assert isinstance(trimmer, ToolOutputTrimmer)
        assert trimmer.recent_turns == 3
        assert trimmer.max_output_chars == 4000
        assert trimmer.preview_chars == 500
        assert trimmer.trimmable_tools == frozenset({"web_search"})

    def test_a_partial_config_leaves_unset_fields_to_the_sdk(self) -> None:
        trimmer = ToolOutputTrimConfig(max_output_chars=4000).build()

        assert trimmer.max_output_chars == 4000
        assert trimmer.recent_turns == ToolOutputTrimmer().recent_turns
        assert trimmer.preview_chars == ToolOutputTrimmer().preview_chars


class TestBuildToolOutputTrimmer:
    """The translator both run-config-building paths share."""

    def test_a_declared_config_becomes_the_sdk_filter(self) -> None:
        trimmer = build_tool_output_trimmer(
            ToolOutputTrimConfig(recent_turns=3, max_output_chars=4000)
        )

        assert trimmer.recent_turns == 3
        assert trimmer.max_output_chars == 4000

    def test_no_declaration_means_no_filter(self) -> None:
        """Trimming removes content the model would have seen, so it is never implied."""
        assert build_tool_output_trimmer(None) is None

    def test_each_call_returns_a_fresh_filter(self) -> None:
        """The SDK dataclass is mutable, so no two runs may share one."""
        config = ToolOutputTrimConfig(max_output_chars=4000)

        assert build_tool_output_trimmer(config) is not build_tool_output_trimmer(config)


class TestFilterBehavior:
    """The built filter is what the SDK actually calls before each model call."""

    def test_an_oversized_older_output_is_replaced_by_a_preview(self) -> None:
        trimmer = ToolOutputTrimConfig(
            recent_turns=1, max_output_chars=500, preview_chars=100
        ).build()

        output = _tool_output(_filtered(trimmer, _conversation()))

        assert len(output) < len(BULKY_OUTPUT)
        assert "web_search" in output

    def test_an_output_under_the_cap_is_left_alone(self) -> None:
        trimmer = ToolOutputTrimConfig(
            recent_turns=1, max_output_chars=500, preview_chars=100
        ).build()

        items = _filtered(trimmer, _conversation(output="short answer"))

        assert _tool_output(items) == "short answer"

    def test_a_preview_at_the_cap_still_shortens_a_much_larger_output(self) -> None:
        """Only outputs near the cap are spared; a big one still collapses."""
        trimmer = ToolOutputTrimConfig(
            recent_turns=1, max_output_chars=500, preview_chars=500
        ).build()
        huge = "x" * 100_000

        output = _tool_output(_filtered(trimmer, _conversation(output=huge)))

        assert len(output) < len(huge) / 100

    def test_a_single_user_turn_is_never_trimmed(self) -> None:
        """The window counts user messages, so one question trims nothing at any size."""
        trimmer = ToolOutputTrimConfig(
            recent_turns=1, max_output_chars=500, preview_chars=100
        ).build()
        items = _single_turn()

        filtered = _filtered(trimmer, items)

        assert [item.get("output") for item in filtered] == [item.get("output") for item in items]

    def test_recent_turns_are_never_trimmed(self) -> None:
        """The tool call sits inside the window, so its size does not matter."""
        trimmer = ToolOutputTrimConfig(
            recent_turns=2, max_output_chars=500, preview_chars=100
        ).build()

        items = _filtered(trimmer, _conversation())

        assert _tool_output(items) == BULKY_OUTPUT

    def test_a_tool_outside_the_declared_list_is_left_alone(self) -> None:
        trimmer = ToolOutputTrimConfig(
            recent_turns=1,
            max_output_chars=500,
            preview_chars=100,
            trimmable_tools=["web_search"],
        ).build()

        items = _filtered(trimmer, _conversation(tool_name="read_file"))

        assert _tool_output(items) == BULKY_OUTPUT

    def test_the_original_items_are_not_mutated(self) -> None:
        """The SDK replays session items, so the filter must copy rather than edit."""
        trimmer = ToolOutputTrimConfig(
            recent_turns=1, max_output_chars=500, preview_chars=100
        ).build()
        items = _conversation()

        _filtered(trimmer, items)

        assert _tool_output(items) == BULKY_OUTPUT
