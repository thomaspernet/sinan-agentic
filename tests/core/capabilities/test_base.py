"""Tests for the Capability base class (core/capabilities/base.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from agents import RunContextWrapper, Tool

from sinan_agentic_core.core.capabilities import Capability


class DummyCapability(Capability):
    """Trivial subclass used to exercise the default behavior."""


class StatefulCapability(Capability):
    """Capability that tracks calls so we can assert on lifecycle wiring."""

    def __init__(self) -> None:
        self.counter: int = 0
        self.tool_starts: list[str] = []
        self.tool_ends: list[str] = []

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        return f"counter={self.counter}"

    def on_tool_start(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        args: str,
    ) -> None:
        self.counter += 1
        self.tool_starts.append(args)

    def on_tool_end(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        result: str,
    ) -> None:
        self.tool_ends.append(result)

    def reset(self) -> None:
        self.counter = 0
        self.tool_starts.clear()
        self.tool_ends.clear()


def _ctx() -> RunContextWrapper[Any]:
    return RunContextWrapper(context=None)


class TestImport:
    def test_capability_is_importable_from_package(self) -> None:
        from sinan_agentic_core.core.capabilities import Capability as ReExported

        assert ReExported is Capability


class TestDefaults:
    def test_instructions_returns_none(self) -> None:
        assert DummyCapability().instructions(_ctx()) is None

    def test_on_tool_start_is_noop(self) -> None:
        cap = DummyCapability()
        tool = Mock(spec=[], name="dummy_tool")
        cap.on_tool_start(_ctx(), tool, "{}")  # must not raise

    def test_on_tool_end_is_noop(self) -> None:
        cap = DummyCapability()
        tool = Mock(spec=[], name="dummy_tool")
        cap.on_tool_end(_ctx(), tool, "result")  # must not raise

    def test_reset_is_callable(self) -> None:
        cap = DummyCapability()
        cap.reset()  # must not raise

    def test_tools_default_is_empty_list(self) -> None:
        assert DummyCapability().tools() == []


class TestClone:
    def test_clone_returns_capability_instance(self) -> None:
        clone = DummyCapability().clone()
        assert isinstance(clone, Capability)

    def test_clone_returns_independent_instance(self) -> None:
        original = StatefulCapability()
        original.counter = 5
        original.tool_starts.append("a")

        clone = original.clone()
        assert isinstance(clone, StatefulCapability)
        assert clone is not original
        assert clone.counter == 5
        assert clone.tool_starts == ["a"]

        clone.counter = 99
        clone.tool_starts.append("b")
        assert original.counter == 5
        assert original.tool_starts == ["a"]

    def test_clone_does_not_share_mutable_state(self) -> None:
        original = StatefulCapability()
        original.tool_ends.append("x")

        clone = original.clone()
        assert isinstance(clone, StatefulCapability)
        assert clone.tool_ends is not original.tool_ends


class TestSubclassLifecycle:
    def test_subclass_can_override_instructions(self) -> None:
        cap = StatefulCapability()
        cap.counter = 7
        assert cap.instructions(_ctx()) == "counter=7"

    def test_subclass_can_override_tool_hooks(self) -> None:
        cap = StatefulCapability()
        tool = Mock(spec=[], name="t")
        cap.on_tool_start(_ctx(), tool, '{"k":1}')
        cap.on_tool_end(_ctx(), tool, "ok")
        assert cap.counter == 1
        assert cap.tool_starts == ['{"k":1}']
        assert cap.tool_ends == ["ok"]

    def test_reset_clears_state(self) -> None:
        cap = StatefulCapability()
        cap.counter = 3
        cap.tool_starts.append("x")
        cap.tool_ends.append("y")
        cap.reset()
        assert cap.counter == 0
        assert cap.tool_starts == []
        assert cap.tool_ends == []
