"""Tests for CapabilityRegistry — registration, lookup, factory invocation."""

from __future__ import annotations

from typing import Any

import pytest
from agents import RunContextWrapper
from pydantic import ValidationError

from sinan_agentic_core.core.capabilities import Capability
from sinan_agentic_core.core.tool_error_recovery import (
    ToolErrorRecovery,
    build_tool_error_recovery,
)
from sinan_agentic_core.core.turn_budget import (
    DEFAULT_ABSOLUTE_MAX_TURNS,
    TurnBudget,
    TurnBudgetConfig,
    build_turn_budget,
)
from sinan_agentic_core.registry.capability_registry import (
    CapabilityNotFoundError,
    CapabilityRegistry,
    get_capability_registry,
    register_capability,
)

# ---------------------------------------------------------------------------
# Test capability
# ---------------------------------------------------------------------------


class _RecorderCapability(Capability):
    """Trivial capability that just stores the kwargs it was built with."""

    def __init__(self, label: str = "default", flag: bool = False) -> None:
        self.label = label
        self.flag = flag

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        return f"recorder({self.label}, flag={self.flag})"


# ---------------------------------------------------------------------------
# CapabilityRegistry — direct API
# ---------------------------------------------------------------------------


class TestCapabilityRegistry:
    def test_register_and_lookup(self) -> None:
        reg = CapabilityRegistry()

        def factory(config: dict[str, Any]) -> Capability:
            return _RecorderCapability(**config)

        reg.register("recorder", factory)
        assert reg.is_registered("recorder")
        assert reg.get("recorder") is factory
        assert "recorder" in reg.list_names()

    def test_unknown_name_raises(self) -> None:
        reg = CapabilityRegistry()
        with pytest.raises(CapabilityNotFoundError, match="not registered"):
            reg.get("missing")

    def test_capability_not_found_is_keyerror(self) -> None:
        # Caller code that catches KeyError must keep working.
        assert issubclass(CapabilityNotFoundError, KeyError)

    def test_build_invokes_factory_with_config(self) -> None:
        reg = CapabilityRegistry()
        captured: dict[str, Any] = {}

        def factory(config: dict[str, Any]) -> Capability:
            captured.update(config)
            return _RecorderCapability(**config)

        reg.register("recorder", factory)
        cap = reg.build("recorder", {"label": "hi", "flag": True})

        assert isinstance(cap, _RecorderCapability)
        assert cap.label == "hi"
        assert cap.flag is True
        assert captured == {"label": "hi", "flag": True}

    def test_build_with_no_config_passes_empty_dict(self) -> None:
        reg = CapabilityRegistry()
        seen: list[dict[str, Any]] = []

        def factory(config: dict[str, Any]) -> Capability:
            seen.append(config)
            return _RecorderCapability()

        reg.register("recorder", factory)
        reg.build("recorder")

        assert seen == [{}]

    def test_build_copies_config_dict(self) -> None:
        """Mutating the original dict after build must not affect the factory."""
        reg = CapabilityRegistry()

        def factory(config: dict[str, Any]) -> Capability:
            return _RecorderCapability(**config)

        reg.register("recorder", factory)
        config = {"label": "before"}
        cap = reg.build("recorder", config)
        config["label"] = "after"

        assert isinstance(cap, _RecorderCapability)
        assert cap.label == "before"

    def test_re_registration_overwrites(self) -> None:
        reg = CapabilityRegistry()

        def first(config: dict[str, Any]) -> Capability:
            return _RecorderCapability(label="first")

        def second(config: dict[str, Any]) -> Capability:
            return _RecorderCapability(label="second")

        reg.register("x", first)
        reg.register("x", second)

        cap = reg.build("x")
        assert isinstance(cap, _RecorderCapability)
        assert cap.label == "second"

    def test_list_names_sorted(self) -> None:
        reg = CapabilityRegistry()

        def factory(config: dict[str, Any]) -> Capability:
            return _RecorderCapability()

        reg.register("zebra", factory)
        reg.register("alpha", factory)
        reg.register("middle", factory)

        assert reg.list_names() == ["alpha", "middle", "zebra"]


# ---------------------------------------------------------------------------
# Decorator + global registry
# ---------------------------------------------------------------------------


class TestRegisterCapabilityDecorator:
    def test_decorator_registers_in_global(self) -> None:
        reg = get_capability_registry()
        name = "test_decorated_cap_unique"

        @register_capability(name)
        def factory(config: dict[str, Any]) -> Capability:
            return _RecorderCapability(**config)

        try:
            assert reg.is_registered(name)
            cap = reg.build(name, {"label": "via-decorator"})
            assert isinstance(cap, _RecorderCapability)
            assert cap.label == "via-decorator"
        finally:
            # Don't pollute the global registry across tests.
            reg._factories.pop(name, None)

    def test_decorator_returns_factory_unchanged(self) -> None:
        @register_capability("test_returns_unchanged")
        def factory(config: dict[str, Any]) -> Capability:
            return _RecorderCapability()

        try:
            assert callable(factory)
            assert factory({}).__class__ is _RecorderCapability
        finally:
            get_capability_registry()._factories.pop("test_returns_unchanged", None)


# ---------------------------------------------------------------------------
# Built-in capabilities (registered on import)
# ---------------------------------------------------------------------------


class TestBuiltInCapabilities:
    def test_turn_budget_registered(self) -> None:
        reg = get_capability_registry()
        assert reg.is_registered("turn_budget")

    def test_error_recovery_registered(self) -> None:
        reg = get_capability_registry()
        assert reg.is_registered("error_recovery")

    def test_build_turn_budget_with_config(self) -> None:
        reg = get_capability_registry()
        cap = reg.build(
            "turn_budget",
            {
                "default_turns": 7,
                "reminder_at": 1,
                "max_extensions": 2,
                "extension_size": 4,
                "absolute_max": 20,
            },
        )
        assert isinstance(cap, TurnBudget)
        assert cap.default_turns == 7
        assert cap.reminder_at == 1
        assert cap.max_extensions == 2
        assert cap.extension_size == 4
        assert cap.absolute_max == 20

    def test_build_turn_budget_defaults(self) -> None:
        reg = get_capability_registry()
        cap = reg.build("turn_budget")
        assert isinstance(cap, TurnBudget)
        # Sanity-check the dataclass defaults hold.
        assert cap.default_turns == 10

    def test_build_turn_budget_without_a_ceiling_falls_back_to_the_default(self) -> None:
        cap = get_capability_registry().build("turn_budget", {"default_turns": 8})
        assert isinstance(cap, TurnBudget)
        assert cap.absolute_max == DEFAULT_ABSOLUTE_MAX_TURNS

    def test_build_turn_budget_matches_the_shorthand_path(self) -> None:
        """Both declaration paths translate one config the same way."""
        declared = {name: 7 for name in TurnBudgetConfig.model_fields}

        from_list = get_capability_registry().build("turn_budget", {**declared, "absolute_max": 20})
        from_shorthand = build_turn_budget(TurnBudgetConfig(**declared), 20)

        assert from_list == from_shorthand

    def test_build_turn_budget_rejects_an_unknown_key(self) -> None:
        """An unrecognized key raised before the shared translator, and still does."""
        with pytest.raises(ValidationError, match="typo"):
            get_capability_registry().build("turn_budget", {"typo": 5})

    def test_build_error_recovery_uses_global_tool_registry(self) -> None:
        from sinan_agentic_core.registry.tool_registry import (
            get_tool_registry,
        )

        reg = get_capability_registry()
        cap = reg.build("error_recovery")
        assert isinstance(cap, ToolErrorRecovery)
        # Defaults to the global tool registry when no override is provided.
        assert cap._registry is get_tool_registry()

    def test_build_error_recovery_with_explicit_max(self) -> None:
        reg = get_capability_registry()
        cap = reg.build("error_recovery", {"max_identical_before_stop": 5})
        assert isinstance(cap, ToolErrorRecovery)
        assert cap.max_identical_before_stop == 5

    def test_build_error_recovery_with_mcp_hints(self) -> None:
        cap = get_capability_registry().build(
            "error_recovery", {"mcp_hints": {"mcp_search": "Use a longer query."}}
        )
        assert isinstance(cap, ToolErrorRecovery)
        assert cap._mcp_hints == {"mcp_search": "Use a longer query."}

    def test_build_error_recovery_honors_an_explicit_registry(self) -> None:
        """A caller holding a registry still overrides the global fallback."""
        from sinan_agentic_core.registry.tool_registry import ToolRegistry

        registry = ToolRegistry()

        cap = get_capability_registry().build("error_recovery", {"tool_registry": registry})

        assert isinstance(cap, ToolErrorRecovery)
        assert cap._registry is registry

    def test_build_error_recovery_matches_the_shorthand_path(self) -> None:
        """Both declaration paths translate one config the same way."""
        from_list = get_capability_registry().build("error_recovery")
        from_shorthand = build_tool_error_recovery(True)

        assert isinstance(from_list, ToolErrorRecovery)
        assert from_shorthand is not None
        assert from_list._registry is from_shorthand._registry
        assert from_list._mcp_hints == from_shorthand._mcp_hints
        assert from_list.max_identical_before_stop == from_shorthand.max_identical_before_stop

    def test_build_error_recovery_rejects_an_unknown_key(self) -> None:
        """An unrecognized key raised before the shared translator, and still does."""
        with pytest.raises(ValidationError, match="typo"):
            get_capability_registry().build("error_recovery", {"typo": 5})
