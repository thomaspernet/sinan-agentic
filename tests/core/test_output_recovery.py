"""Tests for structured-output recovery (core/output_recovery.py)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from agents import Agent, MessageOutputItem, ModelBehaviorError
from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel

from sinan_agentic_core.core.output_recovery import (
    build_error_handlers,
    build_output_schema,
    invalid_final_output_handlers,
    iter_payload_candidates,
    recover_invalid_final_output,
    salvage_structured_output,
)
from sinan_agentic_core.registry.agent_registry import AgentDefinition, AgentRegistry
from tests.core.conftest import assistant_message


class Extraction(BaseModel):
    answer: str
    score: int


@pytest.fixture
def schema():
    return AgentOutputSchema(Extraction)


@pytest.fixture
def recovery_registry():
    """An isolated registry, patched inside the shared by-name definition resolver."""
    registry = AgentRegistry()
    registry.register(
        AgentDefinition(
            name="strict_agent",
            description="fails loudly",
            instructions="answer",
            invalid_output_recovery=False,
        )
    )
    registry.register(
        AgentDefinition(name="recovering_agent", description="recovers", instructions="answer")
    )

    with patch(
        "sinan_agentic_core.registry.agent_registry.get_agent_registry", return_value=registry
    ):
        yield registry


def _message_item(text: str) -> MessageOutputItem:
    """Build a MessageOutputItem carrying *text*, as the SDK would."""
    return MessageOutputItem(raw_item=assistant_message(text, message_id="msg-1"), agent=Mock())


def _handler_input(text: str | None, output_type: type | None = Extraction):
    """Build a RunErrorHandlerInput carrying a final message of *text*."""
    agent = Agent(name="extractor", instructions="extract", output_type=output_type)
    run_data = Mock(
        last_agent=agent,
        new_items=[_message_item(text)] if text is not None else [],
    )
    return Mock(error=ModelBehaviorError("invalid json"), run_data=run_data)


# ------------------------------------------------------------------ #
# build_output_schema
# ------------------------------------------------------------------ #


class TestBuildOutputSchema:
    def test_none_output_type_has_no_schema(self):
        assert build_output_schema(None) is None

    def test_str_output_type_has_no_schema(self):
        assert build_output_schema(str) is None

    def test_model_output_type_builds_schema(self):
        assert build_output_schema(Extraction) is not None

    def test_existing_schema_passes_through(self, schema):
        assert build_output_schema(schema) is schema

    def test_defaults_to_strict_matching_the_sdk(self):
        strict = build_output_schema(Extraction)
        with pytest.raises(ModelBehaviorError):
            strict.validate_json('{"answer": "hi", "score": "3"}')

    def test_non_strict_coerces(self):
        lenient = build_output_schema(Extraction, strict_json_schema=False)
        assert lenient.validate_json('{"answer": "hi", "score": "3"}').score == 3


# ------------------------------------------------------------------ #
# salvage_structured_output
# ------------------------------------------------------------------ #


class TestSalvageStructuredOutput:
    def test_clean_payload(self, schema):
        result = salvage_structured_output('{"answer": "yes", "score": 7}', schema)
        assert (result.answer, result.score) == ("yes", 7)

    def test_fenced_payload(self, schema):
        raw = '```json\n{"answer": "yes", "score": 7}\n```'
        assert salvage_structured_output(raw, schema).answer == "yes"

    def test_payload_after_prose(self, schema):
        raw = 'Here is the result:\n{"answer": "yes", "score": 7}'
        assert salvage_structured_output(raw, schema).score == 7

    def test_payload_with_trailing_commentary(self, schema):
        raw = '{"answer": "yes", "score": 7}\n\nLet me know if you need more.'
        assert salvage_structured_output(raw, schema).answer == "yes"

    def test_skips_earlier_non_matching_container(self, schema):
        raw = 'Ignore {"note": "scratch"} and use {"answer": "yes", "score": 7}'
        assert salvage_structured_output(raw, schema).answer == "yes"

    def test_braces_inside_strings_do_not_break_matching(self, schema):
        raw = 'Result: {"answer": "use {curly} braces", "score": 1}'
        assert salvage_structured_output(raw, schema).answer == "use {curly} braces"

    def test_escaped_quote_inside_string(self, schema):
        raw = 'Result: {"answer": "say \\"hi\\"", "score": 1}'
        assert salvage_structured_output(raw, schema).answer == 'say "hi"'

    def test_unbalanced_payload_is_not_salvaged(self, schema):
        assert salvage_structured_output('{"answer": "yes", "score": 7', schema) is None

    def test_schema_violation_is_not_salvaged(self, schema):
        """A well-formed payload missing a required field is a real failure."""
        assert salvage_structured_output('{"answer": "yes"}', schema) is None

    def test_prose_only_is_not_salvaged(self, schema):
        assert salvage_structured_output("I could not complete the task.", schema) is None

    def test_empty_text_is_not_salvaged(self, schema):
        assert salvage_structured_output("", schema) is None

    def test_whitespace_only_text_is_not_salvaged(self, schema):
        assert salvage_structured_output("   \n\t ", schema) is None

    def test_none_text_is_not_salvaged(self, schema):
        assert salvage_structured_output(None, schema) is None

    def test_wrapped_output_type_unwraps(self):
        """List output types are wrapped by the SDK; salvage must unwrap them."""
        wrapped = AgentOutputSchema(list[str])
        raw = '```json\n{"response": ["a", "b"]}\n```'
        assert salvage_structured_output(raw, wrapped) == ["a", "b"]


# ------------------------------------------------------------------ #
# iter_payload_candidates
# ------------------------------------------------------------------ #


class TestIterPayloadCandidates:
    def test_whole_text_comes_first(self):
        assert list(iter_payload_candidates('prose {"a": 1} more'))[0] == 'prose {"a": 1} more'

    def test_yields_the_span_inside_a_fence(self):
        candidates = list(iter_payload_candidates('```json\n{"a": 1}\n```'))
        assert '{"a": 1}' in candidates

    def test_bare_payload_is_not_yielded_twice(self):
        assert list(iter_payload_candidates('{"a": 1}')) == ['{"a": 1}']

    def test_yields_each_span_in_order(self):
        candidates = list(iter_payload_candidates('{"a": 1} then ["b"]'))
        assert candidates[1:] == ['{"a": 1}', '["b"]']

    def test_delimiters_inside_strings_do_not_close_a_span(self):
        candidates = list(iter_payload_candidates('see {"a": "use {braces}"} here'))
        assert candidates[1] == '{"a": "use {braces}"}'

    def test_unbalanced_container_yields_only_the_whole_text(self):
        assert list(iter_payload_candidates('{"a": 1')) == ['{"a": 1']

    def test_empty_text_yields_nothing(self):
        assert list(iter_payload_candidates("")) == []


# ------------------------------------------------------------------ #
# recover_invalid_final_output
# ------------------------------------------------------------------ #


class TestRecoverInvalidFinalOutput:
    def test_recovers_fenced_payload(self):
        handler_input = _handler_input('```json\n{"answer": "yes", "score": 7}\n```')
        assert recover_invalid_final_output(handler_input).answer == "yes"

    def test_returns_none_for_unsalvageable_text(self):
        assert recover_invalid_final_output(_handler_input("no json here")) is None

    def test_returns_none_when_no_message_item(self):
        assert recover_invalid_final_output(_handler_input(None)) is None

    def test_returns_none_for_plain_text_agent(self):
        handler_input = _handler_input('{"answer": "yes", "score": 7}', output_type=None)
        assert recover_invalid_final_output(handler_input) is None

    def test_uses_the_last_message(self):
        """The failing message is the most recent one, not the first."""
        handler_input = _handler_input('{"answer": "second", "score": 2}')
        handler_input.run_data.new_items = [
            _message_item('{"answer": "first", "score": 1}'),
            _message_item('{"answer": "second", "score": 2}'),
        ]
        assert recover_invalid_final_output(handler_input).answer == "second"

    def test_recovered_output_survives_sdk_revalidation(self):
        """The SDK re-validates whatever the handler returns; it must pass."""
        from agents.run_internal.error_handlers import validate_handler_final_output

        agent = Agent(name="extractor", instructions="extract", output_type=Extraction)
        recovered = recover_invalid_final_output(
            _handler_input('Here you go:\n{"answer": "yes", "score": 7}')
        )

        assert validate_handler_final_output(agent, recovered).answer == "yes"


# ------------------------------------------------------------------ #
# build_error_handlers
# ------------------------------------------------------------------ #


class TestBuildErrorHandlers:
    def test_structured_agent_gets_the_recovery_handler(self):
        agent = Agent(name="extractor", output_type=Extraction)

        handlers = build_error_handlers(agent)

        assert handlers == {"invalid_final_output": recover_invalid_final_output}

    def test_agent_with_a_schema_instance_gets_the_recovery_handler(self, schema):
        assert build_error_handlers(Agent(name="extractor", output_type=schema)) is not None

    def test_plain_text_agent_gets_no_handlers(self):
        assert build_error_handlers(Agent(name="a")) is None

    def test_str_output_type_gets_no_handlers(self):
        assert build_error_handlers(Agent(name="a", output_type=str)) is None


class TestDeclaredInvalidOutputRecovery:
    """The flag has no slot on ``Agent``, so it is resolved through the registry.

    Without it, an agent that declares ``invalid_output_recovery: false`` fails
    loudly when the runner executes it and is silently recovered when the chat
    service does — the same declaration honored on one path and dropped on the
    other.
    """

    def test_an_agent_opting_out_gets_no_handlers(self, recovery_registry):
        assert build_error_handlers(Agent(name="strict_agent", output_type=Extraction)) is None

    def test_an_agent_keeping_the_default_gets_the_recovery_handler(self, recovery_registry):
        handlers = build_error_handlers(Agent(name="recovering_agent", output_type=Extraction))

        assert handlers == {"invalid_final_output": recover_invalid_final_output}

    def test_an_unregistered_agent_keeps_recovery(self, recovery_registry):
        """A hand-assembled agent under an unknown name has no opt-out to honor."""
        agent = Agent(name="assembled_by_hand", output_type=Extraction)

        assert build_error_handlers(agent) is not None

    def test_a_plain_text_agent_opting_out_gets_no_handlers(self, recovery_registry):
        """Both reasons to skip recovery apply at once and still yield defaults."""
        assert build_error_handlers(Agent(name="strict_agent")) is None


class TestInvalidFinalOutputHandlers:
    def test_registers_only_the_invalid_final_output_key(self):
        assert list(invalid_final_output_handlers()) == ["invalid_final_output"]

    def test_each_call_returns_a_fresh_mapping(self):
        first = invalid_final_output_handlers()
        second = invalid_final_output_handlers()

        first.clear()

        assert first is not second
        assert "invalid_final_output" in second
