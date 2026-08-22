"""Tests for typed classification of agent-run failures (core/run_errors.py)."""

import json

import pytest
from agents import MaxTurnsExceeded, ModelBehaviorError, ModelRefusalError, ModelTimeoutError
from openai import BadRequestError, RateLimitError

from sinan_agentic_core.core.run_errors import (
    FALLBACK_RECOVERABLE_KINDS,
    RunErrorKind,
    classify_run_error,
    run_error_payload,
)
from tests.conftest import (
    make_context_overflow_error,
    make_input_tripwire_error,
    make_output_tripwire_error,
    make_provider_status_error,
    registered_input_guardrail,
    registered_output_guardrail,
)


class TestClassifyRunError:
    """classify_run_error keys off exception type, never message text."""

    def test_max_turns_is_typed(self):
        error = MaxTurnsExceeded("Max turns (10) exceeded")
        assert classify_run_error(error) is RunErrorKind.MAX_TURNS

    def test_model_refusal_is_typed(self):
        error = ModelRefusalError("I can't help with that.")
        assert classify_run_error(error) is RunErrorKind.MODEL_REFUSAL

    def test_model_behavior_is_typed(self):
        error = ModelBehaviorError("Invalid JSON in final output")
        assert classify_run_error(error) is RunErrorKind.MODEL_BEHAVIOR

    def test_model_timeout_is_typed(self):
        """A declared model_timeout firing is the caller's own bound, not a crash."""
        error = ModelTimeoutError(30.0)
        assert classify_run_error(error) is RunErrorKind.MODEL_TIMEOUT

    def test_context_overflow_reads_the_provider_error_code(self):
        assert classify_run_error(make_context_overflow_error()) is RunErrorKind.CONTEXT_OVERFLOW

    def test_input_tripwire_is_typed(self):
        error = make_input_tripwire_error(registered_input_guardrail("blocks_pii"))
        assert classify_run_error(error) is RunErrorKind.INPUT_GUARDRAIL_TRIPWIRE

    def test_output_tripwire_is_typed(self):
        error = make_output_tripwire_error(registered_output_guardrail("blocks_secrets"))
        assert classify_run_error(error) is RunErrorKind.OUTPUT_GUARDRAIL_TRIPWIRE

    def test_the_two_tripwires_are_told_apart(self):
        """One kind for both would leave callers reading a nested field to know
        whether anything ran at all."""
        rejected_input = make_input_tripwire_error(registered_input_guardrail("blocks_pii"))
        blocked_answer = make_output_tripwire_error(registered_output_guardrail("blocks_pii"))
        assert classify_run_error(rejected_input) is not classify_run_error(blocked_answer)

    def test_other_provider_error_is_unknown(self):
        """A 429 is a provider error too -- only the overflow code counts."""
        error = make_provider_status_error(
            RateLimitError,
            "Rate limit reached",
            429,
            {"code": "rate_limit_exceeded", "type": "requests"},
        )
        assert classify_run_error(error) is RunErrorKind.UNKNOWN

    def test_provider_error_without_a_code_is_unknown(self):
        error = make_provider_status_error(BadRequestError, "Bad request", 400, None)
        assert classify_run_error(error) is RunErrorKind.UNKNOWN

    def test_plain_exception_is_unknown(self):
        assert classify_run_error(RuntimeError("Something else broke")) is RunErrorKind.UNKNOWN

    @pytest.mark.parametrize(
        "message",
        [
            "Max turns exceeded",
            "the tool returned context_length_exceeded in its payload",
        ],
    )
    def test_message_text_alone_no_longer_classifies(self, message):
        """Regression for #47 -- an error that merely quotes the old needles
        must not be mistaken for the failure it names."""
        assert classify_run_error(RuntimeError(message)) is RunErrorKind.UNKNOWN


class TestFallbackRecoverableKinds:
    """Only the two out-of-room failures are worth a condensed second call.

    A guardrail tripwire is deliberately outside the set: it is a declared check
    saying no, so a second call that reaches a different answer has defeated the
    guardrail rather than recovered from a limit.
    """

    def test_covers_max_turns_and_overflow(self):
        assert FALLBACK_RECOVERABLE_KINDS == {
            RunErrorKind.MAX_TURNS,
            RunErrorKind.CONTEXT_OVERFLOW,
        }

    @pytest.mark.parametrize(
        "kind",
        [
            RunErrorKind.MODEL_REFUSAL,
            RunErrorKind.MODEL_BEHAVIOR,
            RunErrorKind.MODEL_TIMEOUT,
            RunErrorKind.INPUT_GUARDRAIL_TRIPWIRE,
            RunErrorKind.OUTPUT_GUARDRAIL_TRIPWIRE,
            RunErrorKind.UNKNOWN,
        ],
    )
    def test_excludes_everything_else(self, kind):
        assert kind not in FALLBACK_RECOVERABLE_KINDS


class TestRunErrorPayload:
    """The classified kind travels beside the message for callers that get a dict."""

    def test_carries_message_and_kind(self):
        assert run_error_payload(MaxTurnsExceeded("Max turns (10) exceeded")) == {
            "error": "Max turns (10) exceeded",
            "error_kind": RunErrorKind.MAX_TURNS.value,
        }

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (ModelRefusalError("I can't help with that."), RunErrorKind.MODEL_REFUSAL),
            (ModelBehaviorError("Invalid JSON in final output"), RunErrorKind.MODEL_BEHAVIOR),
            (ModelTimeoutError(30.0), RunErrorKind.MODEL_TIMEOUT),
            (RuntimeError("Something else broke"), RunErrorKind.UNKNOWN),
        ],
    )
    def test_kind_matches_the_classification(self, error, expected):
        assert run_error_payload(error)["error_kind"] == expected.value

    def test_context_overflow_reaches_the_payload(self):
        payload = run_error_payload(make_context_overflow_error())
        assert payload["error_kind"] == RunErrorKind.CONTEXT_OVERFLOW.value

    def test_payload_is_json_serializable(self):
        """The payload crosses an API boundary, so the kind is a plain string."""
        payload = run_error_payload(MaxTurnsExceeded("Max turns (10) exceeded"))
        assert json.loads(json.dumps(payload)) == payload
        assert type(payload["error_kind"]) is str


class TestGuardrailTripwirePayload:
    """A tripwire message names the SDK class, so the payload names the guardrail."""

    def test_sdk_message_alone_cannot_identify_the_guardrail(self):
        """The premise: two different guardrails render the identical message."""
        pii = run_error_payload(make_input_tripwire_error(registered_input_guardrail("blocks_pii")))
        topic = run_error_payload(
            make_input_tripwire_error(registered_input_guardrail("off_topic"))
        )

        assert pii["error"] == topic["error"]
        assert pii["guardrail"]["name"] != topic["guardrail"]["name"]

    def test_names_the_guardrail_that_tripped(self):
        error = make_input_tripwire_error(registered_input_guardrail("blocks_pii"))
        assert run_error_payload(error)["guardrail"]["name"] == "blocks_pii"

    def test_results_carry_every_guardrail_that_finished(self):
        """0.19.2 records each result as its guardrail completes, so the passing
        ones survive the raise instead of being discarded with the accumulator."""
        error = make_input_tripwire_error(
            registered_input_guardrail("blocks_pii"),
            passed=[registered_input_guardrail("off_topic")],
        )

        assert run_error_payload(error)["guardrail"]["results"] == [
            {"name": "off_topic", "tripwire_triggered": False},
            {"name": "blocks_pii", "tripwire_triggered": True},
        ]

    def test_the_name_survives_missing_run_data(self):
        """The tripping guardrail comes off the exception itself, so it is
        reported even when the SDK attached no run data."""
        error = make_input_tripwire_error(
            registered_input_guardrail("blocks_pii"), with_run_data=False
        )

        details = run_error_payload(error)["guardrail"]
        assert details["name"] == "blocks_pii"
        assert details["results"] == []

    def test_output_tripwire_reads_the_output_results(self):
        error = make_output_tripwire_error(
            registered_output_guardrail("blocks_secrets"),
            passed=[registered_output_guardrail("checks_tone")],
        )

        payload = run_error_payload(error)
        assert payload["error_kind"] == RunErrorKind.OUTPUT_GUARDRAIL_TRIPWIRE.value
        assert payload["guardrail"] == {
            "name": "blocks_secrets",
            "results": [
                {"name": "checks_tone", "tripwire_triggered": False},
                {"name": "blocks_secrets", "tripwire_triggered": True},
            ],
        }

    @pytest.mark.parametrize(
        "error",
        [
            MaxTurnsExceeded("Max turns (10) exceeded"),
            ModelRefusalError("I can't help with that."),
            RuntimeError("Something else broke"),
        ],
    )
    def test_non_tripwire_failures_carry_no_guardrail_entry(self, error):
        assert "guardrail" not in run_error_payload(error)

    def test_tripwire_payload_is_json_serializable(self):
        """The guardrail's own ``output_info`` is arbitrary, so the payload
        reports names and flags -- everything here crosses an API boundary."""
        payload = run_error_payload(
            make_input_tripwire_error(
                registered_input_guardrail("blocks_pii"),
                passed=[registered_input_guardrail("off_topic")],
            )
        )
        assert json.loads(json.dumps(payload)) == payload
