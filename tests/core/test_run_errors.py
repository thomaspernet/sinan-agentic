"""Tests for typed classification of agent-run failures (core/run_errors.py)."""

import json

import httpx
import pytest
from agents import MaxTurnsExceeded, ModelBehaviorError, ModelRefusalError
from openai import BadRequestError, RateLimitError

from sinan_agentic_core.core.run_errors import (
    FALLBACK_RECOVERABLE_KINDS,
    RunErrorKind,
    classify_run_error,
    run_error_payload,
)
from tests.conftest import make_context_overflow_error


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

    def test_context_overflow_reads_the_provider_error_code(self):
        assert classify_run_error(make_context_overflow_error()) is RunErrorKind.CONTEXT_OVERFLOW

    def test_other_provider_error_is_unknown(self):
        """A 429 is a provider error too -- only the overflow code counts."""
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = RateLimitError(
            "Rate limit reached",
            response=httpx.Response(429, request=request),
            body={"code": "rate_limit_exceeded", "type": "requests"},
        )
        assert classify_run_error(error) is RunErrorKind.UNKNOWN

    def test_provider_error_without_a_code_is_unknown(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = BadRequestError(
            "Bad request",
            response=httpx.Response(400, request=request),
            body=None,
        )
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
    """Only the two out-of-room failures are worth a condensed second call."""

    def test_covers_max_turns_and_overflow(self):
        assert FALLBACK_RECOVERABLE_KINDS == {
            RunErrorKind.MAX_TURNS,
            RunErrorKind.CONTEXT_OVERFLOW,
        }

    @pytest.mark.parametrize(
        "kind",
        [RunErrorKind.MODEL_REFUSAL, RunErrorKind.MODEL_BEHAVIOR, RunErrorKind.UNKNOWN],
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
