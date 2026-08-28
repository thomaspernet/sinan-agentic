"""Tests for the reported token-usage record (core/usage.py)."""

from __future__ import annotations

from unittest.mock import Mock

from sinan_agentic_core.core.usage import (
    aggregate_usage,
    completion_usage_record,
    last_input_tokens,
    usage_record,
)
from tests.conftest import make_completion_usage, make_model_response


class TestTheRecordShape:
    def test_the_counts_land_under_the_names_a_consumer_reads(self) -> None:
        record = usage_record(
            requests=2,
            input_tokens=300,
            output_tokens=40,
            total_tokens=340,
            cached_tokens=256,
            reasoning_tokens=8,
        )

        assert record == {
            "requests": 2,
            "input_tokens": 300,
            "output_tokens": 40,
            "total_tokens": 340,
            "input_tokens_details": {"cached_tokens": 256},
            "output_tokens_details": {"reasoning_tokens": 8},
        }

    def test_two_records_do_not_share_their_nested_details(self) -> None:
        """Each caller owns what it is handed, nested mappings included."""
        counts = {
            "requests": 1,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        first = usage_record(**counts)
        second = usage_record(**counts)

        first["input_tokens_details"]["cached_tokens"] = 42

        assert second["input_tokens_details"]["cached_tokens"] == 0


class TestAggregatingARun:
    def test_a_single_response_reports_the_cached_count_it_carried(self) -> None:
        result = Mock()
        result.raw_responses = [make_model_response(cached_tokens=64)]

        assert aggregate_usage(result)["input_tokens_details"]["cached_tokens"] == 64

    def test_the_cached_count_sums_over_every_response_of_the_run(self) -> None:
        """A run is billed per call, so its cached prefix is counted per call too."""
        result = Mock()
        result.raw_responses = [
            make_model_response(input_tokens=100, cached_tokens=0),
            make_model_response(input_tokens=400, cached_tokens=256),
            make_model_response(input_tokens=700, cached_tokens=512),
        ]

        record = aggregate_usage(result)

        assert record["requests"] == 3
        assert record["input_tokens"] == 1200
        assert record["input_tokens_details"]["cached_tokens"] == 768

    def test_reasoning_tokens_accumulate_the_same_way(self) -> None:
        result = Mock()
        result.raw_responses = [
            make_model_response(reasoning_tokens=5),
            make_model_response(reasoning_tokens=7),
        ]

        assert aggregate_usage(result)["output_tokens_details"]["reasoning_tokens"] == 12

    def test_a_run_with_no_responses_reports_zeros(self) -> None:
        result = Mock()
        result.raw_responses = []

        assert aggregate_usage(result)["total_tokens"] == 0

    def test_a_result_that_does_not_carry_responses_reports_zeros(self) -> None:
        """A bare mock's ``raw_responses`` is not iterable; a run still reports."""
        assert aggregate_usage(Mock())["total_tokens"] == 0


class TestTheContextWindowHighWaterMark:
    def test_the_last_call_input_is_reported_not_the_sum(self) -> None:
        """Summed input counts the replayed history once per call."""
        result = Mock()
        result.raw_responses = [
            make_model_response(input_tokens=100),
            make_model_response(input_tokens=400),
            make_model_response(input_tokens=700),
        ]

        assert last_input_tokens(result) == 700

    def test_a_trailing_response_without_usage_falls_back_to_the_last_that_had_it(self) -> None:
        without_usage = Mock()
        without_usage.usage = None
        result = Mock()
        result.raw_responses = [make_model_response(input_tokens=400), without_usage]

        assert last_input_tokens(result) == 400

    def test_a_run_with_no_responses_reports_zero(self) -> None:
        result = Mock()
        result.raw_responses = []

        assert last_input_tokens(result) == 0


class TestAChatCompletionsResponse:
    """The shape a branch bypassing the SDK gets back from the provider."""

    def test_the_provider_cached_count_is_reported(self) -> None:
        record = completion_usage_record(make_completion_usage(cached_tokens=256))

        assert record["input_tokens_details"]["cached_tokens"] == 256

    def test_the_provider_reasoning_count_is_reported(self) -> None:
        record = completion_usage_record(make_completion_usage(reasoning_tokens=9))

        assert record["output_tokens_details"]["reasoning_tokens"] == 9

    def test_one_response_counts_as_one_request(self) -> None:
        assert completion_usage_record(make_completion_usage())["requests"] == 1

    def test_the_totals_come_from_the_provider(self) -> None:
        record = completion_usage_record(
            make_completion_usage(prompt_tokens=800, completion_tokens=30)
        )

        assert record["input_tokens"] == 800
        assert record["output_tokens"] == 30
        assert record["total_tokens"] == 830

    def test_a_provider_that_omits_both_detail_blocks_reports_zeros(self) -> None:
        """Both blocks are optional on the provider's type."""
        usage = make_completion_usage()
        usage.prompt_tokens_details = None
        usage.completion_tokens_details = None

        record = completion_usage_record(usage)

        assert record["input_tokens_details"]["cached_tokens"] == 0
        assert record["output_tokens_details"]["reasoning_tokens"] == 0

    def test_a_provider_that_omits_the_counts_inside_them_reports_zeros(self) -> None:
        """Every count inside a detail block is nullable on the provider's type."""
        usage = make_completion_usage()
        usage.prompt_tokens_details.cached_tokens = None
        usage.completion_tokens_details.reasoning_tokens = None

        record = completion_usage_record(usage)

        assert record["input_tokens_details"]["cached_tokens"] == 0
        assert record["output_tokens_details"]["reasoning_tokens"] == 0
