"""Tests for metrics.prometheus (PrometheusMetrics, _HistogramData)."""
from __future__ import annotations

import threading

import pytest

from agent_observability.metrics.prometheus import PrometheusMetrics, _HistogramData

_LATENCY_BUCKETS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


# ── _HistogramData ─────────────────────────────────────────────────────────────


class TestHistogramData:
    def test_observe_increments_count(self) -> None:
        h = _HistogramData()
        h.observe(0.1, _LATENCY_BUCKETS)
        assert h.count == 1

    def test_observe_accumulates_total(self) -> None:
        h = _HistogramData()
        h.observe(1.5, _LATENCY_BUCKETS)
        h.observe(2.5, _LATENCY_BUCKETS)
        assert h.total == pytest.approx(4.0)

    def test_observe_fills_correct_buckets(self) -> None:
        h = _HistogramData()
        h.observe(0.05, _LATENCY_BUCKETS)
        # Buckets >= 0.05 should have count 1
        assert h.buckets.get(0.05, 0) == 1
        assert h.buckets.get(0.1, 0) == 1
        assert h.buckets.get(1.0, 0) == 1
        # Buckets < 0.05 should have count 0 (not present)
        assert h.buckets.get(0.025, 0) == 0
        assert h.buckets.get(0.01, 0) == 0

    def test_observe_always_increments_inf_bucket(self) -> None:
        h = _HistogramData()
        h.observe(999.9, _LATENCY_BUCKETS)
        assert h.buckets.get(float("inf"), 0) == 1

    def test_multiple_observations_cumulative_buckets(self) -> None:
        h = _HistogramData()
        h.observe(0.5, _LATENCY_BUCKETS)  # fits in 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, inf
        h.observe(0.5, _LATENCY_BUCKETS)  # same
        assert h.buckets.get(0.5, 0) == 2
        assert h.buckets.get(float("inf"), 0) == 2

    def test_observe_zero_value(self) -> None:
        h = _HistogramData()
        h.observe(0.0, _LATENCY_BUCKETS)
        # 0.0 is <= all bounds
        assert h.buckets.get(0.005, 0) == 1
        assert h.count == 1

    def test_observe_above_all_bounds(self) -> None:
        h = _HistogramData()
        h.observe(100.0, _LATENCY_BUCKETS)  # exceeds all standard buckets
        # No finite bucket should be incremented
        for bound in _LATENCY_BUCKETS:
            assert h.buckets.get(bound, 0) == 0
        # But +Inf must be
        assert h.buckets.get(float("inf"), 0) == 1


# ── PrometheusMetrics ──────────────────────────────────────────────────────────


@pytest.fixture()
def metrics() -> PrometheusMetrics:
    return PrometheusMetrics()


class TestPrometheusMetricsInit:
    def test_default_namespace(self, metrics: PrometheusMetrics) -> None:
        assert metrics._namespace == "agent"

    def test_custom_namespace(self) -> None:
        m = PrometheusMetrics(namespace="myapp")
        assert m._namespace == "myapp"

    def test_initial_state_empty(self, metrics: PrometheusMetrics) -> None:
        assert metrics._llm_calls_total == {}
        assert metrics._tool_invocations_total == {}
        assert metrics._errors_total == {}
        assert metrics._llm_latency == {}
        assert metrics._tool_latency == {}
        assert metrics._active_spans == 0.0
        assert metrics._total_cost_usd == 0.0


class TestRecordLlmCall:
    def test_increments_call_counter(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=1.0, tokens=100, cost=0.01)
        assert metrics._llm_calls_total["gpt-4o"] == 1

    def test_increments_counter_multiple_calls(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=50, cost=0.005)
        metrics.record_llm_call("gpt-4o", latency_seconds=0.8, tokens=80, cost=0.008)
        assert metrics._llm_calls_total["gpt-4o"] == 2

    def test_tracks_multiple_models(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        metrics.record_llm_call("claude-sonnet-4", latency_seconds=0.3, tokens=50, cost=0.005)
        assert metrics._llm_calls_total["gpt-4o"] == 1
        assert metrics._llm_calls_total["claude-sonnet-4"] == 1

    def test_accumulates_cost(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=1.0, tokens=100, cost=0.01)
        metrics.record_llm_call("gpt-4o", latency_seconds=1.0, tokens=100, cost=0.02)
        assert metrics._total_cost_usd == pytest.approx(0.03)

    def test_creates_latency_histogram(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        assert "gpt-4o" in metrics._llm_latency
        hist = metrics._llm_latency["gpt-4o"]
        assert hist.count == 1
        assert hist.total == pytest.approx(0.5)

    def test_zero_cost_allowed(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("local-model", latency_seconds=0.1, tokens=10, cost=0.0)
        assert metrics._llm_calls_total["local-model"] == 1
        assert metrics._total_cost_usd == 0.0


class TestRecordToolCall:
    def test_increments_tool_counter(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("web_search", latency_seconds=0.2, success=True)
        assert metrics._tool_invocations_total["web_search"] == 1

    def test_success_does_not_increment_error_counter(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("web_search", latency_seconds=0.2, success=True)
        assert metrics._errors_total == {}

    def test_failure_increments_error_counter(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("web_search", latency_seconds=0.2, success=False)
        error_key = "tool_error:web_search"
        assert metrics._errors_total.get(error_key, 0) == 1

    def test_creates_tool_latency_histogram(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("calculator", latency_seconds=0.05, success=True)
        assert "calculator" in metrics._tool_latency
        hist = metrics._tool_latency["calculator"]
        assert hist.count == 1

    def test_multiple_tools_tracked_independently(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("search", latency_seconds=0.1, success=True)
        metrics.record_tool_call("calculator", latency_seconds=0.05, success=True)
        assert metrics._tool_invocations_total["search"] == 1
        assert metrics._tool_invocations_total["calculator"] == 1

    def test_multiple_failures_accumulate(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("search", latency_seconds=0.1, success=False)
        metrics.record_tool_call("search", latency_seconds=0.1, success=False)
        assert metrics._errors_total["tool_error:search"] == 2


class TestRecordError:
    def test_increments_error_counter(self, metrics: PrometheusMetrics) -> None:
        metrics.record_error("RateLimitError")
        assert metrics._errors_total["RateLimitError"] == 1

    def test_increments_same_error_multiple_times(self, metrics: PrometheusMetrics) -> None:
        metrics.record_error("TimeoutError")
        metrics.record_error("TimeoutError")
        assert metrics._errors_total["TimeoutError"] == 2

    def test_multiple_error_types(self, metrics: PrometheusMetrics) -> None:
        metrics.record_error("RateLimitError")
        metrics.record_error("TimeoutError")
        assert metrics._errors_total["RateLimitError"] == 1
        assert metrics._errors_total["TimeoutError"] == 1


class TestSetActiveSpans:
    def test_sets_gauge_value(self, metrics: PrometheusMetrics) -> None:
        metrics.set_active_spans(5.0)
        assert metrics._active_spans == 5.0

    def test_overrides_previous_value(self, metrics: PrometheusMetrics) -> None:
        metrics.set_active_spans(3.0)
        metrics.set_active_spans(7.0)
        assert metrics._active_spans == 7.0

    def test_set_to_zero(self, metrics: PrometheusMetrics) -> None:
        metrics.set_active_spans(10.0)
        metrics.set_active_spans(0.0)
        assert metrics._active_spans == 0.0


class TestReset:
    def test_clears_all_counters(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        metrics.record_tool_call("search", latency_seconds=0.1, success=True)
        metrics.record_error("RateLimitError")
        metrics.set_active_spans(3.0)
        metrics.reset()
        assert metrics._llm_calls_total == {}
        assert metrics._tool_invocations_total == {}
        assert metrics._errors_total == {}
        assert metrics._llm_latency == {}
        assert metrics._tool_latency == {}
        assert metrics._active_spans == 0.0
        assert metrics._total_cost_usd == 0.0

    def test_reset_idempotent(self, metrics: PrometheusMetrics) -> None:
        metrics.reset()
        metrics.reset()
        assert metrics._llm_calls_total == {}


class TestExport:
    def test_export_empty_metrics_contains_gauges(self, metrics: PrometheusMetrics) -> None:
        output = metrics.export()
        assert "agent_active_spans" in output
        assert "agent_total_cost_usd" in output

    def test_export_includes_llm_counter_after_record(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=1.0, tokens=100, cost=0.01)
        output = metrics.export()
        assert "agent_llm_calls_total" in output
        assert "gpt-4o" in output

    def test_export_includes_tool_counter_after_record(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("search", latency_seconds=0.1, success=True)
        output = metrics.export()
        assert "agent_tool_invocations_total" in output
        assert "search" in output

    def test_export_includes_error_counter_after_record(self, metrics: PrometheusMetrics) -> None:
        metrics.record_error("RateLimitError")
        output = metrics.export()
        assert "agent_errors_total" in output
        assert "RateLimitError" in output

    def test_export_includes_histogram_blocks(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        output = metrics.export()
        assert "agent_llm_latency_seconds" in output
        assert "_bucket" in output
        assert "_count" in output
        assert "_sum" in output

    def test_export_tool_latency_histogram(self, metrics: PrometheusMetrics) -> None:
        metrics.record_tool_call("calculator", latency_seconds=0.05, success=True)
        output = metrics.export()
        assert "agent_tool_latency_seconds" in output

    def test_export_help_and_type_comments(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        output = metrics.export()
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_export_counter_type_annotation(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        output = metrics.export()
        assert "# TYPE agent_llm_calls_total counter" in output

    def test_export_gauge_type_annotation(self, metrics: PrometheusMetrics) -> None:
        output = metrics.export()
        assert "# TYPE agent_active_spans gauge" in output

    def test_export_histogram_type_annotation(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        output = metrics.export()
        assert "# TYPE agent_llm_latency_seconds histogram" in output

    def test_export_active_spans_value_matches(self, metrics: PrometheusMetrics) -> None:
        metrics.set_active_spans(7.0)
        output = metrics.export()
        assert "agent_active_spans 7.0" in output

    def test_export_total_cost_value_matches(self, metrics: PrometheusMetrics) -> None:
        metrics.record_llm_call("gpt-4o", latency_seconds=1.0, tokens=100, cost=0.01234)
        output = metrics.export()
        # The cost uses 8 decimal places format
        assert "agent_total_cost_usd" in output
        assert "0.01234" in output

    def test_export_label_escaping(self, metrics: PrometheusMetrics) -> None:
        # Model names with special characters get label-escaped
        metrics.record_llm_call('gpt-4o"turbo', latency_seconds=0.5, tokens=100, cost=0.01)
        output = metrics.export()
        # Should not crash and should be valid text
        assert "agent_llm_calls_total" in output

    def test_export_custom_namespace(self) -> None:
        m = PrometheusMetrics(namespace="myservice")
        m.record_llm_call("gpt-4o", latency_seconds=0.5, tokens=100, cost=0.01)
        output = m.export()
        assert "myservice_llm_calls_total" in output
        assert "myservice_active_spans" in output

    def test_export_no_llm_counter_if_no_calls(self, metrics: PrometheusMetrics) -> None:
        output = metrics.export()
        assert "agent_llm_calls_total" not in output

    def test_export_no_tool_counter_if_no_calls(self, metrics: PrometheusMetrics) -> None:
        output = metrics.export()
        assert "agent_tool_invocations_total" not in output


class TestPrometheusMetricsThreadSafety:
    def test_concurrent_llm_calls_no_race(self) -> None:
        m = PrometheusMetrics()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    m.record_llm_call("gpt-4o", latency_seconds=0.1, tokens=10, cost=0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert m._llm_calls_total.get("gpt-4o", 0) == 200

    def test_concurrent_tool_calls_no_race(self) -> None:
        m = PrometheusMetrics()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    m.record_tool_call("search", latency_seconds=0.05, success=True)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert m._tool_invocations_total.get("search", 0) == 200


class TestCounterBlock:
    def test_empty_data_returns_empty_list(self) -> None:
        result = PrometheusMetrics._counter_block(
            name="test_counter",
            help_text="Test counter",
            label_name="model",
            data={},
        )
        assert result == []

    def test_non_empty_includes_help_and_type(self) -> None:
        result = PrometheusMetrics._counter_block(
            name="test_counter",
            help_text="Test counter",
            label_name="model",
            data={"gpt-4o": 5},
        )
        assert any("# HELP test_counter Test counter" in line for line in result)
        assert any("# TYPE test_counter counter" in line for line in result)

    def test_sorted_label_output(self) -> None:
        result = PrometheusMetrics._counter_block(
            name="test_counter",
            help_text="Test",
            label_name="model",
            data={"z_model": 1, "a_model": 2},
        )
        metric_lines = [l for l in result if "{" in l]
        assert "a_model" in metric_lines[0]  # 'a' sorts before 'z'
        assert "z_model" in metric_lines[1]
