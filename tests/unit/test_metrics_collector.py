"""Comprehensive tests for metrics.collector and metrics.definitions."""
from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_observability.metrics.collector import (
    AgentMetricCollector,
    CounterValue,
    GaugeValue,
    HistogramValue,
)
from agent_observability.metrics.definitions import (
    ALL_DEFINITIONS,
    AGENT_ACTIVE_COUNT,
    AGENT_COST_USD_TOTAL,
    AGENT_DRIFT_Z_SCORE,
    AGENT_ERRORS_TOTAL,
    AGENT_LATENCY_SECONDS,
    AGENT_LLM_CALLS_TOTAL,
    AGENT_MEMORY_OPERATIONS_TOTAL,
    AGENT_TOKENS_TOTAL,
    AGENT_TOOL_INVOCATIONS_TOTAL,
    MetricDefinition,
)


# ── CounterValue ───────────────────────────────────────────────────────────────


class TestCounterValue:
    def test_initial_value(self) -> None:
        counter = CounterValue(name="test_counter", labels={})
        assert counter.value == 0.0

    def test_increment_default(self) -> None:
        counter = CounterValue(name="test_counter", labels={})
        counter.inc()
        assert counter.value == 1.0

    def test_increment_by_amount(self) -> None:
        counter = CounterValue(name="test_counter", labels={})
        counter.inc(5.0)
        assert counter.value == 5.0

    def test_multiple_increments(self) -> None:
        counter = CounterValue(name="test_counter", labels={"env": "prod"})
        counter.inc(3.0)
        counter.inc(2.5)
        assert counter.value == pytest.approx(5.5)


# ── GaugeValue ─────────────────────────────────────────────────────────────────


class TestGaugeValue:
    def test_set_absolute(self) -> None:
        gauge = GaugeValue(name="test_gauge", labels={})
        gauge.set(42.0)
        assert gauge.value == 42.0

    def test_increment(self) -> None:
        gauge = GaugeValue(name="test_gauge", labels={})
        gauge.inc(5.0)
        assert gauge.value == 5.0

    def test_decrement(self) -> None:
        gauge = GaugeValue(name="test_gauge", labels={})
        gauge.set(10.0)
        gauge.dec(3.0)
        assert gauge.value == 7.0

    def test_default_inc_dec_amounts(self) -> None:
        gauge = GaugeValue(name="test_gauge", labels={})
        gauge.inc()
        gauge.dec()
        assert gauge.value == 0.0


# ── HistogramValue ─────────────────────────────────────────────────────────────


class TestHistogramValue:
    def test_initial_state(self) -> None:
        hist = HistogramValue(name="test_hist", labels={})
        assert hist.count == 0
        assert hist.sum_value == 0.0

    def test_observe_single(self) -> None:
        hist = HistogramValue(name="test_hist", labels={})
        hist.observe(0.05)
        assert hist.count == 1
        assert hist.sum_value == pytest.approx(0.05)

    def test_observe_multiple(self) -> None:
        hist = HistogramValue(name="test_hist", labels={})
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(2.0)
        assert hist.count == 3

    def test_bucket_counts_cumulative(self) -> None:
        hist = HistogramValue(name="test_hist", labels={})
        hist.observe(0.005)
        hist.observe(0.05)
        hist.observe(5.0)
        buckets = hist.bucket_counts()
        # Verify cumulative counts are non-decreasing
        for i in range(len(buckets) - 1):
            assert buckets[i][1] <= buckets[i + 1][1]

    def test_bucket_counts_length_matches_buckets(self) -> None:
        hist = HistogramValue(name="test_hist", labels={})
        hist.observe(1.0)
        buckets = hist.bucket_counts()
        assert len(buckets) == len(hist.buckets)

    def test_value_above_all_buckets(self) -> None:
        hist = HistogramValue(name="test_hist", labels={})
        hist.observe(100.0)  # exceeds all default bucket bounds
        assert hist.count == 1
        # No bucket should count this observation
        for _, count in hist.bucket_counts():
            assert count == 0


# ── AgentMetricCollector ───────────────────────────────────────────────────────


class TestAgentMetricCollector:
    def setup_method(self) -> None:
        self.collector = AgentMetricCollector()

    def test_increment_counter_creates_and_increments(self) -> None:
        self.collector.increment_counter("my_counter")
        snap = self.collector.snapshot()
        counters = snap["counters"]
        assert any(c["name"] == "my_counter" and c["value"] == 1.0 for c in counters)

    def test_increment_counter_with_labels(self) -> None:
        self.collector.increment_counter("my_counter", labels={"env": "prod"})
        self.collector.increment_counter("my_counter", labels={"env": "dev"})
        snap = self.collector.snapshot()
        counters = snap["counters"]
        assert len([c for c in counters if c["name"] == "my_counter"]) == 2

    def test_increment_counter_accumulates(self) -> None:
        self.collector.increment_counter("counter_a", amount=5.0)
        self.collector.increment_counter("counter_a", amount=3.0)
        snap = self.collector.snapshot()
        counter = next(c for c in snap["counters"] if c["name"] == "counter_a")
        assert counter["value"] == 8.0

    def test_set_gauge(self) -> None:
        self.collector.set_gauge("my_gauge", 99.5)
        snap = self.collector.snapshot()
        gauges = snap["gauges"]
        assert any(g["name"] == "my_gauge" and g["value"] == 99.5 for g in gauges)

    def test_increment_gauge(self) -> None:
        self.collector.increment_gauge("active_agents")
        self.collector.increment_gauge("active_agents")
        snap = self.collector.snapshot()
        gauge = next(g for g in snap["gauges"] if g["name"] == "active_agents")
        assert gauge["value"] == 2.0

    def test_decrement_gauge(self) -> None:
        self.collector.set_gauge("conn_pool", 5.0)
        self.collector.decrement_gauge("conn_pool")
        snap = self.collector.snapshot()
        gauge = next(g for g in snap["gauges"] if g["name"] == "conn_pool")
        assert gauge["value"] == 4.0

    def test_observe_histogram(self) -> None:
        self.collector.observe_histogram("latency_s", 0.25)
        snap = self.collector.snapshot()
        hists = snap["histograms"]
        assert any(h["name"] == "latency_s" and h["count"] == 1 for h in hists)

    def test_observe_histogram_multiple(self) -> None:
        for val in [0.1, 0.5, 1.0]:
            self.collector.observe_histogram("latency_s", val)
        snap = self.collector.snapshot()
        hist = next(h for h in snap["histograms"] if h["name"] == "latency_s")
        assert hist["count"] == 3

    def test_record_llm_call(self) -> None:
        self.collector.record_llm_call(
            agent_id="agent-1",
            model="claude-sonnet-4",
            provider="anthropic",
            status="success",
            latency_seconds=1.5,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.005,
        )
        snap = self.collector.snapshot()
        counter_names = {c["name"] for c in snap["counters"]}
        assert "agent_llm_calls_total" in counter_names
        assert "agent_tokens_total" in counter_names
        assert "agent_cost_usd_total" in counter_names
        hist_names = {h["name"] for h in snap["histograms"]}
        assert "agent_latency_seconds" in hist_names

    def test_record_tool_invocation(self) -> None:
        self.collector.record_tool_invocation(
            agent_id="agent-1",
            tool_name="web_search",
            success=True,
            latency_seconds=0.3,
        )
        snap = self.collector.snapshot()
        counter_names = {c["name"] for c in snap["counters"]}
        assert "agent_tool_invocations_total" in counter_names

    def test_record_tool_invocation_failure(self) -> None:
        self.collector.record_tool_invocation(
            agent_id="agent-1",
            tool_name="web_search",
            success=False,
            latency_seconds=0.1,
        )
        snap = self.collector.snapshot()
        tool_counter = next(
            (c for c in snap["counters"] if c["name"] == "agent_tool_invocations_total"),
            None,
        )
        assert tool_counter is not None
        assert tool_counter["labels"]["success"] == "false"

    def test_record_error(self) -> None:
        self.collector.record_error(
            agent_id="agent-1",
            error_type="RateLimitError",
            recoverable=True,
        )
        snap = self.collector.snapshot()
        assert any(c["name"] == "agent_errors_total" for c in snap["counters"])

    def test_record_memory_operation(self) -> None:
        self.collector.record_memory_operation(
            agent_id="agent-1",
            operation="read",
            backend="redis",
        )
        snap = self.collector.snapshot()
        assert any(c["name"] == "agent_memory_operations_total" for c in snap["counters"])

    def test_set_drift_z_score(self) -> None:
        self.collector.set_drift_z_score("agent-1", 2.5)
        snap = self.collector.snapshot()
        gauge = next(
            (g for g in snap["gauges"] if g["name"] == "agent_drift_z_score"),
            None,
        )
        assert gauge is not None
        assert gauge["value"] == 2.5

    def test_snapshot_structure(self) -> None:
        snap = self.collector.snapshot()
        assert "counters" in snap
        assert "gauges" in snap
        assert "histograms" in snap

    def test_reset_clears_all(self) -> None:
        self.collector.increment_counter("test")
        self.collector.set_gauge("test_g", 1.0)
        self.collector.observe_histogram("test_h", 0.5)
        self.collector.reset()
        snap = self.collector.snapshot()
        assert snap["counters"] == []
        assert snap["gauges"] == []
        assert snap["histograms"] == []

    def test_thread_safety(self) -> None:
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(50):
                    self.collector.increment_counter("concurrent", {"worker": str(worker_id)})
                    self.collector.observe_histogram("concurrent_hist", float(i) * 0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ── MetricDefinition ───────────────────────────────────────────────────────────


class TestMetricDefinitions:
    def test_all_definitions_not_empty(self) -> None:
        assert len(ALL_DEFINITIONS) > 0

    def test_metric_definition_immutable(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            AGENT_LLM_CALLS_TOTAL.name = "changed"  # type: ignore[misc]

    def test_standard_metric_definitions(self) -> None:
        definitions = {d.name: d for d in ALL_DEFINITIONS}
        assert "agent_llm_calls_total" in definitions
        assert "agent_tool_invocations_total" in definitions
        assert "agent_cost_usd_total" in definitions
        assert "agent_latency_seconds" in definitions
        assert "agent_errors_total" in definitions
        assert "agent_active_count" in definitions
        assert "agent_tokens_total" in definitions
        assert "agent_memory_operations_total" in definitions
        assert "agent_drift_z_score" in definitions

    def test_metric_types(self) -> None:
        assert AGENT_LLM_CALLS_TOTAL.metric_type == "counter"
        assert AGENT_LATENCY_SECONDS.metric_type == "histogram"
        assert AGENT_ACTIVE_COUNT.metric_type == "gauge"
        assert AGENT_DRIFT_Z_SCORE.metric_type == "gauge"

    def test_label_names_present(self) -> None:
        assert "agent_id" in AGENT_LLM_CALLS_TOTAL.label_names
        assert "model" in AGENT_LLM_CALLS_TOTAL.label_names

    def test_unit_field(self) -> None:
        assert AGENT_COST_USD_TOTAL.unit == "USD"
        assert AGENT_LATENCY_SECONDS.unit == "seconds"
