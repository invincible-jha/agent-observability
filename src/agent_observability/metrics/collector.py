"""AgentMetricCollector — collect and store agent metrics in memory.

Provides a thread-safe, dependency-free metrics accumulator.  Can feed
:class:`~agent_observability.metrics.exporter.PrometheusExporter`.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class CounterValue:
    """Monotonically increasing counter with labels."""

    name: str
    labels: dict[str, str]
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class GaugeValue:
    """Gauge (can go up or down) with labels."""

    name: str
    labels: dict[str, str]
    value: float = 0.0

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class HistogramValue:
    """Simplified histogram with pre-defined buckets."""

    name: str
    labels: dict[str, str]
    # Bucket upper bounds
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    _counts: list[int] = field(default_factory=list)
    _sum: float = 0.0
    _total_count: int = 0

    def __post_init__(self) -> None:
        self._counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self._sum += value
        self._total_count += 1
        for i, upper_bound in enumerate(self.buckets):
            if value <= upper_bound:
                self._counts[i] += 1

    @property
    def count(self) -> int:
        return self._total_count

    @property
    def sum_value(self) -> float:
        return self._sum

    def bucket_counts(self) -> list[tuple[float, int]]:
        """Return list of ``(upper_bound, cumulative_count)`` pairs."""
        cumulative = 0
        result: list[tuple[float, int]] = []
        for upper_bound, count in zip(self.buckets, self._counts):
            cumulative += count
            result.append((upper_bound, cumulative))
        return result


_LabelKey = tuple[str, ...]


class AgentMetricCollector:
    """Thread-safe in-memory metric store.

    All metric types are stored as dicts keyed by ``(metric_name, label_tuple)``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, _LabelKey], CounterValue] = {}
        self._gauges: dict[tuple[str, _LabelKey], GaugeValue] = {}
        self._histograms: dict[tuple[str, _LabelKey], HistogramValue] = {}

    # ── Counters ──────────────────────────────────────────────────────────────

    def increment_counter(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
        amount: float = 1.0,
    ) -> None:
        """Increment a named counter."""
        label_dict = labels or {}
        key = (name, tuple(sorted(label_dict.items())))
        with self._lock:
            if key not in self._counters:
                self._counters[key] = CounterValue(name=name, labels=label_dict)
            self._counters[key].inc(amount)

    # ── Gauges ────────────────────────────────────────────────────────────────

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Set a gauge to an absolute value."""
        label_dict = labels or {}
        key = (name, tuple(sorted(label_dict.items())))
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = GaugeValue(name=name, labels=label_dict)
            self._gauges[key].set(value)

    def increment_gauge(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
        amount: float = 1.0,
    ) -> None:
        label_dict = labels or {}
        key = (name, tuple(sorted(label_dict.items())))
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = GaugeValue(name=name, labels=label_dict)
            self._gauges[key].inc(amount)

    def decrement_gauge(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
        amount: float = 1.0,
    ) -> None:
        label_dict = labels or {}
        key = (name, tuple(sorted(label_dict.items())))
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = GaugeValue(name=name, labels=label_dict)
            self._gauges[key].dec(amount)

    # ── Histograms ────────────────────────────────────────────────────────────

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        label_dict = labels or {}
        key = (name, tuple(sorted(label_dict.items())))
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = HistogramValue(name=name, labels=label_dict)
            self._histograms[key].observe(value)

    # ── Domain helpers ────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        agent_id: str,
        model: str,
        provider: str,
        status: str,
        latency_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record all metrics associated with a single LLM call."""
        labels = {"agent_id": agent_id, "model": model, "provider": provider, "status": status}
        self.increment_counter("agent_llm_calls_total", labels)
        self.observe_histogram(
            "agent_latency_seconds",
            latency_seconds,
            {"agent_id": agent_id, "operation": "llm_call"},
        )
        self.increment_counter(
            "agent_tokens_total",
            {"agent_id": agent_id, "model": model, "token_type": "input"},
            amount=float(input_tokens),
        )
        self.increment_counter(
            "agent_tokens_total",
            {"agent_id": agent_id, "model": model, "token_type": "output"},
            amount=float(output_tokens),
        )
        cost_labels = {"agent_id": agent_id, "model": model, "provider": provider}
        self.increment_counter("agent_cost_usd_total", cost_labels, amount=cost_usd)

    def record_tool_invocation(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        latency_seconds: float,
    ) -> None:
        """Record metrics for a tool invocation."""
        labels = {"agent_id": agent_id, "tool_name": tool_name, "success": str(success).lower()}
        self.increment_counter("agent_tool_invocations_total", labels)
        self.observe_histogram(
            "agent_latency_seconds",
            latency_seconds,
            {"agent_id": agent_id, "operation": "tool_invoke"},
        )

    def record_error(
        self,
        agent_id: str,
        error_type: str,
        recoverable: bool,
    ) -> None:
        """Record an agent error."""
        labels = {
            "agent_id": agent_id,
            "error_type": error_type,
            "recoverable": str(recoverable).lower(),
        }
        self.increment_counter("agent_errors_total", labels)

    def record_memory_operation(
        self,
        agent_id: str,
        operation: str,
        backend: str,
    ) -> None:
        """Record a memory read or write operation."""
        self.increment_counter(
            "agent_memory_operations_total",
            {"agent_id": agent_id, "operation": operation, "backend": backend},
        )

    def set_drift_z_score(self, agent_id: str, z_score: float) -> None:
        """Update the drift Z-score gauge for an agent."""
        self.set_gauge("agent_drift_z_score", z_score, {"agent_id": agent_id})

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, object]:
        """Return a serialisable snapshot of all current metric values."""
        with self._lock:
            return {
                "counters": [
                    {"name": v.name, "labels": v.labels, "value": v.value}
                    for v in self._counters.values()
                ],
                "gauges": [
                    {"name": v.name, "labels": v.labels, "value": v.value}
                    for v in self._gauges.values()
                ],
                "histograms": [
                    {
                        "name": v.name,
                        "labels": v.labels,
                        "count": v.count,
                        "sum": v.sum_value,
                        "buckets": v.bucket_counts(),
                    }
                    for v in self._histograms.values()
                ],
            }

    def reset(self) -> None:
        """Clear all metrics (primarily for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
