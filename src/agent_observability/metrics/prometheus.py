"""PrometheusMetrics — generate Prometheus text exposition format.

This module provides a pure-Python Prometheus metrics collector with no
dependency on ``prometheus_client``.  Metrics are stored in in-process dicts
and serialised on demand via :meth:`PrometheusMetrics.export`.

Metrics exposed
---------------
Counters
    * ``agent_llm_calls_total`` — labelled by ``model``
    * ``agent_tool_invocations_total`` — labelled by ``tool_name``
    * ``agent_errors_total`` — labelled by ``error_type``

Histograms
    * ``agent_llm_latency_seconds`` — labelled by ``model``
    * ``agent_tool_latency_seconds`` — labelled by ``tool_name``

Gauges
    * ``agent_active_spans`` (set directly)
    * ``agent_total_cost_usd`` (cumulative)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# Default histogram buckets (seconds)
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
)


@dataclass
class _HistogramData:
    """Accumulator for a single histogram label set."""

    buckets: dict[float, int] = field(default_factory=dict)
    count: int = 0
    total: float = 0.0

    def observe(self, value: float, bucket_bounds: tuple[float, ...]) -> None:
        self.count += 1
        self.total += value
        for bound in bucket_bounds:
            if value <= bound:
                self.buckets[bound] = self.buckets.get(bound, 0) + 1
        # +Inf bucket
        self.buckets[float("inf")] = self.buckets.get(float("inf"), 0) + 1


class PrometheusMetrics:
    """Lightweight Prometheus metrics registry with text exposition support.

    No external dependencies — generates the text format directly.

    Example
    -------
    >>> metrics = PrometheusMetrics()
    >>> metrics.record_llm_call("gpt-4o", latency_seconds=1.2, tokens=500, cost=0.005)
    >>> print(metrics.export())
    """

    def __init__(
        self,
        namespace: str = "agent",
        latency_buckets: tuple[float, ...] = _LATENCY_BUCKETS,
    ) -> None:
        self._namespace = namespace
        self._buckets = latency_buckets
        self._lock = threading.Lock()

        # Counters: label_value -> count
        self._llm_calls_total: dict[str, int] = {}
        self._tool_invocations_total: dict[str, int] = {}
        self._errors_total: dict[str, int] = {}

        # Histograms: label_value -> _HistogramData
        self._llm_latency: dict[str, _HistogramData] = {}
        self._tool_latency: dict[str, _HistogramData] = {}

        # Gauges (scalar)
        self._active_spans: float = 0.0
        self._total_cost_usd: float = 0.0

    # ── Record helpers ────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str,
        latency_seconds: float,
        tokens: int,
        cost: float,
    ) -> None:
        """Record a single LLM call.

        Parameters
        ----------
        model:
            Model identifier used as a Prometheus label value.
        latency_seconds:
            Wall-clock latency for the call.
        tokens:
            Total token count (prompt + completion).
        cost:
            USD cost for the call.
        """
        with self._lock:
            self._llm_calls_total[model] = self._llm_calls_total.get(model, 0) + 1
            self._total_cost_usd += cost

            hist = self._llm_latency.setdefault(model, _HistogramData())
            hist.observe(latency_seconds, self._buckets)

    def record_tool_call(
        self,
        tool_name: str,
        latency_seconds: float,
        success: bool,
    ) -> None:
        """Record a single tool invocation.

        Parameters
        ----------
        tool_name:
            Tool name used as a Prometheus label value.
        latency_seconds:
            Wall-clock latency for the call.
        success:
            Whether the tool call succeeded.
        """
        with self._lock:
            self._tool_invocations_total[tool_name] = (
                self._tool_invocations_total.get(tool_name, 0) + 1
            )
            hist = self._tool_latency.setdefault(tool_name, _HistogramData())
            hist.observe(latency_seconds, self._buckets)

            if not success:
                error_label = f"tool_error:{tool_name}"
                self._errors_total[error_label] = self._errors_total.get(error_label, 0) + 1

    def record_error(self, error_type: str) -> None:
        """Increment the error counter for *error_type*.

        Parameters
        ----------
        error_type:
            Error classifier label (e.g. ``"RateLimitError"``).
        """
        with self._lock:
            self._errors_total[error_type] = self._errors_total.get(error_type, 0) + 1

    def set_active_spans(self, count: float) -> None:
        """Set the active spans gauge to *count*."""
        with self._lock:
            self._active_spans = count

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self) -> str:
        """Serialise all metrics to Prometheus text exposition format.

        Returns
        -------
        A UTF-8 string in the Prometheus text format, suitable for exposure
        via an HTTP ``/metrics`` endpoint.
        """
        with self._lock:
            lines: list[str] = []
            ns = self._namespace

            # ── Counters ──────────────────────────────────────────────────────
            lines.extend(self._counter_block(
                name=f"{ns}_llm_calls_total",
                help_text="Total number of LLM calls",
                label_name="model",
                data=dict(self._llm_calls_total),
            ))

            lines.extend(self._counter_block(
                name=f"{ns}_tool_invocations_total",
                help_text="Total number of tool invocations",
                label_name="tool_name",
                data=dict(self._tool_invocations_total),
            ))

            lines.extend(self._counter_block(
                name=f"{ns}_errors_total",
                help_text="Total number of agent errors",
                label_name="error_type",
                data=dict(self._errors_total),
            ))

            # ── Histograms ────────────────────────────────────────────────────
            lines.extend(self._histogram_block(
                name=f"{ns}_llm_latency_seconds",
                help_text="LLM call latency in seconds",
                label_name="model",
                data=dict(self._llm_latency),
            ))

            lines.extend(self._histogram_block(
                name=f"{ns}_tool_latency_seconds",
                help_text="Tool invocation latency in seconds",
                label_name="tool_name",
                data=dict(self._tool_latency),
            ))

            # ── Gauges ────────────────────────────────────────────────────────
            lines.extend([
                f"# HELP {ns}_active_spans Current number of active spans",
                f"# TYPE {ns}_active_spans gauge",
                f"{ns}_active_spans {self._active_spans}",
                "",
            ])

            lines.extend([
                f"# HELP {ns}_total_cost_usd Cumulative LLM cost in USD",
                f"# TYPE {ns}_total_cost_usd gauge",
                f"{ns}_total_cost_usd {self._total_cost_usd:.8f}",
                "",
            ])

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _counter_block(
        name: str,
        help_text: str,
        label_name: str,
        data: dict[str, int],
    ) -> list[str]:
        """Render a counter metric family in Prometheus text format."""
        if not data:
            return []
        lines: list[str] = [
            f"# HELP {name} {help_text}",
            f"# TYPE {name} counter",
        ]
        for label_value, count in sorted(data.items()):
            safe_label = label_value.replace('"', '\\"')
            lines.append(f'{name}{{{label_name}="{safe_label}"}} {count}')
        lines.append("")
        return lines

    def _histogram_block(
        self,
        name: str,
        help_text: str,
        label_name: str,
        data: dict[str, _HistogramData],
    ) -> list[str]:
        """Render a histogram metric family in Prometheus text format."""
        if not data:
            return []
        lines: list[str] = [
            f"# HELP {name} {help_text}",
            f"# TYPE {name} histogram",
        ]
        for label_value, hist in sorted(data.items()):
            safe_label = label_value.replace('"', '\\"')
            label_str = f'{label_name}="{safe_label}"'

            for bound in sorted(b for b in hist.buckets.keys() if b != float("inf")):
                bucket_count = hist.buckets.get(bound, 0)
                lines.append(f'{name}_bucket{{{label_str},le="{bound}"}} {bucket_count}')
            inf_count = hist.buckets.get(float("inf"), 0)
            lines.append(f'{name}_bucket{{{label_str},le="+Inf"}} {inf_count}')
            lines.append(f'{name}_sum{{{label_str}}} {hist.total:.6f}')
            lines.append(f'{name}_count{{{label_str}}} {hist.count}')

        lines.append("")
        return lines

    def reset(self) -> None:
        """Clear all accumulated metric data (useful in tests)."""
        with self._lock:
            self._llm_calls_total.clear()
            self._tool_invocations_total.clear()
            self._errors_total.clear()
            self._llm_latency.clear()
            self._tool_latency.clear()
            self._active_spans = 0.0
            self._total_cost_usd = 0.0
