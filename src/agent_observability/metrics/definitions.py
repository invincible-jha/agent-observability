"""Standard agent metric definitions.

These are the canonical metric names, types, and descriptions used by
:class:`~agent_observability.metrics.collector.AgentMetricCollector`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricType = Literal["counter", "histogram", "gauge"]


@dataclass(frozen=True)
class MetricDefinition:
    """Metadata for a single Prometheus metric."""

    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    label_names: tuple[str, ...] = ()


# ── Standard agent metrics ─────────────────────────────────────────────────────

AGENT_LLM_CALLS_TOTAL = MetricDefinition(
    name="agent_llm_calls_total",
    metric_type="counter",
    description="Total number of LLM API calls made by agents",
    label_names=("agent_id", "model", "provider", "status"),
)

AGENT_TOOL_INVOCATIONS_TOTAL = MetricDefinition(
    name="agent_tool_invocations_total",
    metric_type="counter",
    description="Total number of tool invocations by agents",
    label_names=("agent_id", "tool_name", "success"),
)

AGENT_COST_USD_TOTAL = MetricDefinition(
    name="agent_cost_usd_total",
    metric_type="counter",
    description="Total accumulated cost in USD across all LLM calls",
    unit="USD",
    label_names=("agent_id", "model", "provider"),
)

AGENT_LATENCY_SECONDS = MetricDefinition(
    name="agent_latency_seconds",
    metric_type="histogram",
    description="Latency of agent operations in seconds",
    unit="seconds",
    label_names=("agent_id", "operation"),
)

AGENT_ERRORS_TOTAL = MetricDefinition(
    name="agent_errors_total",
    metric_type="counter",
    description="Total number of agent errors",
    label_names=("agent_id", "error_type", "recoverable"),
)

AGENT_ACTIVE_COUNT = MetricDefinition(
    name="agent_active_count",
    metric_type="gauge",
    description="Number of currently active agent instances",
    label_names=("framework",),
)

AGENT_TOKENS_TOTAL = MetricDefinition(
    name="agent_tokens_total",
    metric_type="counter",
    description="Total tokens consumed across all LLM calls",
    label_names=("agent_id", "model", "token_type"),
)

AGENT_MEMORY_OPERATIONS_TOTAL = MetricDefinition(
    name="agent_memory_operations_total",
    metric_type="counter",
    description="Total number of memory read/write operations",
    label_names=("agent_id", "operation", "backend"),
)

AGENT_DRIFT_Z_SCORE = MetricDefinition(
    name="agent_drift_z_score",
    metric_type="gauge",
    description="Current maximum drift Z-score compared to baseline",
    label_names=("agent_id",),
)

# Canonical list for auto-registration
ALL_DEFINITIONS: list[MetricDefinition] = [
    AGENT_LLM_CALLS_TOTAL,
    AGENT_TOOL_INVOCATIONS_TOTAL,
    AGENT_COST_USD_TOTAL,
    AGENT_LATENCY_SECONDS,
    AGENT_ERRORS_TOTAL,
    AGENT_ACTIVE_COUNT,
    AGENT_TOKENS_TOTAL,
    AGENT_MEMORY_OPERATIONS_TOTAL,
    AGENT_DRIFT_Z_SCORE,
]
