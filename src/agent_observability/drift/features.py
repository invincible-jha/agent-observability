"""FeatureExtractor — extract behavioral features from traces for drift comparison.

Features are simple numeric/categorical summaries of a window of span records
that can be compared statistically against a baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from agent_observability.spans.conventions import (
    AGENT_SPAN_KIND,
    LLM_COST_USD,
    LLM_TOKENS_INPUT,
    LLM_TOKENS_OUTPUT,
    TOOL_NAME,
    TOOL_SUCCESS,
)


@dataclass
class SpanRecord:
    """Minimal representation of a span used for feature extraction.

    This deliberately avoids depending on OTel types so that features can be
    computed from any serialised span format (JSONL, DB rows, etc.).
    """

    span_kind: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_name: str = ""
    tool_success: bool = True
    error: bool = False
    attributes: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "SpanRecord":
        """Build a SpanRecord from a flat attribute dict (e.g. from JSONL)."""
        attrs = d.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        end_ns = d.get("end_time_ns", 0) or 0
        start_ns = d.get("start_time_ns", 0) or 0
        duration_ms = (float(end_ns) - float(start_ns)) / 1_000_000 if end_ns and start_ns else 0.0

        return cls(
            span_kind=str(attrs.get(AGENT_SPAN_KIND, d.get("span_kind", ""))),
            duration_ms=duration_ms,
            input_tokens=int(attrs.get(LLM_TOKENS_INPUT, 0) or 0),
            output_tokens=int(attrs.get(LLM_TOKENS_OUTPUT, 0) or 0),
            cost_usd=float(attrs.get(LLM_COST_USD, 0.0) or 0.0),
            tool_name=str(attrs.get(TOOL_NAME, "")),
            tool_success=bool(attrs.get(TOOL_SUCCESS, True)),
            error=str(d.get("status", "")).upper() == "ERROR",
            attributes=attrs,  # type: ignore[arg-type]
        )


@dataclass
class BehavioralFeatures:
    """Numeric feature vector summarising agent behaviour over a window of spans."""

    span_count: int = 0
    # Latency
    mean_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    # Token usage
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    # Cost
    mean_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    # Error / success rates
    error_rate: float = 0.0
    tool_failure_rate: float = 0.0
    # Span kind distribution (fraction of spans that are each kind)
    kind_fractions: dict[str, float] = field(default_factory=dict)
    # Tool call distribution (fraction of tool_invoke spans per tool)
    tool_fractions: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> dict[str, float]:
        """Return a flat numeric dict suitable for statistical comparison."""
        vec: dict[str, float] = {
            "span_count": float(self.span_count),
            "mean_duration_ms": self.mean_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "mean_input_tokens": self.mean_input_tokens,
            "mean_output_tokens": self.mean_output_tokens,
            "mean_cost_usd": self.mean_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "error_rate": self.error_rate,
            "tool_failure_rate": self.tool_failure_rate,
        }
        for kind, frac in self.kind_fractions.items():
            vec[f"kind.{kind}"] = frac
        for tool, frac in self.tool_fractions.items():
            vec[f"tool.{tool}"] = frac
        return vec


class FeatureExtractor:
    """Extract :class:`BehavioralFeatures` from a list of :class:`SpanRecord`."""

    def extract(self, spans: Sequence[SpanRecord]) -> BehavioralFeatures:
        """Compute features from *spans*.

        Parameters
        ----------
        spans:
            A non-empty sequence of SpanRecords representing a behavioural window.

        Returns
        -------
        A :class:`BehavioralFeatures` dataclass.  If *spans* is empty, returns
        an all-zero feature object.
        """
        if not spans:
            return BehavioralFeatures()

        n = len(spans)
        durations = sorted(s.duration_ms for s in spans)
        input_tokens = [s.input_tokens for s in spans]
        output_tokens = [s.output_tokens for s in spans]
        costs = [s.cost_usd for s in spans]
        errors = sum(1 for s in spans if s.error)
        tool_spans = [s for s in spans if s.span_kind == "tool_invoke"]
        tool_failures = sum(1 for s in tool_spans if not s.tool_success)

        # Kind distribution
        kind_counts: dict[str, int] = {}
        for s in spans:
            kind_counts[s.span_kind] = kind_counts.get(s.span_kind, 0) + 1
        kind_fractions = {k: v / n for k, v in kind_counts.items()}

        # Tool distribution
        tool_counts: dict[str, int] = {}
        for s in tool_spans:
            if s.tool_name:
                tool_counts[s.tool_name] = tool_counts.get(s.tool_name, 0) + 1
        n_tool = len(tool_spans) or 1
        tool_fractions = {t: c / n_tool for t, c in tool_counts.items()}

        p95_index = max(0, math.ceil(0.95 * n) - 1)

        return BehavioralFeatures(
            span_count=n,
            mean_duration_ms=_mean(durations),
            p95_duration_ms=durations[p95_index],
            mean_input_tokens=_mean(input_tokens),
            mean_output_tokens=_mean(output_tokens),
            mean_cost_usd=_mean(costs),
            total_cost_usd=sum(costs),
            error_rate=errors / n,
            tool_failure_rate=tool_failures / n_tool,
            kind_fractions=kind_fractions,
            tool_fractions=tool_fractions,
        )


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
