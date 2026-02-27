"""Tests for DecisionTracker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_observability.decisions.decision_span import DecisionSpan, DecisionStatus
from agent_observability.decisions.decision_tracker import (
    DecisionQuery,
    DecisionQueryResult,
    DecisionTracker,
)


def _make_span(
    agent_id: str = "agent-1",
    decision_point: str = "select_tool",
    confidence: float = 0.8,
    trace_id: str | None = None,
) -> DecisionSpan:
    return DecisionSpan(
        agent_id=agent_id,
        decision_point=decision_point,
        chosen_option="option-a",
        confidence=confidence,
        trace_id=trace_id,
    )


class TestDecisionTracker:
    def setup_method(self) -> None:
        self.tracker = DecisionTracker()

    def test_initial_span_count_zero(self) -> None:
        assert self.tracker.span_count == 0

    def test_record_increments_count(self) -> None:
        self.tracker.record(_make_span())
        assert self.tracker.span_count == 1

    def test_record_multiple_spans(self) -> None:
        for _ in range(5):
            self.tracker.record(_make_span())
        assert self.tracker.span_count == 5

    def test_query_all_returns_all(self) -> None:
        for _ in range(3):
            self.tracker.record(_make_span())
        result = self.tracker.query()
        assert result.total_found == 3

    def test_query_by_agent_id(self) -> None:
        self.tracker.record(_make_span(agent_id="agent-1"))
        self.tracker.record(_make_span(agent_id="agent-2"))
        result = self.tracker.query(DecisionQuery(agent_id="agent-1"))
        assert result.total_found == 1
        assert result.spans[0].agent_id == "agent-1"

    def test_query_by_decision_point(self) -> None:
        self.tracker.record(_make_span(decision_point="select_tool"))
        self.tracker.record(_make_span(decision_point="route_query"))
        result = self.tracker.query(DecisionQuery(decision_point="select_tool"))
        assert result.total_found == 1

    def test_query_by_trace_id(self) -> None:
        self.tracker.record(_make_span(trace_id="trace-abc"))
        self.tracker.record(_make_span(trace_id="trace-xyz"))
        result = self.tracker.query(DecisionQuery(trace_id="trace-abc"))
        assert result.total_found == 1

    def test_query_by_min_confidence(self) -> None:
        self.tracker.record(_make_span(confidence=0.3))
        self.tracker.record(_make_span(confidence=0.9))
        result = self.tracker.query(DecisionQuery(min_confidence=0.7))
        assert result.total_found == 1
        assert result.spans[0].confidence == 0.9

    def test_query_by_status(self) -> None:
        span = _make_span()
        span.mark_overridden()
        self.tracker.record(span)
        self.tracker.record(_make_span())
        result = self.tracker.query(DecisionQuery(status=DecisionStatus.OVERRIDDEN))
        assert result.total_found == 1

    def test_query_with_limit(self) -> None:
        for _ in range(5):
            self.tracker.record(_make_span())
        result = self.tracker.query(DecisionQuery(limit=3))
        assert len(result.spans) == 3
        assert result.total_found == 5

    def test_results_sorted_by_timestamp_descending(self) -> None:
        for _ in range(3):
            self.tracker.record(_make_span())
        result = self.tracker.query()
        timestamps = [s.timestamp_utc for s in result.spans]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_result_average_confidence(self) -> None:
        self.tracker.record(_make_span(confidence=0.6))
        self.tracker.record(_make_span(confidence=0.8))
        result = self.tracker.query()
        assert result.average_confidence == pytest.approx(0.7)

    def test_query_result_low_confidence_count(self) -> None:
        self.tracker.record(_make_span(confidence=0.3))
        self.tracker.record(_make_span(confidence=0.4))
        self.tracker.record(_make_span(confidence=0.8))
        result = self.tracker.query()
        assert result.low_confidence_count == 2

    def test_get_by_agent(self) -> None:
        self.tracker.record(_make_span(agent_id="agent-1"))
        self.tracker.record(_make_span(agent_id="agent-2"))
        spans = self.tracker.get_by_agent("agent-1")
        assert len(spans) == 1
        assert spans[0].agent_id == "agent-1"

    def test_get_by_trace(self) -> None:
        self.tracker.record(_make_span(trace_id="trace-x"))
        spans = self.tracker.get_by_trace("trace-x")
        assert len(spans) == 1

    def test_get_low_confidence_spans(self) -> None:
        self.tracker.record(_make_span(confidence=0.2))
        self.tracker.record(_make_span(confidence=0.9))
        spans = self.tracker.get_low_confidence_spans(threshold=0.5)
        assert len(spans) == 1
        assert spans[0].confidence == 0.2

    def test_clear_removes_all_spans(self) -> None:
        self.tracker.record(_make_span())
        self.tracker.clear()
        assert self.tracker.span_count == 0

    def test_max_capacity_evicts_oldest(self) -> None:
        tracker = DecisionTracker(max_capacity=3)
        for i in range(5):
            span = _make_span()
            span.metadata["index"] = i
            tracker.record(span)
        assert tracker.span_count == 3

    def test_summary_returns_correct_fields(self) -> None:
        self.tracker.record(_make_span(agent_id="agent-1"))
        self.tracker.record(_make_span(agent_id="agent-2"))
        summary = self.tracker.summary()
        assert summary["total_spans"] == 2
        assert summary["unique_agents"] == 2
        assert "average_confidence" in summary

    def test_query_result_to_dict(self) -> None:
        self.tracker.record(_make_span())
        result = self.tracker.query()
        d = result.to_dict()
        assert "total_found" in d
        assert "returned" in d
        assert "spans" in d
