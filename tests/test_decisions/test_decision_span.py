"""Tests for DecisionSpan."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_observability.decisions.decision_span import DecisionSpan, DecisionStatus


def _make_span(**kwargs) -> DecisionSpan:
    defaults = dict(
        agent_id="agent-1",
        decision_point="select_tool",
        chosen_option="web_search",
        alternatives_considered=["calculator", "file_read"],
        confidence=0.9,
        reasoning_summary="Best match for retrieval goal.",
    )
    defaults.update(kwargs)
    return DecisionSpan(**defaults)


class TestDecisionSpan:
    def test_span_id_auto_generated(self) -> None:
        span = _make_span()
        assert span.span_id != ""
        assert len(span.span_id) > 10

    def test_different_spans_have_different_ids(self) -> None:
        span1 = _make_span()
        span2 = _make_span()
        assert span1.span_id != span2.span_id

    def test_timestamp_is_utc(self) -> None:
        span = _make_span()
        assert span.timestamp_utc.tzinfo is not None

    def test_confidence_in_range(self) -> None:
        span = _make_span(confidence=0.75)
        assert span.confidence == 0.75

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_span(confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_span(confidence=1.1)

    def test_alternative_count(self) -> None:
        span = _make_span(alternatives_considered=["a", "b", "c"])
        assert span.alternative_count == 3

    def test_is_high_confidence_true(self) -> None:
        span = _make_span(confidence=0.9)
        assert span.is_high_confidence is True

    def test_is_high_confidence_false(self) -> None:
        span = _make_span(confidence=0.5)
        assert span.is_high_confidence is False

    def test_is_low_confidence_true(self) -> None:
        span = _make_span(confidence=0.3)
        assert span.is_low_confidence is True

    def test_is_low_confidence_false(self) -> None:
        span = _make_span(confidence=0.7)
        assert span.is_low_confidence is False

    def test_status_default_completed(self) -> None:
        span = _make_span()
        assert span.status == DecisionStatus.COMPLETED

    def test_mark_overridden(self) -> None:
        span = _make_span()
        span.mark_overridden(reason="Human operator changed decision.")
        assert span.status == DecisionStatus.OVERRIDDEN
        assert "override_reason" in span.metadata

    def test_mark_failed(self) -> None:
        span = _make_span()
        span.mark_failed(error="Tool unavailable.")
        assert span.status == DecisionStatus.FAILED
        assert "failure_reason" in span.metadata

    def test_to_dict_structure(self) -> None:
        span = _make_span()
        d = span.to_dict()
        assert d["agent_id"] == "agent-1"
        assert d["decision_point"] == "select_tool"
        assert d["chosen_option"] == "web_search"
        assert d["alternatives_considered"] == ["calculator", "file_read"]
        assert d["confidence"] == 0.9
        assert "span_id" in d
        assert "timestamp_utc" in d

    def test_from_dict_round_trip(self) -> None:
        span = _make_span(
            trace_id="trace-abc",
            parent_span_id="parent-123",
            duration_ms=15.5,
        )
        d = span.to_dict()
        reconstructed = DecisionSpan.from_dict(d)
        assert reconstructed.agent_id == span.agent_id
        assert reconstructed.decision_point == span.decision_point
        assert reconstructed.chosen_option == span.chosen_option
        assert reconstructed.confidence == span.confidence
        assert reconstructed.trace_id == span.trace_id
        assert reconstructed.parent_span_id == span.parent_span_id
        assert reconstructed.duration_ms == span.duration_ms

    def test_from_dict_handles_missing_optional_fields(self) -> None:
        minimal = {
            "agent_id": "agent-x",
            "decision_point": "route",
            "chosen_option": "plan_a",
        }
        span = DecisionSpan.from_dict(minimal)
        assert span.agent_id == "agent-x"
        assert span.trace_id is None

    def test_custom_span_id_preserved(self) -> None:
        span = _make_span(span_id="custom-span-id")
        assert span.span_id == "custom-span-id"

    def test_trace_id_and_parent_set(self) -> None:
        span = _make_span(trace_id="trace-1", parent_span_id="parent-1")
        assert span.trace_id == "trace-1"
        assert span.parent_span_id == "parent-1"
