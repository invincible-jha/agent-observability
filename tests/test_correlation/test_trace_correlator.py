"""Tests for TraceCorrelator."""
from __future__ import annotations

import pytest

from agent_observability.correlation.correlation_context import CorrelationContext
from agent_observability.correlation.trace_correlator import (
    CorrelatedSpan,
    SpanRelationship,
    TraceCorrelator,
    TraceTree,
)


class TestCorrelatedSpan:
    def test_is_root_when_no_parent(self) -> None:
        from datetime import datetime, timezone
        span = CorrelatedSpan(
            span_id="s1", trace_id="t1", parent_span_id=None,
            agent_id="agent-1", operation_name="root_op",
            start_time_utc=datetime.now(timezone.utc),
        )
        assert span.is_root is True

    def test_not_root_when_has_parent(self) -> None:
        from datetime import datetime, timezone
        span = CorrelatedSpan(
            span_id="s2", trace_id="t1", parent_span_id="s1",
            agent_id="agent-1", operation_name="child_op",
            start_time_utc=datetime.now(timezone.utc),
        )
        assert span.is_root is False

    def test_duration_ms_none_when_not_completed(self) -> None:
        from datetime import datetime, timezone
        span = CorrelatedSpan(
            span_id="s1", trace_id="t1", parent_span_id=None,
            agent_id="agent-1", operation_name="op",
            start_time_utc=datetime.now(timezone.utc),
        )
        assert span.duration_ms is None

    def test_duration_ms_positive_when_completed(self) -> None:
        from datetime import datetime, timedelta, timezone
        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=100)
        span = CorrelatedSpan(
            span_id="s1", trace_id="t1", parent_span_id=None,
            agent_id="agent-1", operation_name="op",
            start_time_utc=start,
            end_time_utc=end,
        )
        assert span.duration_ms == pytest.approx(100.0, abs=1.0)

    def test_is_completed_false_when_no_end(self) -> None:
        from datetime import datetime, timezone
        span = CorrelatedSpan(
            span_id="s1", trace_id="t1", parent_span_id=None,
            agent_id="agent-1", operation_name="op",
            start_time_utc=datetime.now(timezone.utc),
        )
        assert span.is_completed is False

    def test_to_dict_structure(self) -> None:
        from datetime import datetime, timezone
        span = CorrelatedSpan(
            span_id="s1", trace_id="t1", parent_span_id=None,
            agent_id="agent-1", operation_name="op",
            start_time_utc=datetime.now(timezone.utc),
        )
        d = span.to_dict()
        assert d["span_id"] == "s1"
        assert d["agent_id"] == "agent-1"
        assert "duration_ms" in d
        assert "is_completed" in d


class TestTraceCorrelator:
    def setup_method(self) -> None:
        self.correlator = TraceCorrelator()

    def test_start_root_span_returns_context(self) -> None:
        ctx = self.correlator.start_root_span("agent-1", "plan_task")
        assert isinstance(ctx, CorrelationContext)
        assert ctx.is_root is True

    def test_root_span_stored(self) -> None:
        ctx = self.correlator.start_root_span("agent-1", "plan_task")
        span = self.correlator.get_span(ctx.span_id)
        assert span is not None
        assert span.agent_id == "agent-1"
        assert span.operation_name == "plan_task"

    def test_start_child_span_returns_child_context(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        child_ctx = self.correlator.start_child_span(root_ctx, "agent-2", "step_1")
        assert child_ctx.parent_span_id == root_ctx.span_id
        assert child_ctx.trace_id == root_ctx.trace_id

    def test_child_span_stored(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        child_ctx = self.correlator.start_child_span(root_ctx, "agent-2", "step_1")
        span = self.correlator.get_span(child_ctx.span_id)
        assert span is not None
        assert span.parent_span_id == root_ctx.span_id

    def test_end_span_marks_completed(self) -> None:
        ctx = self.correlator.start_root_span("agent-1", "op")
        result = self.correlator.end_span(ctx.span_id)
        assert result is True
        span = self.correlator.get_span(ctx.span_id)
        assert span is not None
        assert span.is_completed is True

    def test_end_span_unknown_returns_false(self) -> None:
        result = self.correlator.end_span("nonexistent-span")
        assert result is False

    def test_get_span_returns_none_for_unknown(self) -> None:
        span = self.correlator.get_span("unknown")
        assert span is None

    def test_get_trace_tree_returns_tree(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        child_ctx = self.correlator.start_child_span(root_ctx, "agent-2", "child")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        assert isinstance(tree, TraceTree)
        assert tree.span_count == 2

    def test_trace_tree_root_span_correct(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root_op")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        assert tree.root_span is not None
        assert tree.root_span.span_id == root_ctx.span_id

    def test_trace_tree_agent_ids(self) -> None:
        root_ctx = self.correlator.start_root_span("orchestrator", "root")
        self.correlator.start_child_span(root_ctx, "worker-1", "step")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        assert "orchestrator" in tree.agent_ids
        assert "worker-1" in tree.agent_ids

    def test_trace_tree_get_children(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        child1_ctx = self.correlator.start_child_span(root_ctx, "agent-2", "step-1")
        child2_ctx = self.correlator.start_child_span(root_ctx, "agent-3", "step-2")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        children = tree.get_children(root_ctx.span_id)
        child_span_ids = {c.span_id for c in children}
        assert child1_ctx.span_id in child_span_ids
        assert child2_ctx.span_id in child_span_ids

    def test_trace_tree_depth_root_is_zero(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        assert tree.get_depth(root_ctx.span_id) == 0

    def test_trace_tree_depth_child_is_one(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        child_ctx = self.correlator.start_child_span(root_ctx, "agent-2", "step")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        assert tree.get_depth(child_ctx.span_id) == 1

    def test_list_traces_returns_trace_ids(self) -> None:
        ctx1 = self.correlator.start_root_span("agent-1", "trace-1-op")
        ctx2 = self.correlator.start_root_span("agent-2", "trace-2-op")
        traces = self.correlator.list_traces()
        assert ctx1.trace_id in traces
        assert ctx2.trace_id in traces

    def test_clear_removes_all_spans(self) -> None:
        self.correlator.start_root_span("agent-1", "op")
        self.correlator.clear()
        assert len(self.correlator.list_traces()) == 0

    def test_relationship_type_stored(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "root")
        child_ctx = self.correlator.start_child_span(
            root_ctx, "agent-2", "step",
            relationship=SpanRelationship.FOLLOWS_FROM,
        )
        span = self.correlator.get_span(child_ctx.span_id)
        assert span is not None
        assert span.relationship == SpanRelationship.FOLLOWS_FROM

    def test_trace_tree_to_dict(self) -> None:
        root_ctx = self.correlator.start_root_span("agent-1", "op")
        tree = self.correlator.get_trace_tree(root_ctx.trace_id)
        d = tree.to_dict()
        assert "trace_id" in d
        assert "span_count" in d
        assert "agent_ids" in d
        assert "spans" in d
