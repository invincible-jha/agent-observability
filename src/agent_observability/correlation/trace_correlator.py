"""TraceCorrelator — propagate and link spans across agent boundaries.

Allows parent agents to create child contexts for delegated tasks and then
links all spans into a tree structure for visualization and analysis.

Example
-------
::

    from agent_observability.correlation.trace_correlator import TraceCorrelator

    correlator = TraceCorrelator()
    root_ctx = correlator.start_root_span("orchestrator", "plan_task")
    child_ctx = correlator.start_child_span(root_ctx, "worker-1", "execute_step")
    correlator.end_span(child_ctx.span_id)
    correlator.end_span(root_ctx.span_id)
    tree = correlator.get_trace_tree(root_ctx.trace_id)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from agent_observability.correlation.correlation_context import CorrelationContext


class SpanRelationship(str, Enum):
    """Relationship type between spans."""

    CHILD = "child"         # Normal parent-child delegation
    FOLLOWS_FROM = "follows_from"  # Sequential dependency (not nested)


@dataclass
class CorrelatedSpan:
    """A span linked into a distributed trace across agent boundaries.

    Attributes
    ----------
    span_id:
        Unique span identifier.
    trace_id:
        Shared trace identifier.
    parent_span_id:
        Parent span ID, or None for the root span.
    agent_id:
        The agent that owns this span.
    operation_name:
        What operation this span represents.
    start_time_utc:
        When the span started.
    end_time_utc:
        When the span ended. None if still in progress.
    relationship:
        How this span relates to its parent.
    baggage:
        Propagated baggage from the context.
    metadata:
        Arbitrary additional data.
    """

    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    agent_id: str
    operation_name: str
    start_time_utc: datetime
    end_time_utc: Optional[datetime] = None
    relationship: SpanRelationship = SpanRelationship.CHILD
    baggage: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Span duration in milliseconds. None if the span is still running."""
        if self.end_time_utc is None:
            return None
        delta = self.end_time_utc - self.start_time_utc
        return delta.total_seconds() * 1000.0

    @property
    def is_root(self) -> bool:
        """True if this is a root span (no parent)."""
        return self.parent_span_id is None

    @property
    def is_completed(self) -> bool:
        """True if the span has an end time."""
        return self.end_time_utc is not None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "agent_id": self.agent_id,
            "operation_name": self.operation_name,
            "start_time_utc": self.start_time_utc.isoformat(),
            "end_time_utc": self.end_time_utc.isoformat() if self.end_time_utc else None,
            "duration_ms": self.duration_ms,
            "is_completed": self.is_completed,
            "relationship": self.relationship.value,
            "baggage": self.baggage,
        }


@dataclass
class TraceTree:
    """A hierarchical view of all spans in a single trace.

    Attributes
    ----------
    trace_id:
        The trace identifier.
    root_span:
        The root span of the trace.
    spans:
        All spans in the trace, indexed by span_id.
    """

    trace_id: str
    root_span: Optional[CorrelatedSpan]
    spans: dict[str, CorrelatedSpan] = field(default_factory=dict)

    @property
    def span_count(self) -> int:
        """Total number of spans in the trace."""
        return len(self.spans)

    @property
    def agent_ids(self) -> set[str]:
        """Set of all agent IDs that contributed spans to this trace."""
        return {span.agent_id for span in self.spans.values()}

    def get_children(self, parent_span_id: str) -> list[CorrelatedSpan]:
        """Return all direct child spans of the given span."""
        return [
            span for span in self.spans.values()
            if span.parent_span_id == parent_span_id
        ]

    def get_depth(self, span_id: str) -> int:
        """Return the depth of a span (root=0, first child=1, etc.)."""
        depth = 0
        current = self.spans.get(span_id)
        while current and current.parent_span_id is not None:
            parent = self.spans.get(current.parent_span_id)
            if parent is None:
                break
            current = parent
            depth += 1
        return depth

    def total_duration_ms(self) -> Optional[float]:
        """Total trace duration from root start to last end time."""
        if self.root_span is None or not self.root_span.is_completed:
            return None
        return self.root_span.duration_ms

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_count": self.span_count,
            "agent_ids": sorted(self.agent_ids),
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "spans": {sid: span.to_dict() for sid, span in self.spans.items()},
        }


class TraceCorrelator:
    """Propagate and link spans from multiple agents into a coherent trace.

    Creates CorrelationContext objects for new spans and stores CorrelatedSpan
    records that can be linked into a TraceTree.

    Parameters
    ----------
    default_sampled:
        Default sampling decision for new root spans.

    Example
    -------
    ::

        correlator = TraceCorrelator()
        root_ctx = correlator.start_root_span("orchestrator", "plan_task")
        child_ctx = correlator.start_child_span(root_ctx, "worker", "step_1")
        correlator.end_span(child_ctx.span_id)
        correlator.end_span(root_ctx.span_id)
        tree = correlator.get_trace_tree(root_ctx.trace_id)
        print(tree.span_count)
    """

    def __init__(self, *, default_sampled: bool = True) -> None:
        self._default_sampled = default_sampled
        self._spans: dict[str, CorrelatedSpan] = {}
        self._lock = threading.Lock()

    def start_root_span(
        self,
        agent_id: str,
        operation_name: str,
        *,
        metadata: Optional[dict[str, object]] = None,
    ) -> CorrelationContext:
        """Create a new root span and return its context.

        Parameters
        ----------
        agent_id:
            The agent starting this trace.
        operation_name:
            Name of the operation this span represents.
        metadata:
            Optional additional metadata.

        Returns
        -------
        CorrelationContext
            Context for the new root span.
        """
        ctx = CorrelationContext.new_root(sampled=self._default_sampled)
        span = CorrelatedSpan(
            span_id=ctx.span_id,
            trace_id=ctx.trace_id,
            parent_span_id=None,
            agent_id=agent_id,
            operation_name=operation_name,
            start_time_utc=datetime.now(timezone.utc),
            baggage={item.key: item.value for item in ctx.baggage},
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._spans[span.span_id] = span
        return ctx

    def start_child_span(
        self,
        parent_ctx: CorrelationContext,
        agent_id: str,
        operation_name: str,
        *,
        relationship: SpanRelationship = SpanRelationship.CHILD,
        metadata: Optional[dict[str, object]] = None,
    ) -> CorrelationContext:
        """Create a child span inheriting context from a parent.

        Parameters
        ----------
        parent_ctx:
            The parent span's CorrelationContext.
        agent_id:
            The agent creating the child span.
        operation_name:
            Name of the child operation.
        relationship:
            How this span relates to the parent.
        metadata:
            Optional additional metadata.

        Returns
        -------
        CorrelationContext
            Context for the new child span.
        """
        child_ctx = parent_ctx.new_child_span()
        span = CorrelatedSpan(
            span_id=child_ctx.span_id,
            trace_id=child_ctx.trace_id,
            parent_span_id=child_ctx.parent_span_id,
            agent_id=agent_id,
            operation_name=operation_name,
            start_time_utc=datetime.now(timezone.utc),
            relationship=relationship,
            baggage={item.key: item.value for item in child_ctx.baggage},
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._spans[span.span_id] = span
        return child_ctx

    def end_span(
        self,
        span_id: str,
        *,
        metadata_update: Optional[dict[str, object]] = None,
    ) -> bool:
        """Mark a span as completed.

        Parameters
        ----------
        span_id:
            The ID of the span to end.
        metadata_update:
            Optional metadata to merge into the span.

        Returns
        -------
        bool
            True if the span was found and ended, False if not found.
        """
        with self._lock:
            span = self._spans.get(span_id)
            if span is None:
                return False
            span.end_time_utc = datetime.now(timezone.utc)
            if metadata_update:
                span.metadata.update(metadata_update)
        return True

    def get_span(self, span_id: str) -> Optional[CorrelatedSpan]:
        """Return a span by ID, or None if not found."""
        with self._lock:
            return self._spans.get(span_id)

    def get_trace_tree(self, trace_id: str) -> TraceTree:
        """Build a TraceTree for all spans in a given trace.

        Parameters
        ----------
        trace_id:
            The trace to build a tree for.

        Returns
        -------
        TraceTree
            Hierarchical view of all spans in the trace.
        """
        with self._lock:
            trace_spans = {
                sid: span
                for sid, span in self._spans.items()
                if span.trace_id == trace_id
            }

        root_span = next(
            (span for span in trace_spans.values() if span.is_root),
            None,
        )

        return TraceTree(
            trace_id=trace_id,
            root_span=root_span,
            spans=trace_spans,
        )

    def list_traces(self) -> list[str]:
        """Return all distinct trace IDs currently stored."""
        with self._lock:
            return list({span.trace_id for span in self._spans.values()})

    def clear(self) -> None:
        """Remove all stored spans."""
        with self._lock:
            self._spans.clear()
