"""DecisionTracker — accumulate and query DecisionSpans.

Maintains an in-memory log of agent decision spans and provides
querying by agent, time range, decision point, and confidence.

Example
-------
::

    from agent_observability.decisions.decision_tracker import DecisionTracker

    tracker = DecisionTracker()
    tracker.record(span)
    spans = tracker.query(agent_id="agent-1")
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agent_observability.decisions.decision_span import DecisionSpan, DecisionStatus


@dataclass
class DecisionQuery:
    """Parameters for querying the decision tracker.

    Attributes
    ----------
    agent_id:
        Filter by agent ID. None means all agents.
    decision_point:
        Filter by decision point label. None means all points.
    trace_id:
        Filter by trace ID. None means all traces.
    min_confidence:
        Only return spans with confidence >= this value.
    max_confidence:
        Only return spans with confidence <= this value.
    status:
        Filter by decision status. None means all statuses.
    since:
        Only return spans with timestamp_utc >= this datetime.
    until:
        Only return spans with timestamp_utc <= this datetime.
    limit:
        Maximum number of results to return. None means unlimited.
    """

    agent_id: Optional[str] = None
    decision_point: Optional[str] = None
    trace_id: Optional[str] = None
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    status: Optional[DecisionStatus] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: Optional[int] = None


@dataclass
class DecisionQueryResult:
    """Result of a DecisionTracker query.

    Attributes
    ----------
    spans:
        Matching decision spans, sorted by timestamp descending.
    total_found:
        Number of spans matched before applying limit.
    query:
        The query that produced this result.
    """

    spans: list[DecisionSpan]
    total_found: int
    query: DecisionQuery

    @property
    def average_confidence(self) -> float:
        """Average confidence across all returned spans."""
        if not self.spans:
            return 0.0
        return sum(s.confidence for s in self.spans) / len(self.spans)

    @property
    def low_confidence_count(self) -> int:
        """Number of spans with confidence < 0.5."""
        return sum(1 for s in self.spans if s.is_low_confidence)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "total_found": self.total_found,
            "returned": len(self.spans),
            "average_confidence": self.average_confidence,
            "low_confidence_count": self.low_confidence_count,
            "spans": [s.to_dict() for s in self.spans],
        }


class DecisionTracker:
    """Thread-safe in-memory accumulator for agent decision spans.

    Parameters
    ----------
    max_capacity:
        Maximum number of spans to retain. When capacity is exceeded,
        the oldest spans are evicted (FIFO). None means unlimited.

    Example
    -------
    ::

        tracker = DecisionTracker(max_capacity=10000)
        tracker.record(span)
        result = tracker.query(DecisionQuery(agent_id="agent-1"))
        print(result.average_confidence)
    """

    def __init__(self, max_capacity: Optional[int] = None) -> None:
        self._spans: list[DecisionSpan] = []
        self._lock = threading.Lock()
        self._max_capacity = max_capacity

    def record(self, span: DecisionSpan) -> None:
        """Add a decision span to the tracker.

        Parameters
        ----------
        span:
            The DecisionSpan to record.
        """
        with self._lock:
            self._spans.append(span)
            if self._max_capacity is not None and len(self._spans) > self._max_capacity:
                # Evict oldest spans (front of list)
                overflow = len(self._spans) - self._max_capacity
                del self._spans[:overflow]

    def query(self, query_params: Optional[DecisionQuery] = None) -> DecisionQueryResult:
        """Query recorded decision spans.

        Parameters
        ----------
        query_params:
            Filter parameters. If None, all spans are returned.

        Returns
        -------
        DecisionQueryResult
            Matching spans sorted by timestamp descending.
        """
        if query_params is None:
            query_params = DecisionQuery()

        with self._lock:
            candidates = list(self._spans)

        filtered: list[DecisionSpan] = []
        for span in candidates:
            if query_params.agent_id is not None and span.agent_id != query_params.agent_id:
                continue
            if query_params.decision_point is not None and span.decision_point != query_params.decision_point:
                continue
            if query_params.trace_id is not None and span.trace_id != query_params.trace_id:
                continue
            if not (query_params.min_confidence <= span.confidence <= query_params.max_confidence):
                continue
            if query_params.status is not None and span.status != query_params.status:
                continue
            if query_params.since is not None:
                since = query_params.since
                if since.tzinfo is None:
                    since = since.replace(tzinfo=timezone.utc)
                if span.timestamp_utc < since:
                    continue
            if query_params.until is not None:
                until = query_params.until
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if span.timestamp_utc > until:
                    continue
            filtered.append(span)

        # Sort by timestamp descending
        filtered.sort(key=lambda s: s.timestamp_utc, reverse=True)
        total_found = len(filtered)

        if query_params.limit is not None:
            filtered = filtered[: query_params.limit]

        return DecisionQueryResult(
            spans=filtered,
            total_found=total_found,
            query=query_params,
        )

    def get_by_agent(self, agent_id: str) -> list[DecisionSpan]:
        """Return all spans for a given agent ID, sorted by timestamp descending."""
        result = self.query(DecisionQuery(agent_id=agent_id))
        return result.spans

    def get_by_trace(self, trace_id: str) -> list[DecisionSpan]:
        """Return all spans for a given trace ID."""
        result = self.query(DecisionQuery(trace_id=trace_id))
        return result.spans

    def get_low_confidence_spans(self, threshold: float = 0.5) -> list[DecisionSpan]:
        """Return spans where confidence is below the given threshold."""
        result = self.query(DecisionQuery(max_confidence=threshold - 0.0001))
        return result.spans

    def clear(self) -> None:
        """Remove all recorded spans from the tracker."""
        with self._lock:
            self._spans.clear()

    @property
    def span_count(self) -> int:
        """Total number of spans currently stored."""
        with self._lock:
            return len(self._spans)

    def summary(self) -> dict[str, object]:
        """Return a summary of all recorded decisions.

        Returns a dict with total count, agents seen, and average confidence.
        """
        with self._lock:
            spans = list(self._spans)

        agents = {s.agent_id for s in spans}
        decision_points = {s.decision_point for s in spans}
        avg_confidence = sum(s.confidence for s in spans) / len(spans) if spans else 0.0

        return {
            "total_spans": len(spans),
            "unique_agents": len(agents),
            "unique_decision_points": len(decision_points),
            "average_confidence": avg_confidence,
        }
