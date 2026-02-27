"""DecisionSpan — capture a single agent decision point for observability.

Each DecisionSpan represents one meaningful decision made by an agent:
which option was chosen, what alternatives were considered, confidence
level, and a summary of the reasoning used.

Example
-------
::

    from agent_observability.decisions.decision_span import DecisionSpan

    span = DecisionSpan(
        agent_id="agent-1",
        decision_point="select_tool",
        chosen_option="web_search",
        alternatives_considered=["calculator", "file_read"],
        confidence=0.92,
        reasoning_summary="Web search best matches the information retrieval goal.",
    )
    print(span.span_id, span.timestamp_utc)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class DecisionStatus(str, Enum):
    """Status of a decision span."""

    COMPLETED = "completed"
    OVERRIDDEN = "overridden"  # Human or higher-level agent overrode the decision
    FAILED = "failed"          # Decision process encountered an error


@dataclass
class DecisionSpan:
    """Captures a single agent decision for tracing and analysis.

    Attributes
    ----------
    agent_id:
        Identifier of the agent that made this decision.
    decision_point:
        Label describing what type of decision was made
        (e.g. ``"select_tool"``, ``"route_query"``, ``"choose_response"``).
    chosen_option:
        The option the agent selected.
    alternatives_considered:
        Other options that were evaluated but not chosen.
    confidence:
        Confidence score for the chosen option (0.0–1.0).
    reasoning_summary:
        Short human-readable explanation of why this option was chosen.
    span_id:
        Unique ID for this decision span. Auto-generated if not provided.
    trace_id:
        Optional trace ID to correlate this decision with a broader trace.
    parent_span_id:
        Optional parent span ID for nested decision hierarchies.
    agent_session_id:
        Optional session identifier for grouping decisions by session.
    status:
        Current status of this decision span.
    timestamp_utc:
        UTC timestamp when this decision was made. Auto-set if not provided.
    duration_ms:
        How long (in ms) the decision process took. None if not measured.
    metadata:
        Arbitrary additional structured data.
    """

    agent_id: str
    decision_point: str
    chosen_option: str
    alternatives_considered: list[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning_summary: str = ""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    agent_session_id: Optional[str] = None
    status: DecisionStatus = DecisionStatus.COMPLETED
    timestamp_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: Optional[float] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence is in [0.0, 1.0]."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def alternative_count(self) -> int:
        """Number of alternatives considered."""
        return len(self.alternatives_considered)

    @property
    def is_high_confidence(self) -> bool:
        """True if confidence >= 0.8."""
        return self.confidence >= 0.8

    @property
    def is_low_confidence(self) -> bool:
        """True if confidence < 0.5."""
        return self.confidence < 0.5

    def mark_overridden(self, reason: str = "") -> None:
        """Mark this decision as overridden by a higher authority."""
        self.status = DecisionStatus.OVERRIDDEN
        if reason:
            self.metadata["override_reason"] = reason

    def mark_failed(self, error: str = "") -> None:
        """Mark this decision as failed."""
        self.status = DecisionStatus.FAILED
        if error:
            self.metadata["failure_reason"] = error

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "span_id": self.span_id,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "agent_session_id": self.agent_session_id,
            "decision_point": self.decision_point,
            "chosen_option": self.chosen_option,
            "alternatives_considered": self.alternatives_considered,
            "alternative_count": self.alternative_count,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "status": self.status.value,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DecisionSpan":
        """Reconstruct a DecisionSpan from a dictionary.

        Parameters
        ----------
        data:
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        DecisionSpan
            Reconstructed span.
        """
        timestamp_str = data.get("timestamp_utc", "")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        status_str = data.get("status", "completed")
        try:
            status = DecisionStatus(status_str)
        except ValueError:
            status = DecisionStatus.COMPLETED

        return cls(
            agent_id=str(data.get("agent_id", "")),
            decision_point=str(data.get("decision_point", "")),
            chosen_option=str(data.get("chosen_option", "")),
            alternatives_considered=list(data.get("alternatives_considered", [])),
            confidence=float(data.get("confidence", 1.0)),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            span_id=str(data.get("span_id", str(uuid.uuid4()))),
            trace_id=data.get("trace_id"),
            parent_span_id=data.get("parent_span_id"),
            agent_session_id=data.get("agent_session_id"),
            status=status,
            timestamp_utc=timestamp,
            duration_ms=data.get("duration_ms"),
            metadata=dict(data.get("metadata", {})),
        )
