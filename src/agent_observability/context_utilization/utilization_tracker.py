"""UtilizationTracker — track context window token usage over time.

Maintains a time-series history of context window utilization ratios
(tokens_used / max_tokens). Alerts when utilization exceeds a threshold
and supports querying the history for analysis.

Example
-------
::

    from agent_observability.context_utilization.utilization_tracker import UtilizationTracker

    tracker = UtilizationTracker(agent_id="agent-1", max_tokens=128000)
    tracker.record_usage(tokens_used=80000)
    snapshot = tracker.current_snapshot()
    print(snapshot.utilization_ratio, snapshot.is_high_utilization)
    alerts = tracker.get_alerts()
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class UtilizationRecord:
    """A single point-in-time utilization measurement.

    Attributes
    ----------
    agent_id:
        The agent this measurement belongs to.
    tokens_used:
        Number of tokens currently used in the context window.
    max_tokens:
        Total context window capacity in tokens.
    utilization_ratio:
        Fraction of context window used (tokens_used / max_tokens).
    timestamp_utc:
        UTC time of this measurement.
    turn_number:
        Optional conversation turn number for ordering.
    metadata:
        Additional context (e.g. model name, session ID).
    """

    agent_id: str
    tokens_used: int
    max_tokens: int
    utilization_ratio: float
    timestamp_utc: datetime
    turn_number: Optional[int] = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def tokens_remaining(self) -> int:
        """Number of tokens remaining in the context window."""
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def utilization_percent(self) -> float:
        """Utilization as a percentage (0.0–100.0)."""
        return self.utilization_ratio * 100.0

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "agent_id": self.agent_id,
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "utilization_ratio": round(self.utilization_ratio, 4),
            "utilization_percent": round(self.utilization_percent, 2),
            "tokens_remaining": self.tokens_remaining,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "turn_number": self.turn_number,
        }


@dataclass(frozen=True)
class ContextAlert:
    """Alert raised when context window utilization exceeds a threshold.

    Attributes
    ----------
    agent_id:
        The agent that triggered the alert.
    threshold:
        The threshold that was exceeded (0.0–1.0).
    utilization_ratio:
        The actual utilization ratio at alert time.
    tokens_used:
        Tokens used at alert time.
    max_tokens:
        Total context capacity.
    timestamp_utc:
        UTC time the alert was raised.
    alert_type:
        Label for the type of alert (e.g. ``"high_utilization"``, ``"full"``).
    """

    agent_id: str
    threshold: float
    utilization_ratio: float
    tokens_used: int
    max_tokens: int
    timestamp_utc: datetime
    alert_type: str = "high_utilization"

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "agent_id": self.agent_id,
            "alert_type": self.alert_type,
            "threshold": self.threshold,
            "utilization_ratio": round(self.utilization_ratio, 4),
            "utilization_percent": round(self.utilization_ratio * 100, 2),
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "timestamp_utc": self.timestamp_utc.isoformat(),
        }


@dataclass
class ContextSnapshot:
    """Current state snapshot for a UtilizationTracker.

    Attributes
    ----------
    agent_id:
        The agent this snapshot belongs to.
    current_tokens_used:
        Most recent tokens_used value.
    max_tokens:
        Total context window capacity.
    utilization_ratio:
        Current utilization ratio (0.0–1.0).
    peak_utilization_ratio:
        Highest recorded utilization ratio.
    mean_utilization_ratio:
        Mean utilization ratio over all recorded measurements.
    alert_threshold:
        The threshold above which alerts are triggered.
    alert_count:
        Number of alerts raised so far.
    measurement_count:
        Total number of utilization records stored.
    """

    agent_id: str
    current_tokens_used: int
    max_tokens: int
    utilization_ratio: float
    peak_utilization_ratio: float
    mean_utilization_ratio: float
    alert_threshold: float
    alert_count: int
    measurement_count: int

    @property
    def is_high_utilization(self) -> bool:
        """True if current utilization exceeds the alert threshold."""
        return self.utilization_ratio >= self.alert_threshold

    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining in the context window."""
        return max(0, self.max_tokens - self.current_tokens_used)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "agent_id": self.agent_id,
            "current_tokens_used": self.current_tokens_used,
            "max_tokens": self.max_tokens,
            "utilization_ratio": round(self.utilization_ratio, 4),
            "utilization_percent": round(self.utilization_ratio * 100, 2),
            "peak_utilization_ratio": round(self.peak_utilization_ratio, 4),
            "mean_utilization_ratio": round(self.mean_utilization_ratio, 4),
            "alert_threshold": self.alert_threshold,
            "is_high_utilization": self.is_high_utilization,
            "alert_count": self.alert_count,
            "measurement_count": self.measurement_count,
            "tokens_remaining": self.tokens_remaining,
        }


class UtilizationTracker:
    """Track context window utilization over time for a single agent.

    Maintains a rolling history of utilization measurements and raises
    alerts when utilization exceeds the configured threshold.

    Parameters
    ----------
    agent_id:
        Identifier of the agent being tracked.
    max_tokens:
        Total context window capacity in tokens.
    alert_threshold:
        Utilization ratio above which alerts are generated. Defaults to 0.8 (80%).
    max_history:
        Maximum number of utilization records to retain. Defaults to 1000.

    Example
    -------
    ::

        tracker = UtilizationTracker(
            agent_id="agent-1",
            max_tokens=128000,
            alert_threshold=0.8,
        )
        tracker.record_usage(tokens_used=100000)
        snapshot = tracker.current_snapshot()
        print(snapshot.is_high_utilization)  # True
    """

    def __init__(
        self,
        agent_id: str,
        max_tokens: int,
        *,
        alert_threshold: float = 0.8,
        max_history: int = 1000,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if not 0.0 < alert_threshold <= 1.0:
            raise ValueError("alert_threshold must be between 0.0 and 1.0.")

        self.agent_id = agent_id
        self.max_tokens = max_tokens
        self.alert_threshold = alert_threshold
        self._max_history = max_history

        self._history: list[UtilizationRecord] = []
        self._alerts: list[ContextAlert] = []
        self._lock = threading.Lock()

    def record_usage(
        self,
        tokens_used: int,
        *,
        turn_number: Optional[int] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> UtilizationRecord:
        """Record a context window utilization measurement.

        Parameters
        ----------
        tokens_used:
            Number of tokens currently in use.
        turn_number:
            Optional conversation turn number.
        metadata:
            Optional additional metadata to attach.

        Returns
        -------
        UtilizationRecord
            The recorded measurement.
        """
        tokens_used = max(0, min(tokens_used, self.max_tokens))
        ratio = tokens_used / self.max_tokens
        now = datetime.now(timezone.utc)

        record = UtilizationRecord(
            agent_id=self.agent_id,
            tokens_used=tokens_used,
            max_tokens=self.max_tokens,
            utilization_ratio=ratio,
            timestamp_utc=now,
            turn_number=turn_number,
            metadata=metadata or {},
        )

        with self._lock:
            self._history.append(record)
            if len(self._history) > self._max_history:
                del self._history[: len(self._history) - self._max_history]

            # Check for alert condition
            if ratio >= self.alert_threshold:
                alert_type = "full" if ratio >= 1.0 else "high_utilization"
                alert = ContextAlert(
                    agent_id=self.agent_id,
                    threshold=self.alert_threshold,
                    utilization_ratio=ratio,
                    tokens_used=tokens_used,
                    max_tokens=self.max_tokens,
                    timestamp_utc=now,
                    alert_type=alert_type,
                )
                self._alerts.append(alert)

        return record

    def get_history(
        self,
        *,
        limit: Optional[int] = None,
    ) -> list[UtilizationRecord]:
        """Return utilization history, most recent first.

        Parameters
        ----------
        limit:
            Maximum number of records to return. None means all.
        """
        with self._lock:
            history = list(reversed(self._history))
        return history[:limit] if limit is not None else history

    def get_alerts(self) -> list[ContextAlert]:
        """Return all alerts raised, most recent first."""
        with self._lock:
            return list(reversed(self._alerts))

    def current_snapshot(self) -> Optional[ContextSnapshot]:
        """Return a snapshot of the current utilization state.

        Returns None if no measurements have been recorded.
        """
        with self._lock:
            if not self._history:
                return None
            history = list(self._history)
            alert_count = len(self._alerts)

        latest = history[-1]
        peak = max(r.utilization_ratio for r in history)
        mean = sum(r.utilization_ratio for r in history) / len(history)

        return ContextSnapshot(
            agent_id=self.agent_id,
            current_tokens_used=latest.tokens_used,
            max_tokens=self.max_tokens,
            utilization_ratio=latest.utilization_ratio,
            peak_utilization_ratio=peak,
            mean_utilization_ratio=mean,
            alert_threshold=self.alert_threshold,
            alert_count=alert_count,
            measurement_count=len(history),
        )

    def clear(self) -> None:
        """Reset all history and alerts."""
        with self._lock:
            self._history.clear()
            self._alerts.clear()

    @property
    def measurement_count(self) -> int:
        """Total number of measurements recorded."""
        with self._lock:
            return len(self._history)
