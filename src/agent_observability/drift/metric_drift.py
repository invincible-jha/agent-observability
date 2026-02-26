"""Metric-level drift detection using Z-scores over recorded scalar values.

Provides a simpler, self-contained drift detector that works directly on
named scalar metrics rather than on span feature vectors.  Use this when
you want to track a specific numeric signal (e.g. ``latency_ms``,
``tokens_per_call``) and be alerted when it deviates from its historical mean.

For full span-level behavioral drift detection see
:class:`~agent_observability.drift.detector.DriftDetector`.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum number of historical samples required before drift detection activates.
_MIN_SAMPLES: int = 10


class DriftSeverity(str, Enum):
    """Qualitative severity bands for a drift report."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftReport:
    """Result of a single metric drift check."""

    metric_name: str
    current_value: float
    baseline_mean: float
    baseline_stddev: float
    z_score: float
    severity: DriftSeverity
    drifted: bool


class MetricDriftDetector:
    """Detect when a scalar metric deviates from its recorded baseline.

    Z-score thresholds default to:

    * LOW: |z| >= 1.5
    * MEDIUM: |z| >= 2.0
    * HIGH: |z| >= 3.0
    * CRITICAL: |z| >= 4.0

    Parameters
    ----------
    z_score_thresholds:
        Override the default severity thresholds.  Must be a mapping of
        :class:`DriftSeverity` to the minimum absolute Z-score for that band.
    """

    _DEFAULT_THRESHOLDS: dict[DriftSeverity, float] = {
        DriftSeverity.LOW: 1.5,
        DriftSeverity.MEDIUM: 2.0,
        DriftSeverity.HIGH: 3.0,
        DriftSeverity.CRITICAL: 4.0,
    }

    def __init__(
        self,
        z_score_thresholds: Optional[dict[DriftSeverity, float]] = None,
    ) -> None:
        self.z_score_thresholds: dict[DriftSeverity, float] = (
            z_score_thresholds if z_score_thresholds is not None else dict(self._DEFAULT_THRESHOLDS)
        )
        # Storage: agent_id -> metric_name -> list[float]
        self._history: dict[str, dict[str, list[float]]] = {}

    # ── Record ────────────────────────────────────────────────────────────────

    def record_metric(
        self,
        agent_id: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Append *value* to the historical record for (agent_id, metric_name).

        Parameters
        ----------
        agent_id:
            Identifier for the agent whose metric is being recorded.
        metric_name:
            Name of the metric (e.g. ``"latency_ms"``).
        value:
            Observed scalar value.
        """
        agent_history = self._history.setdefault(agent_id, {})
        metric_history = agent_history.setdefault(metric_name, [])
        metric_history.append(value)

    # ── Baseline ──────────────────────────────────────────────────────────────

    def get_baseline(self, agent_id: str, metric_name: str) -> tuple[float, float]:
        """Return the (mean, stddev) of recorded values for the given metric.

        Returns ``(0.0, 0.0)`` when fewer than :data:`_MIN_SAMPLES` values have
        been recorded.

        Parameters
        ----------
        agent_id:
            Agent identifier.
        metric_name:
            Metric name.

        Returns
        -------
        ``(mean, stddev)`` as a tuple of floats.
        """
        values = self._history.get(agent_id, {}).get(metric_name, [])
        if len(values) < _MIN_SAMPLES:
            return (0.0, 0.0)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = math.sqrt(variance)
        return (mean, stddev)

    # ── Check ─────────────────────────────────────────────────────────────────

    def check_drift(
        self,
        agent_id: str,
        metric_name: str,
        current_value: float,
    ) -> DriftReport:
        """Check whether *current_value* deviates from the recorded baseline.

        If fewer than :data:`_MIN_SAMPLES` values have been recorded the report
        will always have ``drifted=False`` and ``severity=NONE``.

        Parameters
        ----------
        agent_id:
            Agent identifier.
        metric_name:
            Metric name to check.
        current_value:
            The value to test against the baseline.

        Returns
        -------
        A :class:`DriftReport` instance.
        """
        values = self._history.get(agent_id, {}).get(metric_name, [])

        if len(values) < _MIN_SAMPLES:
            return DriftReport(
                metric_name=metric_name,
                current_value=current_value,
                baseline_mean=0.0,
                baseline_stddev=0.0,
                z_score=0.0,
                severity=DriftSeverity.NONE,
                drifted=False,
            )

        mean, stddev = self.get_baseline(agent_id, metric_name)

        if stddev == 0.0:
            # Perfectly stable baseline — any deviation is technically infinite
            z_score = 0.0 if current_value == mean else float("inf")
        else:
            z_score = abs(current_value - mean) / stddev

        severity = self._classify_severity(z_score)
        drifted = severity != DriftSeverity.NONE

        report = DriftReport(
            metric_name=metric_name,
            current_value=current_value,
            baseline_mean=mean,
            baseline_stddev=stddev,
            z_score=z_score,
            severity=severity,
            drifted=drifted,
        )

        if drifted:
            logger.warning(
                "MetricDriftDetector: agent=%s metric=%s drifted! z=%.2f severity=%s",
                agent_id,
                metric_name,
                z_score,
                severity.value,
            )

        return report

    def check_all(
        self,
        agent_id: str,
        current_metrics: dict[str, float],
    ) -> list[DriftReport]:
        """Check drift for every metric in *current_metrics*.

        Parameters
        ----------
        agent_id:
            Agent identifier.
        current_metrics:
            Mapping of metric_name -> current_value.

        Returns
        -------
        One :class:`DriftReport` per metric in *current_metrics*.
        """
        return [
            self.check_drift(agent_id, metric_name, value)
            for metric_name, value in current_metrics.items()
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_severity(self, z_score: float) -> DriftSeverity:
        """Map an absolute Z-score to a :class:`DriftSeverity` band."""
        critical_threshold = self.z_score_thresholds.get(DriftSeverity.CRITICAL, 4.0)
        high_threshold = self.z_score_thresholds.get(DriftSeverity.HIGH, 3.0)
        medium_threshold = self.z_score_thresholds.get(DriftSeverity.MEDIUM, 2.0)
        low_threshold = self.z_score_thresholds.get(DriftSeverity.LOW, 1.5)

        if z_score >= critical_threshold:
            return DriftSeverity.CRITICAL
        if z_score >= high_threshold:
            return DriftSeverity.HIGH
        if z_score >= medium_threshold:
            return DriftSeverity.MEDIUM
        if z_score >= low_threshold:
            return DriftSeverity.LOW
        return DriftSeverity.NONE
