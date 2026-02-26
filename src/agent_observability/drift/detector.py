"""DriftDetector — detect when agent behaviour deviates from baseline.

Uses Z-scores over behavioural feature vectors.  The threshold is configurable
so callers can tune sensitivity independently per feature.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from agent_observability.drift.baseline import AgentBaseline
from agent_observability.drift.features import (
    BehavioralFeatures,
    FeatureExtractor,
    SpanRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of a single drift check."""

    agent_id: str
    timestamp: float
    drifted: bool
    max_z_score: float
    threshold: float
    drifted_features: dict[str, float]
    all_z_scores: dict[str, float]
    baseline_age_seconds: float
    window_span_count: int
    notes: str = ""

    @property
    def severity(self) -> str:
        """Qualitative severity label based on max Z-score vs. threshold."""
        ratio = self.max_z_score / max(self.threshold, 0.001)
        if ratio < 1.0:
            return "none"
        if ratio < 1.5:
            return "low"
        if ratio < 2.5:
            return "medium"
        return "high"


class DriftDetector:
    """Detect when agent behaviour deviates more than *sigma_threshold* std devs.

    Parameters
    ----------
    sigma_threshold:
        Number of standard deviations above which a feature is flagged as
        drifted.  Defaults to ``3.0`` (3σ).
    min_window_spans:
        Minimum number of spans required to run a drift check.  Checks on
        smaller windows are skipped and return a ``drifted=False`` result
        with a note explaining the skip.
    extractor:
        Optional :class:`FeatureExtractor` instance.
    """

    def __init__(
        self,
        sigma_threshold: float = 3.0,
        min_window_spans: int = 10,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        if sigma_threshold <= 0:
            raise ValueError("sigma_threshold must be positive")
        self._threshold = sigma_threshold
        self._min_window = min_window_spans
        self._extractor = extractor or FeatureExtractor()

    # ── Check ─────────────────────────────────────────────────────────────────

    def check(
        self,
        baseline: AgentBaseline,
        window: Sequence[SpanRecord],
    ) -> DriftResult:
        """Compare *window* against *baseline* and return a :class:`DriftResult`.

        Parameters
        ----------
        baseline:
            The reference :class:`AgentBaseline` to compare against.
        window:
            Recent spans representing current behaviour.
        """
        now = time.time()
        baseline_age = now - baseline.created_at

        if len(window) < self._min_window:
            return DriftResult(
                agent_id=baseline.agent_id,
                timestamp=now,
                drifted=False,
                max_z_score=0.0,
                threshold=self._threshold,
                drifted_features={},
                all_z_scores={},
                baseline_age_seconds=baseline_age,
                window_span_count=len(window),
                notes=(
                    f"Window has {len(window)} spans — "
                    f"minimum {self._min_window} required. Skipping check."
                ),
            )

        features = self._extractor.extract(window)
        z_scores = baseline.z_scores(features)

        drifted_features = {
            key: z for key, z in z_scores.items() if abs(z) > self._threshold
        }
        max_z = max((abs(z) for z in z_scores.values()), default=0.0)
        drifted = bool(drifted_features)

        result = DriftResult(
            agent_id=baseline.agent_id,
            timestamp=now,
            drifted=drifted,
            max_z_score=max_z,
            threshold=self._threshold,
            drifted_features=drifted_features,
            all_z_scores=z_scores,
            baseline_age_seconds=baseline_age,
            window_span_count=len(window),
        )

        if drifted:
            logger.warning(
                "DriftDetector: agent=%s drifted! max_z=%.2f features=%s",
                baseline.agent_id,
                max_z,
                list(drifted_features.keys()),
            )
        else:
            logger.debug(
                "DriftDetector: agent=%s healthy (max_z=%.2f)",
                baseline.agent_id,
                max_z,
            )

        return result

    def check_features(
        self,
        baseline: AgentBaseline,
        features: BehavioralFeatures,
    ) -> DriftResult:
        """Like :meth:`check` but accepts pre-computed :class:`BehavioralFeatures`."""
        now = time.time()
        baseline_age = now - baseline.created_at
        z_scores = baseline.z_scores(features)
        drifted_features = {
            key: z for key, z in z_scores.items() if abs(z) > self._threshold
        }
        max_z = max((abs(z) for z in z_scores.values()), default=0.0)

        return DriftResult(
            agent_id=baseline.agent_id,
            timestamp=now,
            drifted=bool(drifted_features),
            max_z_score=max_z,
            threshold=self._threshold,
            drifted_features=drifted_features,
            all_z_scores=z_scores,
            baseline_age_seconds=baseline_age,
            window_span_count=features.span_count,
        )
