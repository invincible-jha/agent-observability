"""BaselineComputer — compute statistical baseline of agent behaviour.

A baseline is a set of (mean, std_dev) pairs for each behavioural feature,
computed over a representative window of spans.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

from agent_observability.drift.features import (
    BehavioralFeatures,
    FeatureExtractor,
    SpanRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class BaselineStats:
    """Statistical summary (mean ± std) for a single feature."""

    mean: float
    std_dev: float
    sample_count: int

    def z_score(self, value: float) -> float:
        """Return the Z-score of *value* relative to this baseline."""
        if self.std_dev == 0.0:
            return 0.0
        return (value - self.mean) / self.std_dev


@dataclass
class AgentBaseline:
    """Complete behavioural baseline for one agent.

    Attributes
    ----------
    agent_id:
        Agent this baseline applies to.
    created_at:
        Unix timestamp when this baseline was computed.
    span_count:
        Total number of spans in the training window.
    feature_stats:
        ``{feature_name: BaselineStats}`` mapping.
    metadata:
        Arbitrary key/value labels (e.g. model version, environment).
    """

    agent_id: str
    created_at: float
    span_count: int
    feature_stats: dict[str, BaselineStats]
    metadata: dict[str, str] = field(default_factory=dict)

    def z_scores(self, features: BehavioralFeatures) -> dict[str, float]:
        """Compute Z-scores for all features in *features* against this baseline."""
        vec = features.to_vector()
        scores: dict[str, float] = {}
        for key, value in vec.items():
            stats = self.feature_stats.get(key)
            if stats is not None:
                scores[key] = stats.z_score(value)
        return scores

    def max_z_score(self, features: BehavioralFeatures) -> float:
        """Return the largest absolute Z-score across all features."""
        scores = self.z_scores(features)
        if not scores:
            return 0.0
        return max(abs(v) for v in scores.values())


class BaselineComputer:
    """Compute :class:`AgentBaseline` from a list of span windows.

    Each *window* is a list of :class:`SpanRecord` objects.  Calling
    :meth:`compute` with multiple windows gives a more stable baseline.

    Parameters
    ----------
    extractor:
        Optional :class:`FeatureExtractor` instance.  A default one is
        created if not supplied.
    """

    def __init__(self, extractor: FeatureExtractor | None = None) -> None:
        self._extractor = extractor or FeatureExtractor()

    def compute(
        self,
        agent_id: str,
        windows: Sequence[Sequence[SpanRecord]],
        metadata: dict[str, str] | None = None,
    ) -> AgentBaseline:
        """Compute a baseline from multiple span windows.

        Parameters
        ----------
        agent_id:
            Agent identifier for the baseline.
        windows:
            A sequence of span windows.  Each window becomes one data point
            in the feature distribution.
        metadata:
            Optional labels to attach to the baseline.

        Returns
        -------
        An :class:`AgentBaseline` with ``mean`` and ``std_dev`` for every
        feature across all windows.
        """
        if not windows:
            raise ValueError("At least one span window is required to compute a baseline")

        feature_vectors: list[dict[str, float]] = []
        total_spans = 0

        for window in windows:
            features = self._extractor.extract(window)
            feature_vectors.append(features.to_vector())
            total_spans += len(window)

        # Gather all feature keys seen across all windows
        all_keys: set[str] = set()
        for vec in feature_vectors:
            all_keys.update(vec.keys())

        feature_stats: dict[str, BaselineStats] = {}
        for key in sorted(all_keys):
            values = [vec.get(key, 0.0) for vec in feature_vectors]
            mean = _mean(values)
            std = _std_dev(values, mean)
            feature_stats[key] = BaselineStats(
                mean=mean,
                std_dev=std,
                sample_count=len(values),
            )

        baseline = AgentBaseline(
            agent_id=agent_id,
            created_at=time.time(),
            span_count=total_spans,
            feature_stats=feature_stats,
            metadata=metadata or {},
        )

        logger.info(
            "BaselineComputer: computed baseline for agent=%s from %d windows (%d spans)",
            agent_id,
            len(windows),
            total_spans,
        )
        return baseline


# ── Utilities ──────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std_dev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
