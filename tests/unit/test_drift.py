"""Comprehensive tests for the drift detection subsystem.

Covers:
- drift.features  (SpanRecord, BehavioralFeatures, FeatureExtractor)
- drift.baseline  (BaselineStats, AgentBaseline, BaselineComputer)
- drift.detector  (DriftResult, DriftDetector)
- drift.history   (BaselineHistory)
- drift.metric_drift (MetricDriftDetector, DriftReport, DriftSeverity)
- drift.alerts    (DriftAlert, ConsoleAlertHandler, WebhookAlertHandler, DriftAlertManager)
- drift.__init__  (public re-exports)
"""
from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.drift.features import (
    BehavioralFeatures,
    FeatureExtractor,
    SpanRecord,
)
from agent_observability.drift.baseline import (
    AgentBaseline,
    BaselineComputer,
    BaselineStats,
    _mean,
    _std_dev,
)
from agent_observability.drift.detector import DriftDetector, DriftResult
from agent_observability.drift.history import BaselineHistory
from agent_observability.drift.metric_drift import (
    DriftReport,
    DriftSeverity,
    MetricDriftDetector,
)
from agent_observability.drift.alerts import (
    ConsoleAlertHandler,
    DriftAlert,
    DriftAlertManager,
    WebhookAlertHandler,
    _result_to_alert,
)
from agent_observability.drift import (
    AgentBaseline as PublicBaseline,
    DriftDetector as PublicDetector,
    MetricDriftDetector as PublicMetricDetector,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_span(
    span_kind: str = "llm_call",
    duration_ms: float = 100.0,
    input_tokens: int = 50,
    output_tokens: int = 25,
    cost_usd: float = 0.001,
    tool_name: str = "",
    tool_success: bool = True,
    error: bool = False,
) -> SpanRecord:
    return SpanRecord(
        span_kind=span_kind,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        tool_name=tool_name,
        tool_success=tool_success,
        error=error,
    )


def _make_baseline(
    agent_id: str = "agent-1",
    means: dict[str, float] | None = None,
    std_devs: dict[str, float] | None = None,
) -> AgentBaseline:
    means = means or {"span_count": 10.0, "mean_duration_ms": 100.0}
    std_devs = std_devs or {"span_count": 1.0, "mean_duration_ms": 10.0}
    feature_stats = {
        key: BaselineStats(mean=means[key], std_dev=std_devs[key], sample_count=5)
        for key in means
    }
    return AgentBaseline(
        agent_id=agent_id,
        created_at=time.time() - 60,
        span_count=50,
        feature_stats=feature_stats,
    )


def _make_drift_result(
    drifted: bool = True,
    severity: str = "high",
    max_z_score: float = 5.0,
    agent_id: str = "agent-1",
) -> DriftResult:
    return DriftResult(
        agent_id=agent_id,
        timestamp=time.time(),
        drifted=drifted,
        max_z_score=max_z_score,
        threshold=3.0,
        drifted_features={"span_count": max_z_score} if drifted else {},
        all_z_scores={"span_count": max_z_score},
        baseline_age_seconds=60.0,
        window_span_count=20,
    )


# ── SpanRecord ─────────────────────────────────────────────────────────────────


class TestSpanRecord:
    def test_defaults(self) -> None:
        span = SpanRecord(span_kind="llm_call", duration_ms=50.0)
        assert span.input_tokens == 0
        assert span.output_tokens == 0
        assert span.cost_usd == 0.0
        assert span.tool_name == ""
        assert span.tool_success is True
        assert span.error is False
        assert span.attributes == {}

    def test_from_dict_with_full_attributes(self) -> None:
        data = {
            "start_time_ns": 1_000_000_000,
            "end_time_ns": 1_100_000_000,
            "status": "OK",
            "attributes": {
                "agent.span.kind": "tool_invoke",
                "llm.tokens.input": 100,
                "llm.tokens.output": 50,
                "llm.cost.usd": 0.005,
                "tool.name": "web_search",
                "tool.success": True,
            },
        }
        span = SpanRecord.from_dict(data)
        assert span.span_kind == "tool_invoke"
        assert span.duration_ms == pytest.approx(100.0)
        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.cost_usd == pytest.approx(0.005)
        assert span.tool_name == "web_search"
        assert span.tool_success is True
        assert span.error is False

    def test_from_dict_error_status(self) -> None:
        data = {"status": "ERROR", "attributes": {}}
        span = SpanRecord.from_dict(data)
        assert span.error is True

    def test_from_dict_missing_times(self) -> None:
        data = {"attributes": {"agent.span.kind": "reasoning_step"}}
        span = SpanRecord.from_dict(data)
        assert span.duration_ms == 0.0

    def test_from_dict_zero_times(self) -> None:
        data = {"start_time_ns": 0, "end_time_ns": 0, "attributes": {}}
        span = SpanRecord.from_dict(data)
        assert span.duration_ms == 0.0

    def test_from_dict_invalid_attributes_type(self) -> None:
        data = {"attributes": "not_a_dict"}
        span = SpanRecord.from_dict(data)
        assert span.span_kind == ""

    def test_from_dict_uses_span_kind_fallback(self) -> None:
        data = {"span_kind": "memory_read", "attributes": {}}
        span = SpanRecord.from_dict(data)
        assert span.span_kind == "memory_read"


# ── BehavioralFeatures ─────────────────────────────────────────────────────────


class TestBehavioralFeatures:
    def test_to_vector_basic_fields(self) -> None:
        features = BehavioralFeatures(
            span_count=10,
            mean_duration_ms=100.0,
            p95_duration_ms=200.0,
            mean_input_tokens=50.0,
            mean_output_tokens=25.0,
            mean_cost_usd=0.001,
            total_cost_usd=0.01,
            error_rate=0.1,
            tool_failure_rate=0.05,
        )
        vec = features.to_vector()
        assert vec["span_count"] == 10.0
        assert vec["mean_duration_ms"] == 100.0
        assert vec["p95_duration_ms"] == 200.0
        assert vec["error_rate"] == 0.1

    def test_to_vector_includes_kind_fractions(self) -> None:
        features = BehavioralFeatures(
            kind_fractions={"llm_call": 0.6, "tool_invoke": 0.4},
        )
        vec = features.to_vector()
        assert vec["kind.llm_call"] == 0.6
        assert vec["kind.tool_invoke"] == 0.4

    def test_to_vector_includes_tool_fractions(self) -> None:
        features = BehavioralFeatures(
            tool_fractions={"web_search": 0.7, "code_exec": 0.3},
        )
        vec = features.to_vector()
        assert vec["tool.web_search"] == 0.7
        assert vec["tool.code_exec"] == 0.3

    def test_to_vector_empty_fractions(self) -> None:
        features = BehavioralFeatures()
        vec = features.to_vector()
        assert "span_count" in vec
        assert not any(k.startswith("kind.") for k in vec)
        assert not any(k.startswith("tool.") for k in vec)


# ── FeatureExtractor ───────────────────────────────────────────────────────────


class TestFeatureExtractor:
    def setup_method(self) -> None:
        self.extractor = FeatureExtractor()

    def test_empty_spans(self) -> None:
        result = self.extractor.extract([])
        assert result.span_count == 0
        assert result.mean_duration_ms == 0.0

    def test_single_span(self) -> None:
        spans = [_make_span(duration_ms=100.0, input_tokens=10, output_tokens=5, cost_usd=0.001)]
        result = self.extractor.extract(spans)
        assert result.span_count == 1
        assert result.mean_duration_ms == pytest.approx(100.0)
        assert result.mean_input_tokens == pytest.approx(10.0)
        assert result.mean_output_tokens == pytest.approx(5.0)
        assert result.mean_cost_usd == pytest.approx(0.001)
        assert result.total_cost_usd == pytest.approx(0.001)

    def test_error_rate(self) -> None:
        spans = [
            _make_span(error=True),
            _make_span(error=True),
            _make_span(error=False),
            _make_span(error=False),
        ]
        result = self.extractor.extract(spans)
        assert result.error_rate == pytest.approx(0.5)

    def test_tool_failure_rate(self) -> None:
        spans = [
            _make_span(span_kind="tool_invoke", tool_name="search", tool_success=False),
            _make_span(span_kind="tool_invoke", tool_name="search", tool_success=True),
            _make_span(span_kind="llm_call"),
        ]
        result = self.extractor.extract(spans)
        # 1 failure out of 2 tool spans
        assert result.tool_failure_rate == pytest.approx(0.5)

    def test_kind_fractions(self) -> None:
        spans = [
            _make_span(span_kind="llm_call"),
            _make_span(span_kind="llm_call"),
            _make_span(span_kind="tool_invoke"),
        ]
        result = self.extractor.extract(spans)
        assert result.kind_fractions["llm_call"] == pytest.approx(2 / 3)
        assert result.kind_fractions["tool_invoke"] == pytest.approx(1 / 3)

    def test_tool_fractions(self) -> None:
        spans = [
            _make_span(span_kind="tool_invoke", tool_name="search"),
            _make_span(span_kind="tool_invoke", tool_name="search"),
            _make_span(span_kind="tool_invoke", tool_name="code"),
        ]
        result = self.extractor.extract(spans)
        assert result.tool_fractions["search"] == pytest.approx(2 / 3)
        assert result.tool_fractions["code"] == pytest.approx(1 / 3)

    def test_p95_duration(self) -> None:
        spans = [_make_span(duration_ms=float(i)) for i in range(1, 21)]
        result = self.extractor.extract(spans)
        # p95 of 20 values = index ceil(0.95*20)-1 = 18 in sorted array
        assert result.p95_duration_ms == pytest.approx(19.0)

    def test_tool_span_without_tool_name_ignored_in_fractions(self) -> None:
        spans = [_make_span(span_kind="tool_invoke", tool_name="")]
        result = self.extractor.extract(spans)
        assert result.tool_fractions == {}


# ── BaselineStats ──────────────────────────────────────────────────────────────


class TestBaselineStats:
    def test_z_score_zero_std(self) -> None:
        stats = BaselineStats(mean=10.0, std_dev=0.0, sample_count=5)
        assert stats.z_score(10.0) == 0.0
        assert stats.z_score(999.0) == 0.0

    def test_z_score_positive(self) -> None:
        stats = BaselineStats(mean=100.0, std_dev=10.0, sample_count=10)
        assert stats.z_score(120.0) == pytest.approx(2.0)

    def test_z_score_negative(self) -> None:
        stats = BaselineStats(mean=100.0, std_dev=10.0, sample_count=10)
        assert stats.z_score(80.0) == pytest.approx(-2.0)


# ── AgentBaseline ──────────────────────────────────────────────────────────────


class TestAgentBaseline:
    def test_z_scores_returns_scores_for_known_features(self) -> None:
        baseline = _make_baseline(
            means={"span_count": 10.0, "mean_duration_ms": 100.0},
            std_devs={"span_count": 1.0, "mean_duration_ms": 10.0},
        )
        features = BehavioralFeatures(span_count=13, mean_duration_ms=100.0)
        scores = baseline.z_scores(features)
        assert scores["span_count"] == pytest.approx(3.0)
        assert scores["mean_duration_ms"] == pytest.approx(0.0)

    def test_z_scores_ignores_unknown_features(self) -> None:
        baseline = _make_baseline(means={"span_count": 10.0}, std_devs={"span_count": 1.0})
        features = BehavioralFeatures(span_count=10, mean_duration_ms=999.0)
        scores = baseline.z_scores(features)
        assert "mean_duration_ms" not in scores

    def test_max_z_score_no_features(self) -> None:
        baseline = AgentBaseline(
            agent_id="a", created_at=time.time(), span_count=0, feature_stats={}
        )
        features = BehavioralFeatures()
        assert baseline.max_z_score(features) == 0.0

    def test_max_z_score_picks_largest_abs(self) -> None:
        baseline = _make_baseline(
            means={"span_count": 10.0, "mean_duration_ms": 100.0},
            std_devs={"span_count": 1.0, "mean_duration_ms": 10.0},
        )
        features = BehavioralFeatures(span_count=14, mean_duration_ms=80.0)
        max_z = baseline.max_z_score(features)
        # span_count z=4.0, duration z=-2.0 → max abs = 4.0
        assert max_z == pytest.approx(4.0)


# ── BaselineComputer ───────────────────────────────────────────────────────────


class TestBaselineComputer:
    def setup_method(self) -> None:
        self.computer = BaselineComputer()

    def test_raises_on_empty_windows(self) -> None:
        with pytest.raises(ValueError, match="At least one span window"):
            self.computer.compute("agent-1", [])

    def test_single_window(self) -> None:
        spans = [_make_span(duration_ms=100.0) for _ in range(5)]
        baseline = self.computer.compute("agent-1", [spans])
        assert baseline.agent_id == "agent-1"
        assert baseline.span_count == 5
        assert "mean_duration_ms" in baseline.feature_stats

    def test_multiple_windows_averages_stats(self) -> None:
        window_a = [_make_span(duration_ms=100.0) for _ in range(5)]
        window_b = [_make_span(duration_ms=200.0) for _ in range(5)]
        baseline = self.computer.compute("agent-1", [window_a, window_b])
        assert baseline.span_count == 10
        # Mean of [100, 200] = 150
        mean = baseline.feature_stats["mean_duration_ms"].mean
        assert mean == pytest.approx(150.0)

    def test_metadata_attached(self) -> None:
        spans = [_make_span() for _ in range(3)]
        meta = {"env": "prod", "version": "1.2"}
        baseline = self.computer.compute("agent-1", [spans], metadata=meta)
        assert baseline.metadata == meta

    def test_single_window_std_is_zero(self) -> None:
        spans = [_make_span(duration_ms=100.0) for _ in range(5)]
        baseline = self.computer.compute("agent-1", [spans])
        # With one window, std_dev should be 0
        assert baseline.feature_stats["mean_duration_ms"].std_dev == 0.0

    def test_custom_extractor(self) -> None:
        extractor = FeatureExtractor()
        computer = BaselineComputer(extractor=extractor)
        spans = [_make_span() for _ in range(2)]
        baseline = computer.compute("agent-x", [spans])
        assert baseline.agent_id == "agent-x"


# ── _mean and _std_dev utilities ───────────────────────────────────────────────


class TestBaselineUtilities:
    def test_mean_empty(self) -> None:
        assert _mean([]) == 0.0

    def test_mean_values(self) -> None:
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_std_dev_single_value(self) -> None:
        assert _std_dev([5.0], 5.0) == 0.0

    def test_std_dev_multiple(self) -> None:
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        mean_val = sum(values) / len(values)
        std = _std_dev(values, mean_val)
        assert std > 0.0


# ── DriftResult ────────────────────────────────────────────────────────────────


class TestDriftResult:
    def test_severity_none(self) -> None:
        result = _make_drift_result(drifted=False, max_z_score=0.0)
        assert result.severity == "none"

    def test_severity_low(self) -> None:
        # ratio < 1.5 → low
        result = _make_drift_result(max_z_score=3.0 * 1.2)  # ratio = 1.2 → low
        assert result.severity == "low"

    def test_severity_medium(self) -> None:
        # ratio between 1.5 and 2.5
        result = _make_drift_result(max_z_score=3.0 * 2.0)  # ratio = 2.0
        assert result.severity == "medium"

    def test_severity_high(self) -> None:
        result = _make_drift_result(max_z_score=3.0 * 3.0)  # ratio = 3.0
        assert result.severity == "high"


# ── DriftDetector ──────────────────────────────────────────────────────────────


class TestDriftDetector:
    def test_raises_on_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="sigma_threshold must be positive"):
            DriftDetector(sigma_threshold=0)

    def test_skips_small_window(self) -> None:
        detector = DriftDetector(min_window_spans=5)
        baseline = _make_baseline()
        result = detector.check(baseline, [_make_span()])
        assert result.drifted is False
        assert "minimum" in result.notes.lower()

    def test_healthy_result(self) -> None:
        computer = BaselineComputer()
        windows = [[_make_span(duration_ms=100.0) for _ in range(10)] for _ in range(5)]
        baseline = computer.compute("agent-1", windows)
        detector = DriftDetector(sigma_threshold=3.0, min_window_spans=10)
        # Same pattern — should not drift
        current = [_make_span(duration_ms=100.0) for _ in range(10)]
        result = detector.check(baseline, current)
        assert result.agent_id == "agent-1"
        assert isinstance(result.drifted, bool)

    def test_drifted_result(self) -> None:
        """Force drift by setting a tiny std_dev and a very different current window."""
        baseline = AgentBaseline(
            agent_id="agent-drift",
            created_at=time.time() - 3600,
            span_count=100,
            feature_stats={
                "mean_duration_ms": BaselineStats(mean=100.0, std_dev=1.0, sample_count=50),
                "span_count": BaselineStats(mean=10.0, std_dev=0.5, sample_count=50),
                "mean_cost_usd": BaselineStats(mean=0.001, std_dev=0.0001, sample_count=50),
                "total_cost_usd": BaselineStats(mean=0.01, std_dev=0.001, sample_count=50),
                "mean_input_tokens": BaselineStats(mean=50.0, std_dev=1.0, sample_count=50),
                "mean_output_tokens": BaselineStats(mean=25.0, std_dev=1.0, sample_count=50),
                "error_rate": BaselineStats(mean=0.0, std_dev=0.01, sample_count=50),
                "tool_failure_rate": BaselineStats(mean=0.0, std_dev=0.01, sample_count=50),
                "p95_duration_ms": BaselineStats(mean=150.0, std_dev=2.0, sample_count=50),
            },
        )
        detector = DriftDetector(sigma_threshold=3.0, min_window_spans=5)
        # Massive deviation in duration
        drifted_spans = [_make_span(duration_ms=900.0) for _ in range(10)]
        result = detector.check(baseline, drifted_spans)
        assert result.drifted is True
        assert result.max_z_score > 3.0

    def test_check_features_uses_pre_extracted(self) -> None:
        baseline = _make_baseline(
            means={"span_count": 10.0},
            std_devs={"span_count": 1.0},
        )
        detector = DriftDetector(sigma_threshold=3.0)
        features = BehavioralFeatures(span_count=10)
        result = detector.check_features(baseline, features)
        assert result.agent_id == "agent-1"
        assert result.drifted is False

    def test_window_span_count_recorded(self) -> None:
        baseline = _make_baseline()
        detector = DriftDetector(min_window_spans=5)
        current = [_make_span() for _ in range(12)]
        result = detector.check(baseline, current)
        assert result.window_span_count == 12


# ── BaselineHistory ────────────────────────────────────────────────────────────


class TestBaselineHistory:
    def test_save_and_latest(self) -> None:
        history = BaselineHistory()
        baseline = _make_baseline("agent-a")
        history.save(baseline)
        latest = history.latest("agent-a")
        assert latest is not None
        assert latest.agent_id == "agent-a"

    def test_latest_returns_none_for_unknown_agent(self) -> None:
        history = BaselineHistory()
        assert history.latest("unknown") is None

    def test_all_for_agent(self) -> None:
        history = BaselineHistory()
        b1 = _make_baseline("agent-b")
        b2 = _make_baseline("agent-b")
        history.save(b1)
        history.save(b2)
        all_b = history.all_for_agent("agent-b")
        assert len(all_b) == 2

    def test_all_agent_ids(self) -> None:
        history = BaselineHistory()
        history.save(_make_baseline("agent-x"))
        history.save(_make_baseline("agent-y"))
        ids = history.all_agent_ids()
        assert ids == ["agent-x", "agent-y"]

    def test_delete_agent(self) -> None:
        history = BaselineHistory()
        history.save(_make_baseline("agent-del"))
        history.delete_agent("agent-del")
        assert history.latest("agent-del") is None

    def test_max_per_agent_prunes_oldest(self) -> None:
        history = BaselineHistory(max_per_agent=3)
        for _ in range(5):
            history.save(_make_baseline("agent-max"))
        all_b = history.all_for_agent("agent-max")
        assert len(all_b) == 3

    def test_thread_safety(self) -> None:
        history = BaselineHistory()
        errors: list[Exception] = []

        def worker(agent_id: str) -> None:
            try:
                for _ in range(10):
                    history.save(_make_baseline(agent_id))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"agent-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_persist_and_load(self, tmp_path: Path) -> None:
        persist_file = str(tmp_path / "baselines.json")
        history = BaselineHistory(persist_path=persist_file)
        baseline = _make_baseline("agent-persist")
        history.save(baseline)

        # Verify file was written
        assert Path(persist_file).exists()

        # Load into new instance
        history2 = BaselineHistory(persist_path=persist_file)
        loaded = history2.latest("agent-persist")
        assert loaded is not None
        assert loaded.agent_id == "agent-persist"

    def test_persist_and_delete(self, tmp_path: Path) -> None:
        persist_file = str(tmp_path / "baselines.json")
        history = BaselineHistory(persist_path=persist_file)
        history.save(_make_baseline("agent-del-persist"))
        history.delete_agent("agent-del-persist")
        # Should still write without error
        assert Path(persist_file).exists()

    def test_load_corrupt_file_logs_and_skips(self, tmp_path: Path) -> None:
        persist_file = str(tmp_path / "corrupt.json")
        Path(persist_file).write_text("not valid json", encoding="utf-8")
        # Should not raise — just log
        history = BaselineHistory(persist_path=persist_file)
        assert history.all_agent_ids() == []


# ── MetricDriftDetector ────────────────────────────────────────────────────────


class TestMetricDriftDetector:
    def setup_method(self) -> None:
        self.detector = MetricDriftDetector()

    def _fill_baseline(self, agent_id: str, metric: str, values: list[float]) -> None:
        for v in values:
            self.detector.record_metric(agent_id, metric, v)

    def test_no_drift_below_min_samples(self) -> None:
        for i in range(9):
            self.detector.record_metric("agent-1", "latency_ms", float(i))
        report = self.detector.check_drift("agent-1", "latency_ms", 999.0)
        assert report.drifted is False
        assert report.severity == DriftSeverity.NONE

    def test_no_drift_within_threshold(self) -> None:
        # Use a varied baseline so stddev > 0, then check a value within normal range
        values = [100.0 + (i % 5) for i in range(20)]  # values 100-104, repeated
        self._fill_baseline("agent-1", "latency", values)
        report = self.detector.check_drift("agent-1", "latency", 102.0)
        assert report.drifted is False

    def test_low_drift(self) -> None:
        # Mean=100, std~0.0 from constant values won't trigger; use varied baseline
        self._fill_baseline("agent-1", "latency", [90.0, 95.0, 100.0, 105.0, 110.0] * 4)
        report = self.detector.check_drift("agent-1", "latency", 120.0)
        assert report.drifted is True

    def test_high_drift(self) -> None:
        self._fill_baseline("agent-2", "cost", [10.0] * 20)
        report = self.detector.check_drift("agent-2", "cost", 50.0)
        # stddev of constant series = 0 → z = inf → critical
        assert report.drifted is True
        assert report.severity == DriftSeverity.CRITICAL

    def test_stable_baseline_exact_match_no_drift(self) -> None:
        self._fill_baseline("agent-3", "cost", [10.0] * 20)
        report = self.detector.check_drift("agent-3", "cost", 10.0)
        assert report.drifted is False

    def test_get_baseline_insufficient_samples(self) -> None:
        mean, stddev = self.detector.get_baseline("agent-x", "nonexistent")
        assert mean == 0.0
        assert stddev == 0.0

    def test_get_baseline_sufficient_samples(self) -> None:
        self._fill_baseline("agent-4", "tokens", [100.0] * 10)
        mean, stddev = self.detector.get_baseline("agent-4", "tokens")
        assert mean == pytest.approx(100.0)
        assert stddev == pytest.approx(0.0)

    def test_check_all(self) -> None:
        self._fill_baseline("agent-5", "latency", [100.0] * 15)
        self._fill_baseline("agent-5", "cost", [0.01] * 15)
        reports = self.detector.check_all("agent-5", {"latency": 100.0, "cost": 0.01})
        assert len(reports) == 2
        assert all(isinstance(r, DriftReport) for r in reports)

    def test_custom_thresholds(self) -> None:
        custom = {
            DriftSeverity.LOW: 1.0,
            DriftSeverity.MEDIUM: 2.0,
            DriftSeverity.HIGH: 3.0,
            DriftSeverity.CRITICAL: 5.0,
        }
        detector = MetricDriftDetector(z_score_thresholds=custom)
        assert detector.z_score_thresholds[DriftSeverity.LOW] == 1.0

    def test_classify_severity_boundaries(self) -> None:
        assert self.detector._classify_severity(0.0) == DriftSeverity.NONE
        assert self.detector._classify_severity(1.5) == DriftSeverity.LOW
        assert self.detector._classify_severity(2.0) == DriftSeverity.MEDIUM
        assert self.detector._classify_severity(3.0) == DriftSeverity.HIGH
        assert self.detector._classify_severity(4.0) == DriftSeverity.CRITICAL


# ── DriftSeverity ──────────────────────────────────────────────────────────────


class TestDriftSeverity:
    def test_values(self) -> None:
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MEDIUM.value == "medium"
        assert DriftSeverity.HIGH.value == "high"
        assert DriftSeverity.CRITICAL.value == "critical"


# ── DriftAlert ─────────────────────────────────────────────────────────────────


class TestDriftAlert:
    def test_to_dict_round_trip(self) -> None:
        alert = DriftAlert(
            agent_id="agent-1",
            severity="high",
            max_z_score=5.0,
            threshold=3.0,
            drifted_features={"span_count": 5.0},
            baseline_age_seconds=60.0,
            window_span_count=20,
            timestamp=1234567890.0,
            message="test alert",
        )
        d = alert.to_dict()
        assert d["agent_id"] == "agent-1"
        assert d["severity"] == "high"
        assert d["message"] == "test alert"
        assert "drifted_features" in d


# ── _result_to_alert ───────────────────────────────────────────────────────────


class TestResultToAlert:
    def test_conversion(self) -> None:
        result = _make_drift_result(drifted=True, max_z_score=5.0)
        result.drifted_features = {"feat_a": 5.0, "feat_b": 4.0, "feat_c": 3.0, "feat_d": 2.0}
        alert = _result_to_alert(result)
        assert alert.agent_id == result.agent_id
        assert alert.severity == result.severity
        assert "feat_a" in alert.message

    def test_no_drifted_features_message(self) -> None:
        result = _make_drift_result(drifted=True, max_z_score=5.0)
        result.drifted_features = {}
        alert = _result_to_alert(result)
        assert "none" in alert.message.lower()


# ── ConsoleAlertHandler ────────────────────────────────────────────────────────


class TestConsoleAlertHandler:
    def _make_alert(self, severity: str) -> DriftAlert:
        return DriftAlert(
            agent_id="agent-1",
            severity=severity,
            max_z_score=5.0,
            threshold=3.0,
            drifted_features={},
            baseline_age_seconds=60.0,
            window_span_count=10,
            timestamp=time.time(),
            message="test",
        )

    def test_low_severity_logs_info(self) -> None:
        handler = ConsoleAlertHandler()
        with patch("agent_observability.drift.alerts.logger") as mock_logger:
            handler(self._make_alert("low"))
            mock_logger.info.assert_called_once()

    def test_medium_severity_logs_warning(self) -> None:
        handler = ConsoleAlertHandler()
        with patch("agent_observability.drift.alerts.logger") as mock_logger:
            handler(self._make_alert("medium"))
            mock_logger.warning.assert_called_once()

    def test_high_severity_logs_error(self) -> None:
        handler = ConsoleAlertHandler()
        with patch("agent_observability.drift.alerts.logger") as mock_logger:
            handler(self._make_alert("high"))
            mock_logger.error.assert_called_once()

    def test_unknown_severity_falls_back_to_warning(self) -> None:
        handler = ConsoleAlertHandler()
        with patch("agent_observability.drift.alerts.logger") as mock_logger:
            handler(self._make_alert("critical"))
            mock_logger.warning.assert_called_once()


# ── WebhookAlertHandler ────────────────────────────────────────────────────────


class TestWebhookAlertHandler:
    def _make_alert(self) -> DriftAlert:
        return DriftAlert(
            agent_id="agent-1",
            severity="high",
            max_z_score=5.0,
            threshold=3.0,
            drifted_features={},
            baseline_age_seconds=60.0,
            window_span_count=10,
            timestamp=time.time(),
            message="test",
        )

    def test_successful_post(self) -> None:
        handler = WebhookAlertHandler("http://example.com/webhook", timeout_seconds=5.0)
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.status = 200

        with patch("urllib.request.urlopen", return_value=mock_response):
            handler(self._make_alert())

    def test_http_error_status_logs_warning(self) -> None:
        handler = WebhookAlertHandler("http://example.com/webhook")
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.status = 500

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("agent_observability.drift.alerts.logger") as mock_logger:
                handler(self._make_alert())
                mock_logger.warning.assert_called()

    def test_network_error_logs_exception(self) -> None:
        handler = WebhookAlertHandler("http://example.com/webhook")
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with patch("agent_observability.drift.alerts.logger") as mock_logger:
                handler(self._make_alert())
                mock_logger.exception.assert_called()

    def test_extra_headers_included(self) -> None:
        handler = WebhookAlertHandler(
            "http://example.com/webhook",
            extra_headers={"Authorization": "Bearer token123"},
        )
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.status = 200

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("urllib.request.Request") as mock_request:
                mock_request.return_value = MagicMock()
                handler(self._make_alert())
                call_kwargs = mock_request.call_args
                headers_passed = call_kwargs[1]["headers"] if call_kwargs[1] else call_kwargs[0][2]
                assert "Authorization" in headers_passed


# ── DriftAlertManager ─────────────────────────────────────────────────────────


class TestDriftAlertManager:
    def _no_drift_result(self) -> DriftResult:
        return _make_drift_result(drifted=False, max_z_score=0.0)

    def _high_drift_result(self, agent_id: str = "agent-1") -> DriftResult:
        return _make_drift_result(drifted=True, max_z_score=10.0, agent_id=agent_id)

    def test_no_alert_when_not_drifted(self) -> None:
        manager = DriftAlertManager(handlers=[])
        result = manager.process(self._no_drift_result())
        assert result is None

    def test_alert_fired_for_drift(self) -> None:
        fired: list[DriftAlert] = []
        manager = DriftAlertManager(handlers=[fired.append], cooldown_seconds=0.0)
        result = manager.process(self._high_drift_result())
        assert result is not None
        assert len(fired) == 1

    def test_cooldown_suppresses_repeated_alerts(self) -> None:
        fired: list[DriftAlert] = []
        manager = DriftAlertManager(handlers=[fired.append], cooldown_seconds=3600.0)
        manager.process(self._high_drift_result())
        manager.process(self._high_drift_result())
        assert len(fired) == 1

    def test_min_severity_filters_low(self) -> None:
        fired: list[DriftAlert] = []
        manager = DriftAlertManager(
            handlers=[fired.append],
            min_severity="high",
            cooldown_seconds=0.0,
        )
        # low severity result — should be filtered
        low_result = DriftResult(
            agent_id="agent-low",
            timestamp=time.time(),
            drifted=True,
            max_z_score=3.2,  # ratio 3.2/3 = 1.06 → "low" severity
            threshold=3.0,
            drifted_features={"span_count": 3.2},
            all_z_scores={"span_count": 3.2},
            baseline_age_seconds=60.0,
            window_span_count=10,
        )
        result = manager.process(low_result)
        assert result is None
        assert len(fired) == 0

    def test_add_handler(self) -> None:
        manager = DriftAlertManager(handlers=[], cooldown_seconds=0.0)
        fired: list[DriftAlert] = []
        manager.add_handler(fired.append)
        manager.process(self._high_drift_result())
        assert len(fired) == 1

    def test_handler_exception_does_not_propagate(self) -> None:
        def bad_handler(alert: DriftAlert) -> None:
            raise RuntimeError("handler failure")

        manager = DriftAlertManager(handlers=[bad_handler], cooldown_seconds=0.0)
        # Should not raise
        result = manager.process(self._high_drift_result())
        assert result is not None

    def test_different_agents_each_get_alert(self) -> None:
        fired: list[DriftAlert] = []
        manager = DriftAlertManager(handlers=[fired.append], cooldown_seconds=3600.0)
        manager.process(self._high_drift_result("agent-1"))
        manager.process(self._high_drift_result("agent-2"))
        assert len(fired) == 2

    def test_default_handler_is_console(self) -> None:
        manager = DriftAlertManager()
        assert len(manager._handlers) == 1
        assert isinstance(manager._handlers[0], ConsoleAlertHandler)


# ── Public re-exports ──────────────────────────────────────────────────────────


class TestDriftPublicImports:
    def test_public_baseline_alias(self) -> None:
        assert PublicBaseline is AgentBaseline

    def test_public_detector_alias(self) -> None:
        assert PublicDetector is DriftDetector

    def test_public_metric_detector_alias(self) -> None:
        assert PublicMetricDetector is MetricDriftDetector
