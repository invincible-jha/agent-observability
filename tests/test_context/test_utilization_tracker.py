"""Tests for UtilizationTracker."""
from __future__ import annotations

import pytest

from agent_observability.context_utilization.utilization_tracker import (
    ContextAlert,
    ContextSnapshot,
    UtilizationRecord,
    UtilizationTracker,
)


class TestUtilizationRecord:
    def test_tokens_remaining(self) -> None:
        from datetime import datetime, timezone
        record = UtilizationRecord(
            agent_id="agent-1",
            tokens_used=30000,
            max_tokens=128000,
            utilization_ratio=30000 / 128000,
            timestamp_utc=datetime.now(timezone.utc),
        )
        assert record.tokens_remaining == 98000

    def test_utilization_percent(self) -> None:
        from datetime import datetime, timezone
        record = UtilizationRecord(
            agent_id="agent-1",
            tokens_used=64000,
            max_tokens=128000,
            utilization_ratio=0.5,
            timestamp_utc=datetime.now(timezone.utc),
        )
        assert record.utilization_percent == pytest.approx(50.0)

    def test_to_dict_structure(self) -> None:
        from datetime import datetime, timezone
        record = UtilizationRecord(
            agent_id="agent-1",
            tokens_used=10000,
            max_tokens=100000,
            utilization_ratio=0.1,
            timestamp_utc=datetime.now(timezone.utc),
        )
        d = record.to_dict()
        assert d["agent_id"] == "agent-1"
        assert d["tokens_used"] == 10000
        assert "utilization_ratio" in d
        assert "tokens_remaining" in d


class TestContextAlert:
    def test_to_dict_structure(self) -> None:
        from datetime import datetime, timezone
        alert = ContextAlert(
            agent_id="agent-1",
            threshold=0.8,
            utilization_ratio=0.85,
            tokens_used=85000,
            max_tokens=100000,
            timestamp_utc=datetime.now(timezone.utc),
        )
        d = alert.to_dict()
        assert d["alert_type"] == "high_utilization"
        assert d["threshold"] == 0.8
        assert d["utilization_ratio"] == pytest.approx(0.85)


class TestUtilizationTracker:
    def setup_method(self) -> None:
        self.tracker = UtilizationTracker(
            agent_id="agent-1",
            max_tokens=100000,
            alert_threshold=0.8,
        )

    def test_raises_on_zero_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            UtilizationTracker(agent_id="a", max_tokens=0)

    def test_raises_on_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="alert_threshold"):
            UtilizationTracker(agent_id="a", max_tokens=1000, alert_threshold=0.0)

    def test_initial_measurement_count_zero(self) -> None:
        assert self.tracker.measurement_count == 0

    def test_record_usage_returns_record(self) -> None:
        record = self.tracker.record_usage(tokens_used=10000)
        assert isinstance(record, UtilizationRecord)
        assert record.tokens_used == 10000

    def test_record_usage_increments_count(self) -> None:
        self.tracker.record_usage(tokens_used=10000)
        assert self.tracker.measurement_count == 1

    def test_record_usage_computes_ratio(self) -> None:
        record = self.tracker.record_usage(tokens_used=50000)
        assert record.utilization_ratio == pytest.approx(0.5)

    def test_record_usage_clamps_to_max(self) -> None:
        record = self.tracker.record_usage(tokens_used=200000)
        assert record.tokens_used == 100000
        assert record.utilization_ratio == pytest.approx(1.0)

    def test_record_usage_clamps_to_zero(self) -> None:
        record = self.tracker.record_usage(tokens_used=-5000)
        assert record.tokens_used == 0

    def test_no_alert_below_threshold(self) -> None:
        self.tracker.record_usage(tokens_used=70000)  # 70% < 80% threshold
        alerts = self.tracker.get_alerts()
        assert len(alerts) == 0

    def test_alert_at_threshold(self) -> None:
        self.tracker.record_usage(tokens_used=80000)  # exactly 80%
        alerts = self.tracker.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "high_utilization"

    def test_alert_above_threshold(self) -> None:
        self.tracker.record_usage(tokens_used=90000)
        alerts = self.tracker.get_alerts()
        assert len(alerts) == 1

    def test_full_alert_type_at_100_percent(self) -> None:
        self.tracker.record_usage(tokens_used=100000)
        alerts = self.tracker.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "full"

    def test_multiple_alerts_tracked(self) -> None:
        self.tracker.record_usage(tokens_used=85000)
        self.tracker.record_usage(tokens_used=90000)
        alerts = self.tracker.get_alerts()
        assert len(alerts) == 2

    def test_get_history_returns_most_recent_first(self) -> None:
        for i in range(5):
            self.tracker.record_usage(tokens_used=i * 10000)
        history = self.tracker.get_history()
        assert len(history) == 5
        # Most recent first
        assert history[0].tokens_used == 40000

    def test_get_history_with_limit(self) -> None:
        for i in range(5):
            self.tracker.record_usage(tokens_used=i * 10000)
        history = self.tracker.get_history(limit=2)
        assert len(history) == 2

    def test_current_snapshot_none_when_empty(self) -> None:
        snapshot = self.tracker.current_snapshot()
        assert snapshot is None

    def test_current_snapshot_returns_snapshot(self) -> None:
        self.tracker.record_usage(tokens_used=50000)
        snapshot = self.tracker.current_snapshot()
        assert isinstance(snapshot, ContextSnapshot)
        assert snapshot.agent_id == "agent-1"
        assert snapshot.current_tokens_used == 50000
        assert snapshot.utilization_ratio == pytest.approx(0.5)

    def test_snapshot_is_high_utilization_true(self) -> None:
        self.tracker.record_usage(tokens_used=85000)
        snapshot = self.tracker.current_snapshot()
        assert snapshot is not None
        assert snapshot.is_high_utilization is True

    def test_snapshot_is_high_utilization_false(self) -> None:
        self.tracker.record_usage(tokens_used=50000)
        snapshot = self.tracker.current_snapshot()
        assert snapshot is not None
        assert snapshot.is_high_utilization is False

    def test_snapshot_peak_utilization(self) -> None:
        self.tracker.record_usage(tokens_used=30000)
        self.tracker.record_usage(tokens_used=90000)
        self.tracker.record_usage(tokens_used=40000)
        snapshot = self.tracker.current_snapshot()
        assert snapshot is not None
        assert snapshot.peak_utilization_ratio == pytest.approx(0.9)

    def test_snapshot_mean_utilization(self) -> None:
        self.tracker.record_usage(tokens_used=20000)
        self.tracker.record_usage(tokens_used=60000)
        snapshot = self.tracker.current_snapshot()
        assert snapshot is not None
        assert snapshot.mean_utilization_ratio == pytest.approx(0.4)

    def test_snapshot_to_dict_structure(self) -> None:
        self.tracker.record_usage(tokens_used=50000)
        snapshot = self.tracker.current_snapshot()
        assert snapshot is not None
        d = snapshot.to_dict()
        assert "utilization_ratio" in d
        assert "is_high_utilization" in d
        assert "alert_count" in d

    def test_max_history_evicts_oldest(self) -> None:
        tracker = UtilizationTracker(agent_id="a", max_tokens=1000, max_history=3)
        for i in range(5):
            tracker.record_usage(tokens_used=i * 100)
        assert tracker.measurement_count == 3

    def test_clear_resets_state(self) -> None:
        self.tracker.record_usage(tokens_used=85000)
        self.tracker.clear()
        assert self.tracker.measurement_count == 0
        assert len(self.tracker.get_alerts()) == 0

    def test_turn_number_stored(self) -> None:
        record = self.tracker.record_usage(tokens_used=10000, turn_number=5)
        assert record.turn_number == 5
