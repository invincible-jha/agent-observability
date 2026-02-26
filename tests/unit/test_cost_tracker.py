"""Unit tests for cost.tracker — CostTracker, CostRecord, CostSummary."""
from __future__ import annotations

import csv
import io
import threading
import time

import pytest

from agent_observability.cost.tracker import CostRecord, CostSummary, CostTracker


class TestCostTrackerRecord:
    def test_record_returns_cost_record(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("openai", "gpt-4o", 1000, 500)
        assert isinstance(rec, CostRecord)

    def test_record_sets_provider_and_model(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("anthropic", "claude-3-5-sonnet-20241022", 200, 100)
        assert rec.provider == "anthropic"
        assert rec.model == "claude-3-5-sonnet-20241022"

    def test_record_computes_cost_from_pricing_table(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("openai", "gpt-4o", 1_000_000, 0)
        # gpt-4o input is $2.50/M
        assert rec.cost_usd == pytest.approx(2.50, rel=1e-4)

    def test_record_accepts_explicit_cost_override(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("openai", "gpt-4o", 1000, 500, cost_usd=9.99)
        assert rec.cost_usd == 9.99

    def test_record_uses_tracker_agent_id_as_default(self) -> None:
        tracker = CostTracker(agent_id="default-agent")
        rec = tracker.record("openai", "gpt-4o", 100, 50)
        assert rec.agent_id == "default-agent"

    def test_record_override_agent_id_per_call(self) -> None:
        tracker = CostTracker(agent_id="default-agent")
        rec = tracker.record("openai", "gpt-4o", 100, 50, agent_id="override-agent")
        assert rec.agent_id == "override-agent"

    def test_record_stores_operation_label(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("openai", "gpt-4o", 100, 50, operation="embedding")
        assert rec.operation == "embedding"

    def test_record_stores_tags(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("openai", "gpt-4o", 100, 50, tags={"env": "prod"})
        assert rec.tags["env"] == "prod"

    def test_multiple_records_accumulate(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50)
        tracker.record("openai", "gpt-4o", 200, 80)
        assert len(tracker) == 2

    def test_reset_clears_all_records(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50)
        tracker.reset()
        assert len(tracker) == 0


class TestCostTrackerRecords:
    def test_records_returns_all_without_filter(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50, agent_id="a1")
        tracker.record("anthropic", "claude-3-5-sonnet-20241022", 200, 80, agent_id="a2")
        assert len(tracker.records()) == 2

    def test_records_filtered_by_agent_id(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50, agent_id="agent-1")
        tracker.record("openai", "gpt-4o", 100, 50, agent_id="agent-2")
        assert len(tracker.records(agent_id="agent-1")) == 1

    def test_records_filtered_by_provider(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50)
        tracker.record("anthropic", "claude-3-5-sonnet-20241022", 100, 50)
        assert len(tracker.records(provider="openai")) == 1

    def test_records_filtered_by_since_timestamp(self) -> None:
        tracker = CostTracker()
        early_ts = time.time() - 3600
        recent_ts = time.time()
        # Inject records manually to control timestamp
        from agent_observability.cost.tracker import CostRecord as CR
        tracker._records.append(CR(
            timestamp=early_ts, agent_id="", session_id="", task_id="",
            provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50,
            cached_input_tokens=0, cost_usd=0.01, operation="llm_call"
        ))
        tracker._records.append(CR(
            timestamp=recent_ts, agent_id="", session_id="", task_id="",
            provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50,
            cached_input_tokens=0, cost_usd=0.01, operation="llm_call"
        ))
        results = tracker.records(since=recent_ts - 60)
        assert len(results) == 1


class TestCostTrackerSummary:
    def test_empty_tracker_returns_zero_summary(self) -> None:
        tracker = CostTracker()
        summary = tracker.summary()
        assert summary.total_cost_usd == 0.0
        assert summary.record_count == 0
        assert summary.total_tokens == 0

    def test_summary_sums_total_cost(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 1000, 500, cost_usd=1.0)
        tracker.record("openai", "gpt-4o", 1000, 500, cost_usd=2.0)
        summary = tracker.summary()
        assert summary.total_cost_usd == pytest.approx(3.0)

    def test_summary_groups_by_model(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50, cost_usd=1.0)
        tracker.record("anthropic", "claude-3-5-sonnet-20241022", 100, 50, cost_usd=0.5)
        summary = tracker.summary()
        assert "gpt-4o" in summary.by_model
        assert "claude-3-5-sonnet-20241022" in summary.by_model

    def test_summary_groups_by_provider(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50, cost_usd=1.0)
        tracker.record("anthropic", "claude-3-5-sonnet-20241022", 100, 50, cost_usd=0.5)
        summary = tracker.summary()
        assert "openai" in summary.by_provider
        assert "anthropic" in summary.by_provider

    def test_summary_total_tokens_correct(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 1000, 500, cost_usd=1.0)
        summary = tracker.summary()
        assert summary.total_input_tokens == 1000
        assert summary.total_output_tokens == 500
        assert summary.total_tokens == 1500


class TestCostTrackerExportCsv:
    def test_export_csv_returns_string(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50, cost_usd=0.001)
        csv_output = tracker.export_csv()
        assert isinstance(csv_output, str)

    def test_export_csv_has_header_row(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 100, 50, cost_usd=0.001)
        reader = csv.DictReader(io.StringIO(tracker.export_csv()))
        assert reader.fieldnames is not None
        assert "provider" in reader.fieldnames
        assert "cost_usd" in reader.fieldnames

    def test_export_csv_empty_tracker_has_only_header(self) -> None:
        tracker = CostTracker()
        rows = list(csv.DictReader(io.StringIO(tracker.export_csv())))
        assert rows == []


class TestCostTrackerThreadSafety:
    def test_concurrent_records_all_stored(self) -> None:
        tracker = CostTracker()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    tracker.record("openai", "gpt-4o", 100, 50, cost_usd=0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(tracker) == 500

    def test_iteration_is_safe_under_concurrent_writes(self) -> None:
        tracker = CostTracker()
        for _ in range(20):
            tracker.record("openai", "gpt-4o", 100, 50, cost_usd=0.001)

        count = sum(1 for _ in tracker)
        assert count == 20
