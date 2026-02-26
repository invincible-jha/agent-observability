"""Comprehensive tests for the cost subsystem.

Covers:
- cost.aggregator  (TimePeriod, _period_start, AggregatedCosts, CostAggregator)
- cost.annotator   (CostAnnotator)
- cost.attribution (CostAttributor, CostRecord, CostSummary, PROVIDER_PRICING)
- cost.reporter    (CostReporter)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.cost.aggregator import (
    AggregatedCosts,
    CostAggregator,
    TimePeriod,
    _period_start,
)
from agent_observability.cost.annotator import CostAnnotator
from agent_observability.cost.attribution import (
    PROVIDER_PRICING,
    CostAttributor,
    CostRecord,
    CostSummary,
)
from agent_observability.cost.reporter import CostReporter
from agent_observability.cost.tracker import CostTracker


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_tracker_with_records() -> CostTracker:
    tracker = CostTracker()
    tracker.record(
        provider="anthropic",
        model="claude-sonnet-4",
        input_tokens=1000,
        output_tokens=200,
        agent_id="agent-a",
        task_id="task-1",
    )
    tracker.record(
        provider="openai",
        model="gpt-4o",
        input_tokens=500,
        output_tokens=100,
        agent_id="agent-b",
        task_id="task-2",
    )
    return tracker


# ── TimePeriod ─────────────────────────────────────────────────────────────────


class TestTimePeriod:
    def test_today_period_start_is_today(self) -> None:
        import datetime

        start = _period_start(TimePeriod.TODAY)
        assert start is not None
        today_midnight = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        assert abs(start - today_midnight) < 1.0

    def test_week_period_start(self) -> None:
        start = _period_start(TimePeriod.WEEK)
        assert start is not None
        expected = time.time() - 7 * 86_400
        assert abs(start - expected) < 1.0

    def test_month_period_start(self) -> None:
        start = _period_start(TimePeriod.MONTH)
        assert start is not None
        expected = time.time() - 30 * 86_400
        assert abs(start - expected) < 1.0

    def test_all_period_returns_none(self) -> None:
        assert _period_start(TimePeriod.ALL) is None

    def test_enum_values(self) -> None:
        assert TimePeriod.TODAY.value == "today"
        assert TimePeriod.WEEK.value == "week"
        assert TimePeriod.MONTH.value == "month"
        assert TimePeriod.ALL.value == "all"


# ── CostAggregator ─────────────────────────────────────────────────────────────


class TestCostAggregator:
    def test_aggregate_all_records(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        result = agg.aggregate(period=TimePeriod.ALL)
        assert result.record_count == 2
        assert result.total_usd >= 0.0
        assert "agent-a" in result.by_agent
        assert "agent-b" in result.by_agent

    def test_aggregate_by_model(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        result = agg.aggregate(period=TimePeriod.ALL)
        assert "claude-sonnet-4" in result.by_model
        assert "gpt-4o" in result.by_model

    def test_aggregate_by_provider(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        result = agg.aggregate(period=TimePeriod.ALL)
        assert "anthropic" in result.by_provider
        assert "openai" in result.by_provider

    def test_aggregate_by_task(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        result = agg.aggregate(period=TimePeriod.ALL)
        assert "task-1" in result.by_task
        assert "task-2" in result.by_task

    def test_aggregate_with_agent_filter(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        result = agg.aggregate(period=TimePeriod.ALL, agent_id="agent-a")
        assert result.record_count == 1
        assert "agent-a" in result.by_agent
        assert "agent-b" not in result.by_agent

    def test_aggregate_no_task_id(self) -> None:
        tracker = CostTracker()
        tracker.record(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=20,
            agent_id="agent-c",
        )
        agg = CostAggregator(tracker)
        result = agg.aggregate(period=TimePeriod.ALL)
        assert "(no task)" in result.by_task

    def test_top_agents(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        top = agg.top_agents(n=10)
        assert isinstance(top, list)
        if len(top) >= 2:
            assert top[0][1] >= top[1][1]

    def test_top_models(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        top = agg.top_models(n=5)
        assert isinstance(top, list)

    def test_daily_cost_series_length(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        series = agg.daily_cost_series(days=7)
        assert len(series) == 7

    def test_daily_cost_series_format(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        series = agg.daily_cost_series(days=3)
        for date_str, cost in series:
            assert len(date_str) == 10  # YYYY-MM-DD
            assert isinstance(cost, float)

    def test_empty_tracker_zero_costs(self) -> None:
        tracker = CostTracker()
        agg = CostAggregator(tracker)
        result = agg.aggregate()
        assert result.total_usd == 0.0
        assert result.record_count == 0


# ── CostAnnotator ──────────────────────────────────────────────────────────────


class TestCostAnnotator:
    def test_annotate_returns_cost(self) -> None:
        annotator = CostAnnotator(provider="anthropic", default_model="claude-sonnet-4")
        cost = annotator.annotate(input_tokens=1000, output_tokens=200, model="claude-sonnet-4")
        assert cost > 0.0

    def test_annotate_anthropic_response(self) -> None:
        usage = SimpleNamespace(input_tokens=500, output_tokens=100)
        response = SimpleNamespace(model="claude-sonnet-4", usage=usage)
        annotator = CostAnnotator()
        cost = annotator.annotate_anthropic(response)
        assert cost >= 0.0

    def test_annotate_anthropic_no_usage(self) -> None:
        response = SimpleNamespace(model="claude-sonnet-4", usage=None)
        annotator = CostAnnotator()
        cost = annotator.annotate_anthropic(response)
        assert cost >= 0.0

    def test_annotate_openai_response(self) -> None:
        usage = SimpleNamespace(prompt_tokens=300, completion_tokens=80)
        response = SimpleNamespace(model="gpt-4o", usage=usage)
        annotator = CostAnnotator()
        cost = annotator.annotate_openai(response)
        assert cost >= 0.0

    def test_annotate_openai_no_usage(self) -> None:
        response = SimpleNamespace(model="gpt-4o", usage=None)
        annotator = CostAnnotator()
        cost = annotator.annotate_openai(response)
        assert cost >= 0.0

    def test_annotate_with_tracker(self) -> None:
        tracker = MagicMock()
        tracker.record = MagicMock()
        annotator = CostAnnotator(tracker=tracker)
        annotator.annotate(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
            provider="openai",
        )
        tracker.record.assert_called_once()

    def test_annotate_uses_defaults(self) -> None:
        annotator = CostAnnotator(provider="openai", default_model="gpt-4o")
        # No model/provider override → should use defaults
        cost = annotator.annotate(input_tokens=100, output_tokens=50)
        assert cost >= 0.0

    def test_annotate_with_cached_input_tokens(self) -> None:
        annotator = CostAnnotator()
        cost = annotator.annotate(
            input_tokens=1000,
            output_tokens=200,
            model="claude-sonnet-4",
            cached_input_tokens=500,
        )
        assert isinstance(cost, float)

    def test_stamp_with_active_span(self) -> None:
        """Test that span attributes are set when a recording span is active."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        annotator = CostAnnotator()

        with patch("agent_observability.cost.annotator._OTEL_AVAILABLE", True):
            with patch("agent_observability.cost.annotator.otel_trace") as mock_trace:
                mock_trace.get_current_span.return_value = mock_span
                annotator.annotate(
                    input_tokens=100,
                    output_tokens=50,
                    model="gpt-4o",
                    provider="openai",
                )
                mock_span.set_attribute.assert_called()

    def test_stamp_with_no_otel(self) -> None:
        annotator = CostAnnotator()
        with patch("agent_observability.cost.annotator._OTEL_AVAILABLE", False):
            cost = annotator.annotate(
                input_tokens=100, output_tokens=50, model="gpt-4o", provider="openai"
            )
            assert cost >= 0.0

    def test_tracker_exception_does_not_propagate(self) -> None:
        tracker = MagicMock()
        tracker.record.side_effect = RuntimeError("db error")
        annotator = CostAnnotator(tracker=tracker)
        # Should not raise
        cost = annotator.annotate(input_tokens=100, output_tokens=50)
        assert isinstance(cost, float)


# ── CostAttributor ─────────────────────────────────────────────────────────────


class TestCostAttributor:
    def test_calculate_cost_known_model(self) -> None:
        attributor = CostAttributor()
        cost = attributor.calculate_cost("gpt-4o", 1000, 200)
        # (1000/1000 * 0.0025) + (200/1000 * 0.010) = 0.0025 + 0.002 = 0.0045
        assert cost == pytest.approx(0.0045, abs=1e-7)

    def test_calculate_cost_unknown_model_returns_zero(self) -> None:
        attributor = CostAttributor()
        cost = attributor.calculate_cost("unknown-model-xyz", 1000, 200)
        assert cost == 0.0

    def test_prefix_match_fallback(self) -> None:
        attributor = CostAttributor()
        # "gpt-4o-2024-11-20" should prefix-match "gpt-4o"
        cost = attributor.calculate_cost("gpt-4o-2024-11-20", 1000, 200)
        assert cost > 0.0

    def test_record_creates_cost_record(self) -> None:
        attributor = CostAttributor()
        record = attributor.record("span-001", "gpt-4o", 1000, 200)
        assert isinstance(record, CostRecord)
        assert record.span_id == "span-001"
        assert record.model == "gpt-4o"
        assert record.input_tokens == 1000
        assert record.output_tokens == 200
        assert record.cost_usd >= 0.0

    def test_summary_empty(self) -> None:
        attributor = CostAttributor()
        summary = attributor.summary()
        assert summary.total_cost == 0.0
        assert summary.record_count == 0
        assert summary.by_model == {}

    def test_summary_accumulates_costs(self) -> None:
        attributor = CostAttributor()
        attributor.record("span-1", "gpt-4o", 1000, 200)
        attributor.record("span-2", "gpt-4o", 500, 100)
        summary = attributor.summary()
        assert summary.record_count == 2
        assert summary.total_cost > 0.0
        assert "gpt-4o" in summary.by_model

    def test_summary_by_model_breakdown(self) -> None:
        attributor = CostAttributor()
        attributor.record("span-1", "gpt-4o", 1000, 200)
        attributor.record("span-2", "claude-sonnet-4", 500, 100)
        summary = attributor.summary()
        assert "gpt-4o" in summary.by_model
        assert "claude-sonnet-4" in summary.by_model

    def test_reset_clears_records(self) -> None:
        attributor = CostAttributor()
        attributor.record("span-1", "gpt-4o", 100, 50)
        attributor.reset()
        summary = attributor.summary()
        assert summary.record_count == 0

    def test_provider_pricing_not_empty(self) -> None:
        assert len(PROVIDER_PRICING) > 0

    def test_thread_safety(self) -> None:
        import threading

        attributor = CostAttributor()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(20):
                    attributor.record(f"span-{i}", "gpt-4o", 100, 50)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        summary = attributor.summary()
        assert summary.record_count == 100


# ── CostReporter ───────────────────────────────────────────────────────────────


class TestCostReporter:
    def test_to_csv_returns_string(self) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        csv_output = reporter.to_csv()
        assert isinstance(csv_output, str)

    def test_to_json_valid(self) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        json_output = reporter.to_json()
        data = json.loads(json_output)
        assert "total_usd" in data
        assert "by_agent" in data
        assert "daily_series" in data

    def test_to_json_today_period(self) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        json_output = reporter.to_json(period=TimePeriod.TODAY)
        data = json.loads(json_output)
        assert data["period"] == "today"
        assert len(data["daily_series"]) == 1

    def test_to_markdown_contains_sections(self) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        md = reporter.to_markdown()
        assert "# Agent Cost Report" in md
        assert "## By Model" in md
        assert "## By Provider" in md
        assert "## By Agent" in md

    def test_write_json(self, tmp_path: Path) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        output_file = str(tmp_path / "report.json")
        reporter.write(output_file, fmt="json")
        assert Path(output_file).exists()
        data = json.loads(Path(output_file).read_text())
        assert "total_usd" in data

    def test_write_csv(self, tmp_path: Path) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        output_file = str(tmp_path / "report.csv")
        reporter.write(output_file, fmt="csv")
        assert Path(output_file).exists()

    def test_write_markdown(self, tmp_path: Path) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        output_file = str(tmp_path / "report.md")
        reporter.write(output_file, fmt="md")
        assert Path(output_file).exists()

    def test_write_markdown_alias(self, tmp_path: Path) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        output_file = str(tmp_path / "report2.md")
        reporter.write(output_file, fmt="markdown")
        assert Path(output_file).exists()

    def test_write_unknown_format_raises(self, tmp_path: Path) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        with pytest.raises(ValueError, match="Unknown report format"):
            reporter.write(str(tmp_path / "report.xyz"), fmt="xyz")

    def test_custom_aggregator(self) -> None:
        tracker = _make_tracker_with_records()
        agg = CostAggregator(tracker)
        reporter = CostReporter(tracker, aggregator=agg)
        json_output = reporter.to_json()
        data = json.loads(json_output)
        assert "total_usd" in data

    def test_to_csv_with_agent_filter(self) -> None:
        tracker = _make_tracker_with_records()
        reporter = CostReporter(tracker)
        csv_output = reporter.to_csv(agent_id="agent-a")
        assert isinstance(csv_output, str)
