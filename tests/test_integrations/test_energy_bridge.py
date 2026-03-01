"""Tests for agent_observability.integrations.energy_bridge.

All tests use hand-rolled fakes that satisfy EnergyTrackerProtocol via
structural typing — agent-energy-budget is not required in CI.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from agent_observability.integrations.energy_bridge import (
    BudgetStatus,
    CostAttribution,
    EnergyBudgetBridge,
    EnergyTrackerProtocol,
)


# ---------------------------------------------------------------------------
# Minimal fakes satisfying EnergyTrackerProtocol
# ---------------------------------------------------------------------------


class _FakeTracker:
    """Fake energy tracker that records all calls without importing energy-budget."""

    def __init__(
        self,
        budget_id: str = "agent-test",
        allocated_usd: float = 1.0,
        initial_consumed: float = 0.0,
    ) -> None:
        self._budget_id = budget_id
        self._allocated = allocated_usd
        self._consumed = initial_consumed
        self.usage_calls: list[tuple[str, int, float]] = []

    def record_usage(self, model: str, tokens: int, cost: float) -> None:
        self.usage_calls.append((model, tokens, cost))
        self._consumed += cost

    def check_budget(self) -> BudgetStatus:
        consumed = round(self._consumed, 8)
        remaining = round(self._allocated - consumed, 8)
        pct = round((consumed / self._allocated) * 100.0, 4) if self._allocated > 0 else 0.0
        return BudgetStatus(
            budget_id=self._budget_id,
            allocated_usd=self._allocated,
            consumed_usd=consumed,
            remaining_usd=remaining,
            usage_percentage=pct,
            is_exceeded=consumed > self._allocated,
        )


class _BrokenTracker:
    """Tracker that raises on every call."""

    def record_usage(self, model: str, tokens: int, cost: float) -> None:
        raise RuntimeError("record_usage broken")

    def check_budget(self) -> BudgetStatus:
        raise RuntimeError("check_budget broken")


def _make_llm_span(
    model: str = "gpt-4o",
    input_tokens: int = 500,
    output_tokens: int = 200,
    cost_usd: float = 0.003,
) -> MagicMock:
    """Build a minimal span-like mock with LLM cost attributes."""
    span = MagicMock()
    inner = MagicMock()
    inner._attributes = {
        "agent.span.kind": "llm_call",
        "llm.model": model,
        "llm.tokens.input": input_tokens,
        "llm.tokens.output": output_tokens,
        "llm.cost.usd": cost_usd,
    }
    span._span = inner
    return span


def _make_non_llm_span(span_kind: str = "tool_invoke") -> MagicMock:
    """Build a minimal span-like mock for a non-LLM span kind."""
    span = MagicMock()
    inner = MagicMock()
    inner._attributes = {
        "agent.span.kind": span_kind,
    }
    span._span = inner
    return span


def _make_empty_span() -> MagicMock:
    """Span with no recognisable attribute structure."""
    span = MagicMock(spec=[])  # no _span or _attributes attributes
    return span


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_fake_tracker_satisfies_protocol(self) -> None:
        tracker = _FakeTracker()
        assert isinstance(tracker, EnergyTrackerProtocol)

    def test_none_is_not_protocol(self) -> None:
        assert not isinstance(None, EnergyTrackerProtocol)

    def test_object_without_methods_is_not_protocol(self) -> None:
        assert not isinstance(object(), EnergyTrackerProtocol)


# ---------------------------------------------------------------------------
# BudgetStatus dataclass
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_construction(self) -> None:
        status = BudgetStatus(
            budget_id="agent-1",
            allocated_usd=1.0,
            consumed_usd=0.5,
            remaining_usd=0.5,
            usage_percentage=50.0,
            is_exceeded=False,
        )
        assert status.budget_id == "agent-1"
        assert status.allocated_usd == 1.0
        assert status.remaining_usd == 0.5
        assert status.is_exceeded is False

    def test_exceeded_flag_set_correctly(self) -> None:
        status = BudgetStatus(
            budget_id="agent-2",
            allocated_usd=1.0,
            consumed_usd=1.5,
            remaining_usd=-0.5,
            usage_percentage=150.0,
            is_exceeded=True,
        )
        assert status.is_exceeded is True


# ---------------------------------------------------------------------------
# CostAttribution dataclass
# ---------------------------------------------------------------------------


class TestCostAttribution:
    def test_construction(self) -> None:
        attr = CostAttribution(
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
            estimated_cost_usd=0.003,
        )
        assert attr.model == "gpt-4o"
        assert attr.input_tokens == 500
        assert attr.output_tokens == 200
        assert attr.estimated_cost_usd == 0.003

    def test_timestamp_defaults_to_utc(self) -> None:
        from datetime import timezone  # noqa: PLC0415

        attr = CostAttribution(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            estimated_cost_usd=0.001,
        )
        assert attr.timestamp.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# EnergyBudgetBridge — construction
# ---------------------------------------------------------------------------


class TestEnergyBudgetBridgeConstruction:
    def test_default_no_tracker(self) -> None:
        bridge = EnergyBudgetBridge()
        assert bridge._tracker is None

    def test_tracker_stored(self) -> None:
        tracker = _FakeTracker()
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        assert bridge._tracker is tracker


# ---------------------------------------------------------------------------
# on_cost_event — with tracker
# ---------------------------------------------------------------------------


class TestOnCostEventWithTracker:
    def test_returns_budget_status(self) -> None:
        tracker = _FakeTracker()
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        status = bridge.on_cost_event(model="gpt-4o", tokens=1000, cost=0.002)
        assert isinstance(status, BudgetStatus)

    def test_forwards_cost_to_tracker(self) -> None:
        tracker = _FakeTracker()
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        bridge.on_cost_event(model="gpt-4o", tokens=500, cost=0.001)
        assert len(tracker.usage_calls) == 1
        model, tokens, cost = tracker.usage_calls[0]
        assert model == "gpt-4o"
        assert tokens == 500
        assert cost == 0.001

    def test_status_reflects_consumption(self) -> None:
        tracker = _FakeTracker(allocated_usd=1.0)
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        status = bridge.on_cost_event(model="gpt-4o", tokens=100, cost=0.5)
        assert status is not None
        assert status.consumed_usd == pytest.approx(0.5)

    def test_exceeded_flag_true_when_over_budget(self) -> None:
        tracker = _FakeTracker(allocated_usd=0.001)
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        status = bridge.on_cost_event(model="gpt-4o", tokens=1000, cost=0.5)
        assert status is not None
        assert status.is_exceeded is True

    def test_multiple_events_accumulate_consumption(self) -> None:
        tracker = _FakeTracker(allocated_usd=1.0)
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        bridge.on_cost_event(model="gpt-4o", tokens=100, cost=0.1)
        bridge.on_cost_event(model="gpt-4o", tokens=200, cost=0.2)
        status = bridge.on_cost_event(model="gpt-4o", tokens=50, cost=0.05)
        assert status is not None
        assert status.consumed_usd == pytest.approx(0.35)

    def test_broken_tracker_returns_none(self) -> None:
        bridge = EnergyBudgetBridge(energy_tracker=_BrokenTracker())  # type: ignore[arg-type]
        result = bridge.on_cost_event(model="gpt-4o", tokens=100, cost=0.001)
        assert result is None


# ---------------------------------------------------------------------------
# on_cost_event — no tracker (no-op)
# ---------------------------------------------------------------------------


class TestOnCostEventNoTracker:
    def test_returns_none_when_no_tracker(self) -> None:
        bridge = EnergyBudgetBridge()
        result = bridge.on_cost_event(model="gpt-4o", tokens=100, cost=0.001)
        assert result is None

    def test_no_tracker_does_not_raise(self) -> None:
        bridge = EnergyBudgetBridge()
        bridge.on_cost_event(model="gpt-4o", tokens=100, cost=0.001)  # must not raise

    def test_no_tracker_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge = EnergyBudgetBridge()
        with caplog.at_level(
            logging.DEBUG, logger="agent_observability.integrations.energy_bridge"
        ):
            bridge.on_cost_event(model="gpt-4o", tokens=100, cost=0.001)
        assert any("no tracker" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# on_span_end — LLM call span
# ---------------------------------------------------------------------------


class TestOnSpanEndLlmCall:
    def test_extracts_attribution_from_llm_span(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_llm_span(model="gpt-4o", input_tokens=500, output_tokens=200, cost_usd=0.003)
        attribution = bridge.on_span_end(span)
        assert isinstance(attribution, CostAttribution)

    def test_attribution_has_correct_model(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_llm_span(model="claude-haiku-4", input_tokens=100, output_tokens=50, cost_usd=0.0005)
        attribution = bridge.on_span_end(span)
        assert attribution is not None
        assert attribution.model == "claude-haiku-4"

    def test_attribution_has_correct_token_counts(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_llm_span(input_tokens=700, output_tokens=300)
        attribution = bridge.on_span_end(span)
        assert attribution is not None
        assert attribution.input_tokens == 700
        assert attribution.output_tokens == 300

    def test_attribution_has_correct_cost(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_llm_span(cost_usd=0.00567)
        attribution = bridge.on_span_end(span)
        assert attribution is not None
        assert attribution.estimated_cost_usd == pytest.approx(0.00567)

    def test_forwards_to_tracker_when_configured(self) -> None:
        tracker = _FakeTracker()
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        span = _make_llm_span(input_tokens=600, output_tokens=400, cost_usd=0.004)
        bridge.on_span_end(span)
        assert len(tracker.usage_calls) == 1
        model, tokens, cost = tracker.usage_calls[0]
        assert tokens == 1000  # 600 + 400


# ---------------------------------------------------------------------------
# on_span_end — non-LLM spans (skipped)
# ---------------------------------------------------------------------------


class TestOnSpanEndNonLlmSpan:
    def test_tool_invoke_span_returns_none(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_non_llm_span(span_kind="tool_invoke")
        result = bridge.on_span_end(span)
        assert result is None

    def test_reasoning_step_span_returns_none(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_non_llm_span(span_kind="reasoning_step")
        result = bridge.on_span_end(span)
        assert result is None

    def test_empty_span_returns_none(self) -> None:
        bridge = EnergyBudgetBridge()
        span = _make_empty_span()
        result = bridge.on_span_end(span)
        assert result is None

    def test_non_llm_span_does_not_forward_to_tracker(self) -> None:
        tracker = _FakeTracker()
        bridge = EnergyBudgetBridge(energy_tracker=tracker)
        span = _make_non_llm_span(span_kind="tool_invoke")
        bridge.on_span_end(span)
        assert len(tracker.usage_calls) == 0


# ---------------------------------------------------------------------------
# on_span_end — span with direct _attributes (no inner _span)
# ---------------------------------------------------------------------------


class TestOnSpanEndDirectAttributes:
    def test_span_with_direct_attributes_extracted(self) -> None:
        span = MagicMock(spec=["_attributes"])
        span._attributes = {
            "agent.span.kind": "llm_call",
            "llm.model": "gpt-4o-mini",
            "llm.tokens.input": 100,
            "llm.tokens.output": 50,
            "llm.cost.usd": 0.0002,
        }
        bridge = EnergyBudgetBridge()
        attribution = bridge.on_span_end(span)
        assert attribution is not None
        assert attribution.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# on_span_end — span with no cost data (skipped)
# ---------------------------------------------------------------------------


class TestOnSpanEndNoCostData:
    def test_span_without_model_or_cost_returns_none(self) -> None:
        span = MagicMock()
        inner = MagicMock()
        inner._attributes = {
            "agent.span.kind": "llm_call",
            "llm.model": "",
            "llm.cost.usd": 0.0,
        }
        span._span = inner
        bridge = EnergyBudgetBridge()
        result = bridge.on_span_end(span)
        assert result is None
