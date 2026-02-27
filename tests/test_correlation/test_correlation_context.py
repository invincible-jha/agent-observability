"""Tests for CorrelationContext."""
from __future__ import annotations

import pytest

from agent_observability.correlation.correlation_context import (
    BaggageItem,
    CorrelationContext,
)


class TestBaggageItem:
    def test_basic_construction(self) -> None:
        item = BaggageItem(key="tenant", value="acme-corp")
        assert item.key == "tenant"
        assert item.value == "acme-corp"

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="key"):
            BaggageItem(key="", value="value")

    def test_is_frozen(self) -> None:
        item = BaggageItem(key="tenant", value="acme")
        with pytest.raises(Exception):
            item.key = "other"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        item = BaggageItem(key="user", value="alice")
        d = item.to_dict()
        assert d == {"key": "user", "value": "alice"}


class TestCorrelationContext:
    def test_new_root_generates_ids(self) -> None:
        ctx = CorrelationContext.new_root()
        assert ctx.trace_id != ""
        assert ctx.span_id != ""
        assert ctx.parent_span_id is None

    def test_new_root_is_root(self) -> None:
        ctx = CorrelationContext.new_root()
        assert ctx.is_root is True

    def test_new_root_with_custom_trace_id(self) -> None:
        ctx = CorrelationContext.new_root(trace_id="custom-trace")
        assert ctx.trace_id == "custom-trace"

    def test_new_child_span_inherits_trace_id(self) -> None:
        parent = CorrelationContext.new_root()
        child = parent.new_child_span()
        assert child.trace_id == parent.trace_id

    def test_new_child_span_sets_parent(self) -> None:
        parent = CorrelationContext.new_root()
        child = parent.new_child_span()
        assert child.parent_span_id == parent.span_id

    def test_new_child_span_is_not_root(self) -> None:
        parent = CorrelationContext.new_root()
        child = parent.new_child_span()
        assert child.is_root is False

    def test_new_child_span_custom_span_id(self) -> None:
        parent = CorrelationContext.new_root()
        child = parent.new_child_span(span_id="custom-child")
        assert child.span_id == "custom-child"

    def test_different_roots_have_different_trace_ids(self) -> None:
        ctx1 = CorrelationContext.new_root()
        ctx2 = CorrelationContext.new_root()
        assert ctx1.trace_id != ctx2.trace_id

    def test_get_baggage_returns_value(self) -> None:
        ctx = CorrelationContext.new_root()
        ctx_with_baggage = ctx.with_baggage("tenant", "acme")
        assert ctx_with_baggage.get_baggage("tenant") == "acme"

    def test_get_baggage_returns_none_for_missing(self) -> None:
        ctx = CorrelationContext.new_root()
        assert ctx.get_baggage("nonexistent") is None

    def test_with_baggage_returns_new_context(self) -> None:
        ctx = CorrelationContext.new_root()
        ctx2 = ctx.with_baggage("key", "value")
        assert ctx2 is not ctx
        assert ctx.get_baggage("key") is None
        assert ctx2.get_baggage("key") == "value"

    def test_with_baggage_replaces_existing_key(self) -> None:
        ctx = CorrelationContext.new_root()
        ctx = ctx.with_baggage("key", "old")
        ctx = ctx.with_baggage("key", "new")
        assert ctx.get_baggage("key") == "new"
        assert len(ctx.baggage) == 1

    def test_baggage_propagated_to_child(self) -> None:
        ctx = CorrelationContext.new_root()
        ctx = ctx.with_baggage("tenant", "corp")
        child = ctx.new_child_span()
        assert child.get_baggage("tenant") == "corp"

    def test_to_dict_structure(self) -> None:
        ctx = CorrelationContext.new_root()
        d = ctx.to_dict()
        assert "trace_id" in d
        assert "span_id" in d
        assert "parent_span_id" in d
        assert "sampled" in d
        assert "baggage" in d

    def test_from_dict_round_trip(self) -> None:
        ctx = CorrelationContext.new_root()
        ctx = ctx.with_baggage("tenant", "acme")
        d = ctx.to_dict()
        reconstructed = CorrelationContext.from_dict(d)
        assert reconstructed.trace_id == ctx.trace_id
        assert reconstructed.span_id == ctx.span_id
        assert reconstructed.get_baggage("tenant") == "acme"

    def test_from_dict_raises_on_missing_trace_id(self) -> None:
        with pytest.raises(ValueError):
            CorrelationContext.from_dict({"span_id": "s1"})

    def test_from_dict_raises_on_missing_span_id(self) -> None:
        with pytest.raises(ValueError):
            CorrelationContext.from_dict({"trace_id": "t1"})

    def test_to_headers_produces_traceparent(self) -> None:
        ctx = CorrelationContext.new_root()
        headers = ctx.to_headers()
        assert "traceparent" in headers
        assert headers["traceparent"].startswith("00-")

    def test_to_headers_includes_baggage(self) -> None:
        ctx = CorrelationContext.new_root()
        ctx = ctx.with_baggage("tenant", "acme")
        headers = ctx.to_headers()
        assert "baggage" in headers
        assert "tenant=acme" in headers["baggage"]

    def test_sampled_false_propagated_to_child(self) -> None:
        ctx = CorrelationContext.new_root(sampled=False)
        child = ctx.new_child_span()
        assert child.sampled is False
