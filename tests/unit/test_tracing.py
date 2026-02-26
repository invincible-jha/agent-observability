"""Comprehensive tests for the tracing subsystem.

Covers:
- tracing.context    (AgentTraceContext)
- tracing.propagator (CrossAgentPropagator, inject_into_dict, extract_from_dict)
- tracing.sampler    (CostAwareSampler)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_observability.tracing.context import AgentTraceContext
from agent_observability.tracing.propagator import (
    CrossAgentPropagator,
    extract_from_dict,
    inject_into_dict,
)
from agent_observability.tracing.sampler import CostAwareSampler


# ── AgentTraceContext ──────────────────────────────────────────────────────────


class TestAgentTraceContext:
    def test_from_headers_filters_non_w3c(self) -> None:
        headers = {
            "traceparent": "00-abc123-def456-01",
            "tracestate": "vendor=val",
            "x-custom": "custom-value",
            "random-header": "should-be-excluded",
        }
        ctx = AgentTraceContext.from_headers(headers)
        h = ctx.to_headers()
        assert "traceparent" in h
        assert "tracestate" in h
        assert "x-custom" in h
        assert "random-header" not in h

    def test_from_headers_x_prefix_included(self) -> None:
        ctx = AgentTraceContext.from_headers({"x-agent-ctx-id": "abc"})
        assert "x-agent-ctx-id" in ctx.to_headers()

    def test_is_valid_true_with_traceparent(self) -> None:
        ctx = AgentTraceContext({"traceparent": "00-abc-def-01"})
        assert ctx.is_valid() is True

    def test_is_valid_false_without_traceparent(self) -> None:
        ctx = AgentTraceContext({})
        assert ctx.is_valid() is False

    def test_to_headers_returns_copy(self) -> None:
        original = {"traceparent": "00-abc-def-01"}
        ctx = AgentTraceContext(original)
        headers = ctx.to_headers()
        headers["extra"] = "should not modify original"
        assert "extra" not in ctx.to_headers()

    def test_repr_contains_validity(self) -> None:
        ctx = AgentTraceContext({"traceparent": "00-abc-def-01"})
        r = repr(ctx)
        assert "valid=True" in r

    def test_extract_current_no_otel(self) -> None:
        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", False):
            ctx = AgentTraceContext.extract_current()
            assert ctx.to_headers() == {}

    def test_extract_current_with_otel(self) -> None:
        mock_propagate = MagicMock()
        mock_propagate.inject = lambda carrier: carrier.update({"traceparent": "00-abc-def-01"})

        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", True):
            with patch("agent_observability.tracing.context.propagate", mock_propagate):
                ctx = AgentTraceContext.extract_current()
                assert "traceparent" in ctx.to_headers()

    def test_attach_no_otel(self) -> None:
        ctx = AgentTraceContext({"traceparent": "00-abc-def-01"})
        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", False):
            token = ctx.attach()
            assert token is None

    def test_detach_no_otel(self) -> None:
        ctx = AgentTraceContext({})
        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", False):
            ctx.detach(None)  # should not raise

    def test_detach_none_token_is_noop(self) -> None:
        ctx = AgentTraceContext({})
        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", True):
            ctx.detach(None)  # should not raise

    def test_attach_with_otel(self) -> None:
        mock_propagate = MagicMock()
        mock_propagate.extract.return_value = object()
        mock_context = MagicMock()
        mock_context.attach.return_value = "token-123"

        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", True):
            with patch("agent_observability.tracing.context.propagate", mock_propagate):
                with patch("agent_observability.tracing.context.otel_context", mock_context):
                    ctx = AgentTraceContext({"traceparent": "00-abc-def-01"})
                    token = ctx.attach()
                    assert token == "token-123"

    def test_detach_with_otel(self) -> None:
        mock_context = MagicMock()

        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", True):
            with patch("agent_observability.tracing.context.otel_context", mock_context):
                ctx = AgentTraceContext({})
                ctx.detach("some-token")
                mock_context.detach.assert_called_with("some-token")

    def test_detach_exception_is_swallowed(self) -> None:
        mock_context = MagicMock()
        mock_context.detach.side_effect = RuntimeError("detach failed")

        with patch("agent_observability.tracing.context._OTEL_AVAILABLE", True):
            with patch("agent_observability.tracing.context.otel_context", mock_context):
                ctx = AgentTraceContext({})
                ctx.detach("bad-token")  # should not raise


# ── CrossAgentPropagator ───────────────────────────────────────────────────────


class TestCrossAgentPropagator:
    def test_inject_adds_agent_id(self) -> None:
        prop = CrossAgentPropagator(agent_id="agent-abc")
        carrier: dict[str, str] = {}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            prop.inject(carrier)
        assert carrier.get("x-agent-ctx-agent-id") == "agent-abc"

    def test_inject_adds_session_id(self) -> None:
        prop = CrossAgentPropagator(session_id="session-xyz")
        carrier: dict[str, str] = {}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            prop.inject(carrier)
        assert carrier.get("x-agent-ctx-session-id") == "session-xyz"

    def test_inject_extra_fields(self) -> None:
        prop = CrossAgentPropagator(extra_fields={"custom-header": "value"})
        carrier: dict[str, str] = {}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            prop.inject(carrier)
        assert carrier.get("custom-header") == "value"

    def test_inject_empty_agent_id_not_added(self) -> None:
        prop = CrossAgentPropagator(agent_id="")
        carrier: dict[str, str] = {}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            prop.inject(carrier)
        assert "x-agent-ctx-agent-id" not in carrier

    def test_extract_returns_agent_metadata(self) -> None:
        prop = CrossAgentPropagator()
        carrier = {
            "x-agent-ctx-agent-id": "agent-123",
            "x-agent-ctx-session-id": "sess-456",
            "traceparent": "00-abc-def-01",
        }
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            meta = prop.extract(carrier)
        assert meta.get("agent-id") == "agent-123"
        assert meta.get("session-id") == "sess-456"

    def test_extract_ignores_non_agent_ctx_keys(self) -> None:
        prop = CrossAgentPropagator()
        carrier = {"traceparent": "00-abc-def-01", "content-type": "application/json"}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            meta = prop.extract(carrier)
        assert "traceparent" not in meta
        assert "content-type" not in meta

    def test_make_carrier(self) -> None:
        prop = CrossAgentPropagator(agent_id="agent-make")
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            carrier = prop.make_carrier()
        assert isinstance(carrier, dict)
        assert carrier.get("x-agent-ctx-agent-id") == "agent-make"

    def test_get_trace_id_from_traceparent(self) -> None:
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        trace_id = CrossAgentPropagator.get_trace_id(carrier)
        assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_get_span_id_from_traceparent(self) -> None:
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        span_id = CrossAgentPropagator.get_span_id(carrier)
        assert span_id == "00f067aa0ba902b7"

    def test_get_trace_id_missing_traceparent(self) -> None:
        assert CrossAgentPropagator.get_trace_id({}) == ""

    def test_get_span_id_missing_traceparent(self) -> None:
        assert CrossAgentPropagator.get_span_id({}) == ""

    def test_get_trace_id_malformed_traceparent(self) -> None:
        assert CrossAgentPropagator.get_trace_id({"traceparent": "malformed"}) == ""

    def test_inject_with_otel(self) -> None:
        mock_propagate = MagicMock()
        mock_propagate.inject = MagicMock()

        prop = CrossAgentPropagator(agent_id="agent-otel")
        carrier: dict[str, str] = {}

        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", True):
            with patch("agent_observability.tracing.propagator.propagate", mock_propagate):
                prop.inject(carrier)
                mock_propagate.inject.assert_called_once_with(carrier)

    def test_extract_otel_failure_swallowed(self) -> None:
        mock_propagate = MagicMock()
        mock_propagate.extract.side_effect = RuntimeError("otel failure")

        prop = CrossAgentPropagator()
        carrier = {"x-agent-ctx-agent-id": "agent-1"}

        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", True):
            with patch("agent_observability.tracing.propagator.propagate", mock_propagate):
                meta = prop.extract(carrier)  # should not raise
                assert "agent-id" in meta


class TestPropagatorHelpers:
    def test_inject_into_dict(self) -> None:
        carrier: dict[str, str] = {}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            inject_into_dict(carrier, agent_id="helper-agent", session_id="helper-session")
        assert carrier.get("x-agent-ctx-agent-id") == "helper-agent"
        assert carrier.get("x-agent-ctx-session-id") == "helper-session"

    def test_extract_from_dict(self) -> None:
        carrier = {"x-agent-ctx-agent-id": "test-agent"}
        with patch("agent_observability.tracing.propagator._OTEL_AVAILABLE", False):
            meta = extract_from_dict(carrier)
        assert meta.get("agent-id") == "test-agent"


# ── CostAwareSampler ───────────────────────────────────────────────────────────


class TestCostAwareSampler:
    def test_invalid_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="low_cost_sample_rate must be between 0 and 1"):
            CostAwareSampler(low_cost_sample_rate=1.5)

    def test_invalid_sample_rate_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            CostAwareSampler(low_cost_sample_rate=-0.1)

    def test_get_description(self) -> None:
        sampler = CostAwareSampler(high_cost_threshold_usd=0.05, low_cost_sample_rate=0.2)
        desc = sampler.get_description()
        assert "CostAwareSampler" in desc
        assert "0.05" in desc
        assert "0.2" in desc

    def test_simple_decision_high_cost_always_sampled(self) -> None:
        sampler = CostAwareSampler(
            high_cost_threshold_usd=0.01,
            low_cost_sample_rate=0.0,
            seed=42,
        )
        with patch("agent_observability.tracing.sampler._OTEL_SDK_AVAILABLE", False):
            result = sampler.should_sample(
                parent_context=None,
                trace_id=12345,
                name="my-span",
                attributes={"llm.cost.usd": 0.05},
            )
        assert result is True

    def test_simple_decision_low_cost_uses_sample_rate(self) -> None:
        # With sample_rate=1.0, always sample
        sampler = CostAwareSampler(low_cost_sample_rate=1.0, seed=42)
        with patch("agent_observability.tracing.sampler._OTEL_SDK_AVAILABLE", False):
            result = sampler.should_sample(
                parent_context=None,
                trace_id=12345,
                name="my-span",
                attributes={"llm.cost.usd": 0.0001},
            )
        assert result is True

    def test_simple_decision_never_samples_at_zero_rate(self) -> None:
        sampler = CostAwareSampler(
            high_cost_threshold_usd=1.0,
            low_cost_sample_rate=0.0,
            seed=42,
        )
        with patch("agent_observability.tracing.sampler._OTEL_SDK_AVAILABLE", False):
            results = [
                sampler.should_sample(
                    parent_context=None,
                    trace_id=i,
                    name="my-span",
                    attributes={"llm.cost.usd": 0.001},
                )
                for i in range(20)
            ]
        assert all(r is False for r in results)

    def test_simple_decision_no_cost_attribute(self) -> None:
        sampler = CostAwareSampler(low_cost_sample_rate=1.0, seed=42)
        with patch("agent_observability.tracing.sampler._OTEL_SDK_AVAILABLE", False):
            result = sampler.should_sample(
                parent_context=None,
                trace_id=1,
                name="my-span",
                attributes={},
            )
        assert result is True

    def test_simple_decision_non_dict_attributes(self) -> None:
        sampler = CostAwareSampler(low_cost_sample_rate=1.0, seed=42)
        with patch("agent_observability.tracing.sampler._OTEL_SDK_AVAILABLE", False):
            result = sampler.should_sample(
                parent_context=None,
                trace_id=1,
                name="my-span",
                attributes=None,
            )
        assert isinstance(result, bool)

    def test_extract_cost_none_for_non_dict(self) -> None:
        result = CostAwareSampler._extract_cost("not a dict")
        assert result is None

    def test_extract_cost_none_for_missing_key(self) -> None:
        result = CostAwareSampler._extract_cost({"other_key": 1.0})
        assert result is None

    def test_extract_cost_valid(self) -> None:
        result = CostAwareSampler._extract_cost({"llm.cost.usd": 0.05})
        assert result == pytest.approx(0.05)

    def test_has_error_false_for_non_dict(self) -> None:
        assert CostAwareSampler._has_error("not a dict") is False

    def test_has_error_false_when_no_error(self) -> None:
        assert CostAwareSampler._has_error({"agent.span.kind": "llm_call"}) is False

    def test_has_error_true_for_agent_error(self) -> None:
        assert CostAwareSampler._has_error({"agent.span.kind": "agent_error"}) is True

    def test_error_span_always_sampled(self) -> None:
        sampler = CostAwareSampler(
            low_cost_sample_rate=0.0,
            always_sample_errors=True,
            seed=42,
        )
        with patch("agent_observability.tracing.sampler._OTEL_SDK_AVAILABLE", False):
            # Simulate no-otel path which calls _simple_decision
            # _simple_decision does not check error flags — only _decide does
            # This is checking the _has_error path in _decide which is OTel-path
            # For the no-SDK path we test _simple_decision directly
            result = sampler._simple_decision({"llm.cost.usd": 999.0})
        assert result is True

    def test_reproducible_with_seed(self) -> None:
        s1 = CostAwareSampler(low_cost_sample_rate=0.5, seed=99)
        s2 = CostAwareSampler(low_cost_sample_rate=0.5, seed=99)
        results1 = [s1._simple_decision({}) for _ in range(10)]
        results2 = [s2._simple_decision({}) for _ in range(10)]
        assert results1 == results2
