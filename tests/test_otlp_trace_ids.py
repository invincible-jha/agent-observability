"""Tests for OTLP exporter trace/span ID generation.

Verifies that:
- Trace IDs are 32 lowercase hex chars (128-bit).
- Span IDs are 16 lowercase hex chars (64-bit).
- All spans in a single export batch share the same trace ID.
- Different batches receive different trace IDs.
- No two span IDs collide across 1000 calls.
- All generated IDs contain only valid hex characters (0-9a-f).
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from agent_observability.exporter.otlp import (
    OTLPExporter,
    _agent_span_to_otlp_dict,
    _generate_span_id,
    _generate_trace_id,
)
from agent_observability.spans.types import AgentSpan, AgentSpanKind

# ── Constants ──────────────────────────────────────────────────────────────────

_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")
_TRACE_ID_LENGTH = 32
_SPAN_ID_LENGTH = 16
_UNIQUENESS_SAMPLE_SIZE = 1000


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_agent_span() -> AgentSpan:
    """Build a minimal AgentSpan backed by a _NoOpSpan for testing."""
    from agent_observability.spans.types import _NoOpSpan

    noop = _NoOpSpan()
    return AgentSpan(span=noop, kind=AgentSpanKind.LLM_CALL)


# ── _generate_trace_id ─────────────────────────────────────────────────────────


class TestGenerateTraceId:
    def test_length_is_32_chars(self) -> None:
        trace_id = _generate_trace_id()
        assert len(trace_id) == _TRACE_ID_LENGTH, (
            f"Expected 32 hex chars, got {len(trace_id)}: {trace_id!r}"
        )

    def test_contains_only_valid_hex_chars(self) -> None:
        trace_id = _generate_trace_id()
        assert _HEX_PATTERN.match(trace_id), (
            f"Trace ID contains non-hex characters: {trace_id!r}"
        )

    def test_successive_calls_produce_unique_ids(self) -> None:
        first = _generate_trace_id()
        second = _generate_trace_id()
        assert first != second, "Two successive trace IDs must not collide"


# ── _generate_span_id ──────────────────────────────────────────────────────────


class TestGenerateSpanId:
    def test_length_is_16_chars(self) -> None:
        span_id = _generate_span_id()
        assert len(span_id) == _SPAN_ID_LENGTH, (
            f"Expected 16 hex chars, got {len(span_id)}: {span_id!r}"
        )

    def test_contains_only_valid_hex_chars(self) -> None:
        span_id = _generate_span_id()
        assert _HEX_PATTERN.match(span_id), (
            f"Span ID contains non-hex characters: {span_id!r}"
        )

    def test_no_collisions_across_1000_calls(self) -> None:
        generated_ids: set[str] = set()
        for _ in range(_UNIQUENESS_SAMPLE_SIZE):
            span_id = _generate_span_id()
            assert span_id not in generated_ids, (
                f"Span ID collision detected: {span_id!r}"
            )
            generated_ids.add(span_id)


# ── _agent_span_to_otlp_dict ───────────────────────────────────────────────────


class TestAgentSpanToOtlpDict:
    def test_trace_id_is_not_all_zeros_when_no_trace_id_provided(self) -> None:
        span = _make_agent_span()
        result = _agent_span_to_otlp_dict(span)
        trace_id = result["traceId"]
        assert isinstance(trace_id, str)
        assert trace_id != "00000000000000000000000000000000", (
            "traceId must not be the all-zero placeholder"
        )

    def test_trace_id_length_when_auto_generated(self) -> None:
        span = _make_agent_span()
        result = _agent_span_to_otlp_dict(span)
        assert len(str(result["traceId"])) == _TRACE_ID_LENGTH

    def test_span_id_is_not_all_zeros(self) -> None:
        span = _make_agent_span()
        result = _agent_span_to_otlp_dict(span)
        span_id = result["spanId"]
        assert isinstance(span_id, str)
        assert span_id != "0000000000000000", (
            "spanId must not be the all-zero placeholder"
        )

    def test_span_id_length(self) -> None:
        span = _make_agent_span()
        result = _agent_span_to_otlp_dict(span)
        assert len(str(result["spanId"])) == _SPAN_ID_LENGTH

    def test_provided_trace_id_is_used(self) -> None:
        span = _make_agent_span()
        provided_trace_id = "abcdef1234567890abcdef1234567890"
        result = _agent_span_to_otlp_dict(span, trace_id=provided_trace_id)
        assert result["traceId"] == provided_trace_id

    def test_trace_id_hex_chars_only_when_auto_generated(self) -> None:
        span = _make_agent_span()
        result = _agent_span_to_otlp_dict(span)
        assert _HEX_PATTERN.match(str(result["traceId"]))

    def test_span_id_hex_chars_only(self) -> None:
        span = _make_agent_span()
        result = _agent_span_to_otlp_dict(span)
        assert _HEX_PATTERN.match(str(result["spanId"]))


# ── OTLPExporter._build_payload batch correlation ──────────────────────────────


class TestBuildPayloadBatchCorrelation:
    def _extract_span_dicts(self, payload: dict[str, object]) -> list[dict[str, object]]:
        """Navigate the OTLP resourceSpans structure to the flat span list."""
        resource_spans = payload["resourceSpans"]
        assert isinstance(resource_spans, list)
        scope_spans = resource_spans[0]["scopeSpans"]
        assert isinstance(scope_spans, list)
        spans = scope_spans[0]["spans"]
        assert isinstance(spans, list)
        return spans  # type: ignore[return-value]

    def test_all_spans_in_batch_share_trace_id(self) -> None:
        exporter = OTLPExporter()
        spans = [_make_agent_span() for _ in range(5)]
        payload = exporter._build_payload(spans)

        span_dicts = self._extract_span_dicts(payload)
        trace_ids = {str(sd["traceId"]) for sd in span_dicts}

        assert len(trace_ids) == 1, (
            f"Expected all spans to share one trace ID, got {len(trace_ids)}: {trace_ids}"
        )

    def test_batch_trace_id_is_not_all_zeros(self) -> None:
        exporter = OTLPExporter()
        spans = [_make_agent_span()]
        payload = exporter._build_payload(spans)

        span_dicts = self._extract_span_dicts(payload)
        trace_id = str(span_dicts[0]["traceId"])
        assert trace_id != "00000000000000000000000000000000"

    def test_batch_trace_id_is_valid_32_char_hex(self) -> None:
        exporter = OTLPExporter()
        spans = [_make_agent_span()]
        payload = exporter._build_payload(spans)

        span_dicts = self._extract_span_dicts(payload)
        trace_id = str(span_dicts[0]["traceId"])
        assert len(trace_id) == _TRACE_ID_LENGTH
        assert _HEX_PATTERN.match(trace_id)

    def test_different_batches_get_different_trace_ids(self) -> None:
        exporter = OTLPExporter()
        span_a = _make_agent_span()
        span_b = _make_agent_span()

        payload_a = exporter._build_payload([span_a])
        payload_b = exporter._build_payload([span_b])

        dicts_a = self._extract_span_dicts(payload_a)
        dicts_b = self._extract_span_dicts(payload_b)

        trace_id_a = str(dicts_a[0]["traceId"])
        trace_id_b = str(dicts_b[0]["traceId"])

        assert trace_id_a != trace_id_b, (
            "Two separate batch payloads must not share the same trace ID"
        )

    def test_each_span_in_batch_has_unique_span_id(self) -> None:
        exporter = OTLPExporter()
        spans = [_make_agent_span() for _ in range(10)]
        payload = exporter._build_payload(spans)

        span_dicts = self._extract_span_dicts(payload)
        span_ids = [str(sd["spanId"]) for sd in span_dicts]

        assert len(span_ids) == len(set(span_ids)), (
            "Every span in a batch must have a unique spanId"
        )

    def test_span_ids_in_batch_are_valid_16_char_hex(self) -> None:
        exporter = OTLPExporter()
        spans = [_make_agent_span() for _ in range(3)]
        payload = exporter._build_payload(spans)

        span_dicts = self._extract_span_dicts(payload)
        for span_dict in span_dicts:
            span_id = str(span_dict["spanId"])
            assert len(span_id) == _SPAN_ID_LENGTH, (
                f"spanId length expected 16, got {len(span_id)}: {span_id!r}"
            )
            assert _HEX_PATTERN.match(span_id), (
                f"spanId contains non-hex characters: {span_id!r}"
            )
