"""Tests for SpanEmitter — zero-overhead hot-path span emission."""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from agent_observability.async_trace.config import AsyncTraceConfig
from agent_observability.async_trace.ring_buffer import RingBuffer
from agent_observability.async_trace.span_emitter import SpanEmitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_emitter(
    buffer_size: int = 10_000,
    flush_interval: float = 60.0,  # effectively no auto-flush in unit tests
) -> tuple[SpanEmitter, list[Any]]:
    collected: list[Any] = []

    def exporter(spans: list[Any]) -> None:
        collected.extend(spans)

    config = AsyncTraceConfig(
        buffer_size=buffer_size,
        flush_interval_seconds=flush_interval,
    )
    emitter = SpanEmitter.create(exporter=exporter, config=config)
    return emitter, collected


# ---------------------------------------------------------------------------
# Construction via factory
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_create_returns_span_emitter(self) -> None:
        emitter, _ = make_emitter()
        assert isinstance(emitter, SpanEmitter)

    def test_create_with_default_config(self) -> None:
        collected: list[Any] = []
        emitter = SpanEmitter.create(exporter=collected.extend)
        assert isinstance(emitter, SpanEmitter)

    def test_buffer_size_initially_zero(self) -> None:
        emitter, _ = make_emitter()
        assert emitter.buffer_size == 0


# ---------------------------------------------------------------------------
# emit() — hot path
# ---------------------------------------------------------------------------


class TestEmit:
    def test_emit_adds_to_buffer(self) -> None:
        emitter, _ = make_emitter()
        emitter.emit("llm.call", {"model": "gpt-4o"})
        assert emitter.buffer_size == 1

    def test_emit_span_has_required_fields(self) -> None:
        emitter, collected = make_emitter()
        emitter.emit("test.span", {"key": "value"})
        emitter.flush()
        assert len(collected) == 1
        span = collected[0]
        assert span["name"] == "test.span"
        assert "span_id" in span
        assert "start_time_ns" in span
        assert span["attributes"] == {"key": "value"}

    def test_emit_without_attributes(self) -> None:
        emitter, collected = make_emitter()
        emitter.emit("bare.span")
        emitter.flush()
        assert len(collected) == 1
        assert collected[0]["attributes"] == {}

    def test_emit_returns_none(self) -> None:
        emitter, _ = make_emitter()
        result = emitter.emit("span")
        assert result is None

    def test_emit_multiple_spans(self) -> None:
        emitter, collected = make_emitter()
        for i in range(5):
            emitter.emit(f"span-{i}", {"index": i})
        emitter.flush()
        assert len(collected) == 5

    def test_emit_span_ids_are_unique(self) -> None:
        emitter, collected = make_emitter()
        for _ in range(100):
            emitter.emit("span")
        emitter.flush()
        span_ids = [s["span_id"] for s in collected]
        assert len(set(span_ids)) == 100

    def test_emit_does_not_block(self) -> None:
        """emit() must complete in under 1ms for 10K calls total."""
        emitter, _ = make_emitter(buffer_size=10_000)
        start = time.perf_counter()
        for _ in range(10_000):
            emitter.emit("perf.span")
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Total of 10K emits should complete well under 1000ms
        # (generous limit to avoid flakiness on slow CI runners)
        assert elapsed_ms < 1000, f"10K emits took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# span() context manager
# ---------------------------------------------------------------------------


class TestSpanContextManager:
    def test_span_records_name(self) -> None:
        emitter, collected = make_emitter()
        with emitter.span("my.operation"):
            pass
        emitter.flush()
        assert len(collected) == 1
        assert collected[0]["name"] == "my.operation"

    def test_span_has_start_and_end_times(self) -> None:
        emitter, collected = make_emitter()
        with emitter.span("timed.op"):
            pass
        emitter.flush()
        span = collected[0]
        assert "start_time_ns" in span
        assert "end_time_ns" in span
        assert span["end_time_ns"] >= span["start_time_ns"]

    def test_span_records_duration(self) -> None:
        emitter, collected = make_emitter()
        with emitter.span("timed.op"):
            time.sleep(0.01)
        emitter.flush()
        span = collected[0]
        assert span["duration_ns"] >= 10_000_000  # at least 10ms in ns

    def test_span_status_ok_on_success(self) -> None:
        emitter, collected = make_emitter()
        with emitter.span("success.op"):
            pass
        emitter.flush()
        assert collected[0]["status"] == "ok"

    def test_span_status_error_on_exception(self) -> None:
        emitter, collected = make_emitter()
        with pytest.raises(ValueError):
            with emitter.span("failing.op"):
                raise ValueError("something went wrong")
        emitter.flush()
        span = collected[0]
        assert span["status"] == "error"
        assert "something went wrong" in span["error"]

    def test_span_error_type_recorded(self) -> None:
        emitter, collected = make_emitter()
        with pytest.raises(RuntimeError):
            with emitter.span("bad.op"):
                raise RuntimeError("boom")
        emitter.flush()
        assert collected[0]["error_type"] == "RuntimeError"

    def test_span_mutable_dict_allows_attribute_addition(self) -> None:
        emitter, collected = make_emitter()
        with emitter.span("mutable.op") as span_ctx:
            span_ctx["custom_key"] = "custom_value"
            span_ctx["attributes"]["model"] = "claude"
        emitter.flush()
        span = collected[0]
        assert span["custom_key"] == "custom_value"
        assert span["attributes"]["model"] == "claude"

    def test_span_with_initial_attributes(self) -> None:
        emitter, collected = make_emitter()
        with emitter.span("op", {"initial": True}):
            pass
        emitter.flush()
        assert collected[0]["attributes"]["initial"] is True

    def test_span_reraises_exception(self) -> None:
        emitter, _ = make_emitter()
        with pytest.raises(KeyError, match="test_key"):
            with emitter.span("op"):
                raise KeyError("test_key")


# ---------------------------------------------------------------------------
# flush() integration
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_returns_count(self) -> None:
        emitter, _ = make_emitter()
        emitter.emit("a")
        emitter.emit("b")
        count = emitter.flush()
        assert count == 2

    def test_flush_clears_buffer(self) -> None:
        emitter, _ = make_emitter()
        emitter.emit("span")
        emitter.flush()
        assert emitter.buffer_size == 0

    def test_flush_passes_spans_to_exporter(self) -> None:
        emitter, collected = make_emitter()
        emitter.emit("span-a", {"x": 1})
        emitter.emit("span-b", {"x": 2})
        emitter.flush()
        assert len(collected) == 2
        names = [s["name"] for s in collected]
        assert "span-a" in names
        assert "span-b" in names


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_and_stop(self) -> None:
        emitter, _ = make_emitter(flush_interval=0.05)
        emitter.start()
        emitter.stop()

    def test_stop_flushes_remaining_spans(self) -> None:
        emitter, collected = make_emitter(flush_interval=60.0)
        emitter.start()
        emitter.emit("final-span")
        emitter.stop()
        assert len(collected) == 1
        assert collected[0]["name"] == "final-span"

    def test_auto_flush_via_timer(self) -> None:
        collected: list[Any] = []

        def exporter(spans: list[Any]) -> None:
            collected.extend(spans)

        config = AsyncTraceConfig(
            buffer_size=100,
            flush_interval_seconds=0.05,
        )
        emitter = SpanEmitter.create(exporter=exporter, config=config)
        emitter.start()
        try:
            emitter.emit("auto-flush-span")
            deadline = time.time() + 0.5
            while time.time() < deadline and len(collected) == 0:
                time.sleep(0.01)
            assert len(collected) >= 1
        finally:
            emitter.stop()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_emits_from_multiple_threads(self) -> None:
        emitter, collected = make_emitter(buffer_size=10_000)
        thread_count = 10
        emits_per_thread = 100

        def emit_worker(thread_id: int) -> None:
            for i in range(emits_per_thread):
                emitter.emit(f"thread-{thread_id}.span-{i}")

        threads = [threading.Thread(target=emit_worker, args=(t,)) for t in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        emitter.flush()
        assert len(collected) == thread_count * emits_per_thread


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_is_string(self) -> None:
        emitter, _ = make_emitter()
        assert isinstance(repr(emitter), str)
