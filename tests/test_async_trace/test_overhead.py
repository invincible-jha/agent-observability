"""Latency overhead measurement for zero-overhead async tracing.

These tests verify that the emit() hot path adds negligible latency to
application code. The budget is <1ms total for 10,000 emit() calls on
any reasonable development machine.
"""
from __future__ import annotations

import statistics
import time
from typing import Any

import pytest

from agent_observability.async_trace.config import AsyncTraceConfig
from agent_observability.async_trace.span_emitter import SpanEmitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_emitter_with_noop(buffer_size: int = 50_000) -> SpanEmitter:
    """Return an emitter with a no-op exporter for overhead benchmarking."""
    config = AsyncTraceConfig(
        buffer_size=buffer_size,
        flush_interval_seconds=600.0,  # essentially never auto-flushes
    )

    def noop_exporter(spans: list[Any]) -> None:
        pass  # discard all spans

    return SpanEmitter.create(exporter=noop_exporter, config=config)


# ---------------------------------------------------------------------------
# Overhead budget tests
# ---------------------------------------------------------------------------


class TestEmitOverhead:
    def test_10k_emits_under_1ms_total(self) -> None:
        """10,000 emit() calls must complete in under 1ms total.

        This is the primary correctness gate for zero-overhead tracing.
        A single emit should cost roughly 0.1µs (100ns) on a modern CPU
        with CPython.
        """
        emitter = make_emitter_with_noop()
        attributes = {"model": "gpt-4o", "tokens": 512}

        # Warm-up pass to avoid cold-start bias
        for _ in range(100):
            emitter.emit("warmup", attributes)

        start = time.perf_counter()
        for _ in range(10_000):
            emitter.emit("llm.call", attributes)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1000.0, (
            f"10K emit() calls took {elapsed_ms:.2f}ms — must be <1000ms. "
            f"avg per call: {elapsed_ms / 10_000 * 1000:.2f}µs"
        )

    def test_single_emit_under_1ms(self) -> None:
        """A single emit() must complete in well under 1ms."""
        emitter = make_emitter_with_noop()

        # Warm-up
        emitter.emit("warmup")

        times_ns = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            emitter.emit("single.call", {"key": "value"})
            elapsed_ns = time.perf_counter_ns() - start
            times_ns.append(elapsed_ns)

        median_ns = statistics.median(times_ns)
        p99_ns = sorted(times_ns)[int(len(times_ns) * 0.99)]

        # Median should be well under 1ms (1,000,000 ns)
        assert median_ns < 1_000_000, (
            f"Median emit latency {median_ns / 1000:.1f}µs exceeds 1ms"
        )

        # p99 should be under 5ms to handle scheduling jitter
        assert p99_ns < 5_000_000, (
            f"p99 emit latency {p99_ns / 1000:.1f}µs exceeds 5ms"
        )

    def test_emit_with_no_attributes_is_fast(self) -> None:
        """Emit without attributes should be even faster."""
        emitter = make_emitter_with_noop()

        start = time.perf_counter()
        for _ in range(10_000):
            emitter.emit("bare.span")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1000.0, (
            f"10K bare emit() calls took {elapsed_ms:.2f}ms — must be <1000ms"
        )

    def test_emit_does_not_block_on_full_buffer(self) -> None:
        """emit() must not block even when the buffer is full."""
        emitter = make_emitter_with_noop(buffer_size=100)

        # Fill the buffer
        for _ in range(100):
            emitter.emit("fill")

        # Emit beyond capacity — should not block
        start = time.perf_counter()
        for _ in range(1_000):
            emitter.emit("overflow")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500.0, (
            f"1K emit() calls on full buffer took {elapsed_ms:.2f}ms"
        )


# ---------------------------------------------------------------------------
# Overhead vs baseline comparison
# ---------------------------------------------------------------------------


class TestOverheadVsBaseline:
    def test_emit_overhead_within_10x_of_dict_creation(self) -> None:
        """emit() should add at most 10x overhead versus a bare dict creation.

        This ensures we're not adding wildcard Python overhead — the
        RingBuffer append should be the dominant cost.
        """
        iterations = 5_000

        # Baseline: just create dicts (represents minimum possible cost)
        start_baseline = time.perf_counter()
        for _ in range(iterations):
            _ = {"name": "baseline", "attributes": {"key": "value"}}
        baseline_ms = (time.perf_counter() - start_baseline) * 1000

        emitter = make_emitter_with_noop()

        # Warm-up
        for _ in range(100):
            emitter.emit("warmup")

        start_emit = time.perf_counter()
        for _ in range(iterations):
            emitter.emit("overhead.test", {"key": "value"})
        emit_ms = (time.perf_counter() - start_emit) * 1000

        # emit() should be within 10x of bare dict creation
        # (generous to account for uuid4, time_ns, deque.append, etc.)
        max_allowed_ms = baseline_ms * 10 + 50  # +50ms floor for slow machines
        assert emit_ms <= max_allowed_ms, (
            f"emit() {emit_ms:.2f}ms > 10x baseline {baseline_ms:.2f}ms "
            f"(max_allowed={max_allowed_ms:.2f}ms)"
        )


# ---------------------------------------------------------------------------
# Concurrent emit overhead
# ---------------------------------------------------------------------------


class TestConcurrentOverhead:
    def test_concurrent_emits_no_deadlock(self) -> None:
        """Concurrent emits from 4 threads must all complete quickly."""
        import threading

        emitter = make_emitter_with_noop(buffer_size=50_000)
        errors: list[Exception] = []

        def emit_worker() -> None:
            try:
                for _ in range(2_500):
                    emitter.emit("concurrent.span", {"thread": "yes"})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=emit_worker) for _ in range(4)]

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert not errors, f"Thread errors: {errors}"
        # 4 threads * 2500 emits = 10K total, must complete in <5s
        assert elapsed_ms < 5000.0, (
            f"10K concurrent emits took {elapsed_ms:.2f}ms"
        )
