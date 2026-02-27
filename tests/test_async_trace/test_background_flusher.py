"""Tests for BackgroundFlusher — timer-based drain with exporter callback."""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from agent_observability.async_trace.background_flusher import BackgroundFlusher
from agent_observability.async_trace.ring_buffer import RingBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_flusher(
    buffer_size: int = 100,
    interval: float = 0.05,
    max_batch: int = 1_000,
) -> tuple[RingBuffer, list[Any], BackgroundFlusher]:
    """Create a RingBuffer + BackgroundFlusher with a list-appending exporter."""
    collected: list[Any] = []
    lock = threading.Lock()

    def exporter(spans: list[Any]) -> None:
        with lock:
            collected.extend(spans)

    buf = RingBuffer(maxlen=buffer_size)
    flusher = BackgroundFlusher(
        ring_buffer=buf,
        exporter=exporter,
        flush_interval_seconds=interval,
        max_batch_size=max_batch,
    )
    return buf, collected, flusher


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_not_running_initially(self) -> None:
        buf, _, flusher = make_flusher()
        assert not flusher._running

    def test_repr_contains_interval(self) -> None:
        buf, _, flusher = make_flusher(interval=0.5)
        assert "0.5" in repr(flusher)


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_sets_running_flag(self) -> None:
        buf, _, flusher = make_flusher()
        flusher.start()
        try:
            assert flusher._running
        finally:
            flusher.stop()

    def test_double_start_is_safe(self) -> None:
        buf, _, flusher = make_flusher()
        flusher.start()
        flusher.start()  # should not raise or create duplicate timers
        try:
            assert flusher._running
        finally:
            flusher.stop()

    def test_stop_clears_running_flag(self) -> None:
        buf, _, flusher = make_flusher()
        flusher.start()
        flusher.stop()
        assert not flusher._running

    def test_stop_without_start_is_safe(self) -> None:
        buf, _, flusher = make_flusher()
        flusher.stop()  # should not raise


# ---------------------------------------------------------------------------
# Automatic timer-based flush
# ---------------------------------------------------------------------------


class TestAutoFlush:
    def test_items_appear_in_exporter_after_interval(self) -> None:
        buf, collected, flusher = make_flusher(interval=0.05)
        flusher.start()
        try:
            buf.append({"name": "span-a"})
            buf.append({"name": "span-b"})
            # Wait up to 500ms for the timer to fire
            deadline = time.time() + 0.5
            while time.time() < deadline and len(collected) < 2:
                time.sleep(0.01)
            assert len(collected) >= 2
        finally:
            flusher.stop()

    def test_multiple_flush_cycles_collect_all_items(self) -> None:
        buf, collected, flusher = make_flusher(interval=0.04)
        flusher.start()
        try:
            for i in range(10):
                buf.append(i)
                time.sleep(0.01)  # spread appends across multiple cycles
            # Allow time for final flush
            time.sleep(0.2)
            # All items should have been collected
            assert len(collected) == 10
        finally:
            flusher.stop()


# ---------------------------------------------------------------------------
# Manual flush
# ---------------------------------------------------------------------------


class TestManualFlush:
    def test_flush_exports_buffered_spans(self) -> None:
        buf, collected, flusher = make_flusher()
        buf.append({"name": "x"})
        buf.append({"name": "y"})
        count = flusher.flush()
        assert count == 2
        assert len(collected) == 2

    def test_flush_returns_zero_on_empty_buffer(self) -> None:
        buf, collected, flusher = make_flusher()
        count = flusher.flush()
        assert count == 0
        assert len(collected) == 0

    def test_flush_drains_buffer_completely(self) -> None:
        buf, collected, flusher = make_flusher()
        for i in range(5):
            buf.append(i)
        flusher.flush()
        assert len(buf) == 0

    def test_flush_on_stopped_flusher(self) -> None:
        buf, collected, flusher = make_flusher()
        buf.append("span")
        flusher.stop()
        # stop() performs a final flush, so collected should have "span"
        assert "span" in collected


# ---------------------------------------------------------------------------
# Batch size limiting
# ---------------------------------------------------------------------------


class TestBatchSize:
    def test_large_drain_split_into_batches(self) -> None:
        batch_calls: list[int] = []

        def counting_exporter(spans: list[Any]) -> None:
            batch_calls.append(len(spans))

        buf = RingBuffer(maxlen=1_000)
        flusher = BackgroundFlusher(
            ring_buffer=buf,
            exporter=counting_exporter,
            flush_interval_seconds=60.0,  # won't fire automatically
            max_batch_size=10,
        )

        for i in range(25):
            buf.append(i)

        flusher.flush()

        # 25 items / batch size 10 → 3 calls (10, 10, 5)
        assert len(batch_calls) == 3
        assert batch_calls[0] == 10
        assert batch_calls[1] == 10
        assert batch_calls[2] == 5


# ---------------------------------------------------------------------------
# Exporter error handling
# ---------------------------------------------------------------------------


class TestExporterErrors:
    def test_exporter_exception_does_not_crash_flusher(self) -> None:
        def bad_exporter(spans: list[Any]) -> None:
            raise RuntimeError("exporter failed")

        buf = RingBuffer(maxlen=100)
        flusher = BackgroundFlusher(
            ring_buffer=buf,
            exporter=bad_exporter,
            flush_interval_seconds=60.0,
        )
        buf.append("span")
        # Should not raise
        flusher.flush()

    def test_exporter_error_is_logged(self) -> None:
        def bad_exporter(spans: list[Any]) -> None:
            raise ValueError("export failed")

        buf = RingBuffer(maxlen=100)
        flusher = BackgroundFlusher(
            ring_buffer=buf,
            exporter=bad_exporter,
            flush_interval_seconds=60.0,
        )
        buf.append("span")

        import logging

        with patch.object(
            logging.getLogger("agent_observability.async_trace.background_flusher"),
            "exception",
        ) as mock_log:
            flusher.flush()
            mock_log.assert_called_once()

    def test_timer_continues_after_exporter_error(self) -> None:
        call_count = 0
        fail_count = 0

        def flaky_exporter(spans: list[Any]) -> None:
            nonlocal call_count, fail_count
            call_count += 1
            if call_count == 1:
                fail_count += 1
                raise RuntimeError("transient error")

        buf = RingBuffer(maxlen=100)
        flusher = BackgroundFlusher(
            ring_buffer=buf,
            exporter=flaky_exporter,
            flush_interval_seconds=0.05,
        )
        flusher.start()
        try:
            buf.append("first")
            time.sleep(0.15)  # allow timer to fire at least twice
            buf.append("second")
            time.sleep(0.15)
        finally:
            flusher.stop()

        # Timer should have continued after the first failure
        assert call_count >= 2


# ---------------------------------------------------------------------------
# Stop flushes remaining spans
# ---------------------------------------------------------------------------


class TestStopFlushesRemaining:
    def test_stop_exports_remaining_spans(self) -> None:
        buf, collected, flusher = make_flusher(interval=60.0)  # long interval
        buf.append({"name": "final-span"})
        flusher.start()
        flusher.stop()
        assert any(s.get("name") == "final-span" for s in collected)
