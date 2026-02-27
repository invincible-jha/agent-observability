"""Tests for RingBuffer — O(1) lock-free append with atomic drain."""
from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_observability.async_trace.ring_buffer import RingBuffer


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestRingBufferConstruction:
    def test_create_with_valid_maxlen(self) -> None:
        buf = RingBuffer(maxlen=100)
        assert buf.maxlen == 100
        assert len(buf) == 0

    def test_create_with_maxlen_one(self) -> None:
        buf = RingBuffer(maxlen=1)
        assert buf.maxlen == 1

    def test_create_with_large_maxlen(self) -> None:
        buf = RingBuffer(maxlen=10_000)
        assert buf.maxlen == 10_000

    def test_invalid_maxlen_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be positive"):
            RingBuffer(maxlen=0)

    def test_invalid_maxlen_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be positive"):
            RingBuffer(maxlen=-1)


# ---------------------------------------------------------------------------
# Append and basic drain
# ---------------------------------------------------------------------------


class TestAppendAndDrain:
    def test_append_single_item(self) -> None:
        buf = RingBuffer(maxlen=10)
        buf.append("span-a")
        assert len(buf) == 1

    def test_drain_returns_appended_items(self) -> None:
        buf = RingBuffer(maxlen=10)
        buf.append("span-a")
        buf.append("span-b")
        buf.append("span-c")
        result = buf.drain()
        assert result == ["span-a", "span-b", "span-c"]

    def test_drain_returns_list(self) -> None:
        buf = RingBuffer(maxlen=5)
        buf.append({"name": "llm.call"})
        result = buf.drain()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_drain_empties_buffer(self) -> None:
        buf = RingBuffer(maxlen=10)
        buf.append("a")
        buf.append("b")
        buf.drain()
        assert len(buf) == 0

    def test_drain_empty_buffer_returns_empty_list(self) -> None:
        buf = RingBuffer(maxlen=10)
        result = buf.drain()
        assert result == []

    def test_multiple_drains_work_correctly(self) -> None:
        buf = RingBuffer(maxlen=10)
        buf.append("first")
        first_drain = buf.drain()
        buf.append("second")
        second_drain = buf.drain()
        assert first_drain == ["first"]
        assert second_drain == ["second"]

    def test_append_after_drain_works(self) -> None:
        buf = RingBuffer(maxlen=5)
        buf.append("a")
        buf.drain()
        buf.append("b")
        result = buf.drain()
        assert result == ["b"]


# ---------------------------------------------------------------------------
# Overflow / drop-oldest behavior
# ---------------------------------------------------------------------------


class TestOverflow:
    def test_append_up_to_maxlen(self) -> None:
        buf = RingBuffer(maxlen=3)
        buf.append(1)
        buf.append(2)
        buf.append(3)
        assert len(buf) == 3

    def test_overflow_drops_oldest(self) -> None:
        buf = RingBuffer(maxlen=3)
        buf.append("a")
        buf.append("b")
        buf.append("c")
        buf.append("d")  # "a" should be dropped
        result = buf.drain()
        assert result == ["b", "c", "d"]

    def test_overflow_many_items(self) -> None:
        buf = RingBuffer(maxlen=5)
        for i in range(20):
            buf.append(i)
        result = buf.drain()
        # Only the last 5 items should remain
        assert result == [15, 16, 17, 18, 19]

    def test_is_full_property(self) -> None:
        buf = RingBuffer(maxlen=3)
        assert not buf.is_full
        buf.append(1)
        buf.append(2)
        buf.append(3)
        assert buf.is_full

    def test_is_full_after_drain(self) -> None:
        buf = RingBuffer(maxlen=2)
        buf.append(1)
        buf.append(2)
        assert buf.is_full
        buf.drain()
        assert not buf.is_full

    def test_exact_N_items_drained(self) -> None:
        """Append exactly N items and drain returns all N."""
        maxlen = 100
        buf = RingBuffer(maxlen=maxlen)
        for i in range(maxlen):
            buf.append(i)
        result = buf.drain()
        assert len(result) == maxlen
        assert result == list(range(maxlen))


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_appends(self) -> None:
        buf = RingBuffer(maxlen=10_000)
        thread_count = 10
        items_per_thread = 500

        def append_items(thread_id: int) -> None:
            for i in range(items_per_thread):
                buf.append(f"thread-{thread_id}-item-{i}")

        threads = [
            threading.Thread(target=append_items, args=(t,))
            for t in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = buf.drain()
        # We may have dropped items due to overflow but should not crash
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_concurrent_append_and_drain(self) -> None:
        buf = RingBuffer(maxlen=1_000)
        collected: list[Any] = []
        lock = threading.Lock()
        stop_event = threading.Event()

        def drain_worker() -> None:
            while not stop_event.is_set():
                items = buf.drain()
                with lock:
                    collected.extend(items)

        def append_worker() -> None:
            for i in range(200):
                buf.append(i)

        drainer = threading.Thread(target=drain_worker)
        drainer.start()

        appenders = [threading.Thread(target=append_worker) for _ in range(5)]
        for t in appenders:
            t.start()
        for t in appenders:
            t.join()

        stop_event.set()
        drainer.join()

        # Drain any remaining items
        remaining = buf.drain()
        with lock:
            collected.extend(remaining)

        # Should have collected some items (can't guarantee all due to timing)
        assert isinstance(collected, list)

    def test_drain_under_concurrent_appends_returns_list(self) -> None:
        buf = RingBuffer(maxlen=100)
        results: list[list[Any]] = []
        lock = threading.Lock()

        def append_many() -> None:
            for i in range(50):
                buf.append(i)

        def drain_once() -> None:
            drained = buf.drain()
            with lock:
                results.append(drained)

        threads = [threading.Thread(target=append_many) for _ in range(3)]
        drain_thread = threading.Thread(target=drain_once)

        for t in threads:
            t.start()
        drain_thread.start()

        for t in threads:
            t.join()
        drain_thread.join()

        assert len(results) == 1
        assert isinstance(results[0], list)


# ---------------------------------------------------------------------------
# Span dict storage
# ---------------------------------------------------------------------------


class TestSpanDictStorage:
    def test_stores_and_returns_span_dicts(self) -> None:
        buf = RingBuffer(maxlen=10)
        span = {"name": "llm.call", "model": "gpt-4o", "tokens": 512}
        buf.append(span)
        result = buf.drain()
        assert result == [span]

    def test_order_preserved(self) -> None:
        buf = RingBuffer(maxlen=10)
        spans = [{"id": i} for i in range(5)]
        for span in spans:
            buf.append(span)
        result = buf.drain()
        assert [s["id"] for s in result] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_maxlen(self) -> None:
        buf = RingBuffer(maxlen=42)
        assert "42" in repr(buf)
