"""Lock-free ring buffer for span collection.

The hot path (append) is lock-free: it delegates directly to
``collections.deque`` which is thread-safe for single-item appends on
CPython due to the GIL. The drain path acquires a short lock to perform
an atomic swap.

Design notes
------------
- ``deque(maxlen=N)`` automatically discards the oldest item when full,
  giving us O(1) append with zero blocking.
- ``drain()`` swaps in a fresh deque under a lock so concurrent appenders
  are blocked for the minimum possible time (one object creation + one
  attribute assignment).
- No locks are taken inside ``append()``, keeping the emit hot-path free
  of synchronization overhead.
"""
from __future__ import annotations

import threading
from collections import deque

# A span record is a plain mapping of string keys to arbitrary Python values.
# Using a type alias here avoids importing typing.Any in core code while still
# keeping the public API self-documenting.
SpanRecord = dict[str, object]


class RingBuffer:
    """O(1) thread-safe ring buffer that drops oldest entries when full.

    Parameters
    ----------
    maxlen:
        Maximum number of items the buffer holds. When the buffer is full
        and a new item is appended, the oldest item is silently dropped.

    Examples
    --------
    >>> buf = RingBuffer(maxlen=3)
    >>> buf.append("a")
    >>> buf.append("b")
    >>> buf.append("c")
    >>> buf.append("d")  # "a" is dropped
    >>> buf.drain()
    ['b', 'c', 'd']
    """

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._maxlen = maxlen
        self._deque: deque[SpanRecord] = deque(maxlen=maxlen)
        # Lock is ONLY used in drain() to swap out the deque atomically.
        # append() does NOT acquire the lock.
        self._drain_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hot path — no lock
    # ------------------------------------------------------------------

    def append(self, span: SpanRecord) -> None:
        """Append *span* to the buffer.

        This method is lock-free and never blocks. If the buffer is at
        capacity the oldest span is silently discarded.

        Parameters
        ----------
        span:
            A span record dict representing a span or event to buffer.
        """
        self._deque.append(span)

    # ------------------------------------------------------------------
    # Drain path — short lock for atomic swap
    # ------------------------------------------------------------------

    def drain(self) -> list[SpanRecord]:
        """Atomically drain and return all buffered spans.

        Replaces the internal deque with a fresh empty one so that
        concurrent appenders are blocked for only the duration of the
        swap (a single attribute assignment).

        Returns
        -------
        list
            All spans that were in the buffer at the time of the call.
            Returns an empty list when the buffer is empty.
        """
        with self._drain_lock:
            old = self._deque
            self._deque = deque(maxlen=self._maxlen)
        return list(old)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current number of items in the buffer.

        Note: this is a snapshot; the value may change immediately after
        being read in a multi-threaded context.
        """
        return len(self._deque)

    @property
    def maxlen(self) -> int:
        """Return the configured maximum buffer capacity."""
        return self._maxlen

    @property
    def is_full(self) -> bool:
        """Return True when the buffer is at capacity."""
        return len(self._deque) >= self._maxlen

    def __repr__(self) -> str:
        return f"RingBuffer(maxlen={self._maxlen}, current={len(self._deque)})"
