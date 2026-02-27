"""Timer-based background flusher for the ring buffer.

BackgroundFlusher runs a repeating timer thread that periodically drains
the RingBuffer and forwards batches to a caller-supplied exporter callback.
Exporter errors are caught and logged so they never propagate to the
application thread.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from agent_observability.async_trace.ring_buffer import RingBuffer, SpanRecord

logger = logging.getLogger(__name__)

ExporterCallback = Callable[[list[SpanRecord]], None]


class BackgroundFlusher:
    """Periodically drains a RingBuffer and calls an exporter callback.

    Parameters
    ----------
    ring_buffer:
        The buffer to drain.
    exporter:
        A callable that receives a list of spans and exports them. Called
        from the timer thread — must be thread-safe.
    flush_interval_seconds:
        Seconds between automatic flush cycles (default 5.0).
    max_batch_size:
        Maximum number of spans per exporter call. Remaining items are
        carried over to the next cycle if this limit is reached (no data
        is lost — items stay in the already-drained list and are passed
        in subsequent calls within the same flush cycle).

    Examples
    --------
    >>> received = []
    >>> flusher = BackgroundFlusher(RingBuffer(100), received.extend, flush_interval_seconds=0.05)
    >>> flusher.start()
    >>> # emit spans to the buffer …
    >>> flusher.stop()
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        exporter: ExporterCallback,
        flush_interval_seconds: float = 5.0,
        max_batch_size: int = 1_000,
    ) -> None:
        self._buffer = ring_buffer
        self._exporter = exporter
        self._interval = flush_interval_seconds
        self._max_batch_size = max_batch_size
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background flush timer.

        Calling start() when already running is a no-op.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
        self._schedule_next()

    def stop(self) -> None:
        """Stop the background flush timer and perform a final flush.

        Blocks briefly to cancel any pending timer and drain remaining
        spans from the buffer.
        """
        with self._lock:
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        # Final drain so no spans are lost on shutdown
        self.flush()

    def flush(self) -> int:
        """Manually drain the buffer and call the exporter immediately.

        Returns
        -------
        int
            Number of spans exported in this call.
        """
        spans = self._buffer.drain()
        if not spans:
            return 0
        total = 0
        offset = 0
        while offset < len(spans):
            batch = spans[offset : offset + self._max_batch_size]
            offset += self._max_batch_size
            self._call_exporter(batch)
            total += len(batch)
        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Schedule the next timer tick if still running."""
        with self._lock:
            if not self._running:
                return
            self._timer = threading.Timer(self._interval, self._on_timer)
            self._timer.daemon = True
            self._timer.start()

    def _on_timer(self) -> None:
        """Timer callback — drain the buffer and reschedule."""
        try:
            self.flush()
        except Exception:
            logger.exception("Unexpected error in BackgroundFlusher._on_timer")
        finally:
            self._schedule_next()

    def _call_exporter(self, batch: list[SpanRecord]) -> None:
        """Call the exporter with *batch*, swallowing all exceptions."""
        try:
            self._exporter(batch)
        except Exception:
            logger.exception(
                "BackgroundFlusher exporter raised an exception; batch of %d spans dropped",
                len(batch),
            )

    def __repr__(self) -> str:
        return (
            f"BackgroundFlusher("
            f"interval={self._interval}s, "
            f"max_batch={self._max_batch_size}, "
            f"running={self._running})"
        )
