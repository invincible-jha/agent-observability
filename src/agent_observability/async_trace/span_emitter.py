"""Zero-overhead span emitter for the hot path.

SpanEmitter is the primary user-facing API. ``emit()`` places a span dict
into the RingBuffer with no awaits, no locks, and no network I/O. Actual
export happens in the BackgroundFlusher on a separate timer thread.

Usage
-----
>>> config = AsyncTraceConfig()
>>> emitter = SpanEmitter.create(exporter=my_callback, config=config)
>>> emitter.start()

>>> # Hot path — zero blocking
>>> emitter.emit("llm.call", {"model": "gpt-4o", "tokens": 1024})

>>> # Context manager — records start/end timestamps automatically
>>> with emitter.span("tool.execution") as span_ctx:
...     span_ctx["tool"] = "web_search"
...     result = do_work()

>>> emitter.stop()
"""
from __future__ import annotations

import contextlib
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Generator

from agent_observability.async_trace.background_flusher import BackgroundFlusher
from agent_observability.async_trace.config import AsyncTraceConfig
from agent_observability.async_trace.ring_buffer import RingBuffer, SpanRecord


class SpanEmitter:
    """Zero-allocation span emission to a ring buffer.

    Parameters
    ----------
    ring_buffer:
        The RingBuffer that receives emitted spans.
    flusher:
        BackgroundFlusher instance responsible for exporting batches.

    Notes
    -----
    ``emit()`` takes the GIL-protected ``deque.append`` path — no Python
    lock acquisition occurs in the hot path. Allocation is limited to one
    ``dict`` per span.
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        flusher: BackgroundFlusher,
    ) -> None:
        self._buffer = ring_buffer
        self._flusher = flusher

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        exporter: Callable[[list[SpanRecord]], None],
        config: AsyncTraceConfig | None = None,
    ) -> "SpanEmitter":
        """Create a fully configured SpanEmitter.

        Parameters
        ----------
        exporter:
            Callback that receives a list of span dicts for export.
        config:
            Optional configuration. Defaults to AsyncTraceConfig().

        Returns
        -------
        SpanEmitter
            A configured but not-yet-started emitter. Call ``start()``
            to begin background flushing.
        """
        cfg = config or AsyncTraceConfig()
        ring_buffer = RingBuffer(maxlen=cfg.buffer_size)
        flusher = BackgroundFlusher(
            ring_buffer=ring_buffer,
            exporter=exporter,
            flush_interval_seconds=cfg.flush_interval_seconds,
            max_batch_size=cfg.max_batch_size,
        )
        return cls(ring_buffer=ring_buffer, flusher=flusher)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background flush timer."""
        self._flusher.start()

    def stop(self) -> None:
        """Stop the background flush timer and flush remaining spans."""
        self._flusher.stop()

    def flush(self) -> int:
        """Manually flush all buffered spans.

        Returns
        -------
        int
            Number of spans exported.
        """
        return self._flusher.flush()

    # ------------------------------------------------------------------
    # Hot path — zero-overhead emit
    # ------------------------------------------------------------------

    def emit(
        self,
        span_name: str,
        attributes: dict[str, object] | None = None,
    ) -> None:
        """Emit a span to the ring buffer.

        This is the primary hot-path method. It performs no I/O, no locks,
        and no awaits. A span dict is constructed and appended to the ring
        buffer in O(1) time.

        Parameters
        ----------
        span_name:
            Human-readable name for the span (e.g. ``"llm.call"``).
        attributes:
            Optional key-value attributes to include in the span.
        """
        span: SpanRecord = {
            "span_id": uuid.uuid4().hex,
            "name": span_name,
            "start_time_ns": time.time_ns(),
            "attributes": attributes or {},
        }
        self._buffer.append(span)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        operation_name: str,
        attributes: dict[str, object] | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Context manager that records a span with start and end timestamps.

        The yielded dict is the mutable span record. Callers may add
        attributes inside the ``with`` block before it is emitted on exit.

        Parameters
        ----------
        operation_name:
            Name for the span operation.
        attributes:
            Optional initial attributes for the span.

        Yields
        ------
        SpanRecord
            The mutable span record. Modifications are captured.

        Examples
        --------
        >>> with emitter.span("tool.call") as span_ctx:
        ...     span_ctx["tool"] = "calculator"
        ...     result = calculate()
        ...     span_ctx["success"] = True
        """
        span_record: SpanRecord = {
            "span_id": uuid.uuid4().hex,
            "name": operation_name,
            "start_time_ns": time.time_ns(),
            "attributes": dict(attributes) if attributes else {},
        }
        error: BaseException | None = None
        try:
            yield span_record
        except BaseException as exc:
            error = exc
            span_record["error"] = str(exc)
            span_record["error_type"] = type(exc).__name__
            raise
        finally:
            span_record["end_time_ns"] = time.time_ns()
            span_record["duration_ns"] = (
                span_record["end_time_ns"] - span_record["start_time_ns"]
            )
            span_record["status"] = "error" if error else "ok"
            self._buffer.append(span_record)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Return the current number of spans waiting in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"SpanEmitter(buffer={self._buffer!r}, flusher={self._flusher!r})"
        )
