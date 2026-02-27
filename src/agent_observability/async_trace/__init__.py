"""async_trace — zero-overhead async span tracing for production agents.

This subpackage provides a ring-buffer-based span pipeline that adds
negligible latency to the hot path. Spans are emitted synchronously into
a fixed-size ring buffer; a background timer thread drains the buffer and
calls the exporter asynchronously.

Public surface
--------------
- :class:`RingBuffer` — lock-free O(1) append, atomic drain
- :class:`BackgroundFlusher` — timer-driven drain → exporter
- :class:`SpanEmitter` — user-facing emit / context manager API
- :class:`AsyncTraceConfig` — configuration dataclass

Quick start
-----------
::

    from agent_observability.async_trace import SpanEmitter, AsyncTraceConfig

    def my_exporter(spans):
        for span in spans:
            print(span)

    config = AsyncTraceConfig(buffer_size=5_000, flush_interval_seconds=1.0)
    emitter = SpanEmitter.create(exporter=my_exporter, config=config)
    emitter.start()

    # Hot path — zero blocking
    emitter.emit("llm.call", {"model": "gpt-4o", "tokens": 512})

    # Context manager
    with emitter.span("tool.execution") as span_ctx:
        span_ctx["tool"] = "web_search"

    emitter.stop()
"""
from __future__ import annotations

from agent_observability.async_trace.background_flusher import BackgroundFlusher
from agent_observability.async_trace.config import AsyncTraceConfig
from agent_observability.async_trace.ring_buffer import RingBuffer
from agent_observability.async_trace.span_emitter import SpanEmitter

__all__ = [
    "AsyncTraceConfig",
    "BackgroundFlusher",
    "RingBuffer",
    "SpanEmitter",
]
