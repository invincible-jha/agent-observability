"""Configuration for zero-overhead async tracing.

AsyncTraceConfig holds all tunable parameters for the ring buffer,
background flusher, and span emitter subsystems.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AsyncTraceConfig:
    """Configuration for the async trace pipeline.

    Parameters
    ----------
    buffer_size:
        Maximum number of spans the ring buffer holds before the oldest
        entries are dropped. Should be sized to absorb the peak emission
        rate times the flush interval.
    flush_interval_seconds:
        How often (in seconds) the background flusher drains the buffer
        and calls the exporter callback. Lower values reduce end-to-end
        latency; higher values batch more efficiently.
    max_batch_size:
        Maximum number of spans sent to the exporter per flush cycle.
        Excess spans beyond this limit are carried over to the next cycle.
    exporter_retry_on_error:
        When True the flusher retags failed spans for retry on the next
        interval. When False, failed batches are silently dropped.
    """

    buffer_size: int = 10_000
    flush_interval_seconds: float = 5.0
    max_batch_size: int = 1_000
    exporter_retry_on_error: bool = False

    def __post_init__(self) -> None:
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")
        if self.flush_interval_seconds <= 0:
            raise ValueError(
                f"flush_interval_seconds must be positive, got {self.flush_interval_seconds}"
            )
        if self.max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive, got {self.max_batch_size}"
            )
