"""Multi-agent trace correlation — propagate trace context across agent boundaries."""
from __future__ import annotations

from agent_observability.correlation.correlation_context import (
    CorrelationContext,
    BaggageItem,
)
from agent_observability.correlation.trace_correlator import (
    CorrelatedSpan,
    SpanRelationship,
    TraceCorrelator,
    TraceTree,
)

__all__ = [
    "CorrelationContext",
    "BaggageItem",
    "CorrelatedSpan",
    "SpanRelationship",
    "TraceCorrelator",
    "TraceTree",
]
