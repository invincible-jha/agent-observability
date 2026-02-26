"""OTel agent tracing — provider, exporters, context propagation, sampling."""
from __future__ import annotations

from agent_observability.tracing.context import AgentTraceContext
from agent_observability.tracing.exporter import (
    AgentSpanExporter,
    ConsoleSpanExporter,
    JsonLinesExporter,
    build_otlp_exporter,
)
from agent_observability.tracing.propagator import (
    CrossAgentPropagator,
    extract_from_dict,
    inject_into_dict,
)
from agent_observability.tracing.sampler import CostAwareSampler
from agent_observability.tracing.tracer import AgentTracerProvider, setup_tracing

__all__ = [
    "AgentTracerProvider",
    "setup_tracing",
    "AgentSpanExporter",
    "ConsoleSpanExporter",
    "JsonLinesExporter",
    "build_otlp_exporter",
    "AgentTraceContext",
    "CrossAgentPropagator",
    "inject_into_dict",
    "extract_from_dict",
    "CostAwareSampler",
]
