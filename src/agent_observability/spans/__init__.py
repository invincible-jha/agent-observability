"""Agent-semantic span types for OpenTelemetry.

Public surface
--------------
- AgentSpanKind — enum of the 8 span kinds
- CostAnnotation — dataclass for token/cost data
- AgentSpan — fluent OTel span wrapper
- AgentTracer — context-manager factory
- SpanEnricher — OTel SpanProcessor that stamps agent metadata
- Semantic convention constants from ``conventions``
"""
from __future__ import annotations

from agent_observability.spans.conventions import (
    AGENT_ENVIRONMENT,
    AGENT_FRAMEWORK,
    AGENT_ID,
    AGENT_NAME,
    AGENT_SESSION_ID,
    AGENT_SPAN_KIND,
    AGENT_VERSION,
    LLM_COST_USD,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TOKENS_INPUT,
    LLM_TOKENS_OUTPUT,
    LLM_TOKENS_TOTAL,
    MEMORY_KEY,
    MEMORY_OPERATION,
    TOOL_NAME,
    TOOL_SUCCESS,
)
from agent_observability.spans.enricher import SpanEnricher
from agent_observability.spans.types import (
    AgentSpan,
    AgentSpanKind,
    AgentTracer,
    CostAnnotation,
)

__all__ = [
    # Span types
    "AgentSpanKind",
    "CostAnnotation",
    "AgentSpan",
    "AgentTracer",
    "SpanEnricher",
    # Frequently used convention keys
    "AGENT_SPAN_KIND",
    "AGENT_ID",
    "AGENT_SESSION_ID",
    "AGENT_FRAMEWORK",
    "AGENT_NAME",
    "AGENT_VERSION",
    "AGENT_ENVIRONMENT",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_TOKENS_INPUT",
    "LLM_TOKENS_OUTPUT",
    "LLM_TOKENS_TOTAL",
    "LLM_COST_USD",
    "TOOL_NAME",
    "TOOL_SUCCESS",
    "MEMORY_KEY",
    "MEMORY_OPERATION",
]
