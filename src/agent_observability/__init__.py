"""agent-observability — OpenTelemetry-native agent tracing, cost attribution, and drift detection.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Example
-------
>>> import agent_observability
>>> agent_observability.__version__
'0.1.0'
"""
from __future__ import annotations

__version__: str = "0.1.0"

from agent_observability.convenience import Tracer

# Prompts
from agent_observability.prompts import (
    PromptDiff,
    PromptNotFoundError,
    PromptRegistry,
    PromptRegistryError,
    PromptTemplate,
    PromptVersion,
    PromptVersionNotFoundError,
)

# Decision tracing (Phase 5)
from agent_observability.decisions import (
    DecisionQuery,
    DecisionQueryResult,
    DecisionSpan,
    DecisionStatus,
    DecisionTracker,
)

# Multi-agent trace correlation (Phase 6)
from agent_observability.correlation import (
    BaggageItem,
    CorrelatedSpan,
    CorrelationContext,
    SpanRelationship,
    TraceCorrelator,
    TraceTree,
)

# Hierarchical cost attribution (Phase 6)
from agent_observability.cost_attribution import (
    AttributionNode,
    CostRollup,
    HierarchicalCostAttributor,
)

# Live replay debugger (Phase 5)
from agent_observability.replay import (
    DiffResult,
    TraceDiff,
    TracePlayer,
    TraceRecorder,
)

__all__ = [
    "__version__",
    "Tracer",
    # Prompts
    "PromptDiff",
    "PromptNotFoundError",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptTemplate",
    "PromptVersion",
    "PromptVersionNotFoundError",
    # Decision tracing
    "DecisionQuery",
    "DecisionQueryResult",
    "DecisionSpan",
    "DecisionStatus",
    "DecisionTracker",
    # Trace correlation
    "BaggageItem",
    "CorrelatedSpan",
    "CorrelationContext",
    "SpanRelationship",
    "TraceCorrelator",
    "TraceTree",
    # Cost attribution
    "AttributionNode",
    "CostRollup",
    "HierarchicalCostAttributor",
    # Replay debugger
    "DiffResult",
    "TraceDiff",
    "TracePlayer",
    "TraceRecorder",
]
