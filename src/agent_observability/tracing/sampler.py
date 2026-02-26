"""CostAwareSampler — sample traces based on LLM cost signals.

High-cost traces are always sampled (they are most interesting).
Low-cost traces are sampled at a configurable rate to reduce volume.
Error spans are always sampled regardless of cost.

Falls back gracefully when the OTel SDK is not installed.
"""
from __future__ import annotations

import logging
import random
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_OFF,
        ALWAYS_ON,
        Decision,
        ParentBased,
        Sampler,
        SamplingResult,
        StaticSampler,
    )
    from opentelemetry.trace import Link, SpanContext, SpanKind
    from opentelemetry.util.types import Attributes

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False
    Sampler = object  # type: ignore[assignment,misc]
    SamplingResult = None  # type: ignore[assignment]
    Decision = None  # type: ignore[assignment]
    SpanKind = None  # type: ignore[assignment]
    Attributes = None  # type: ignore[assignment]
    SpanContext = None  # type: ignore[assignment]
    Link = None  # type: ignore[assignment]
    ALWAYS_ON = None  # type: ignore[assignment]
    ALWAYS_OFF = None  # type: ignore[assignment]

# Attribute key used to communicate cost hints to the sampler
_COST_HINT_KEY = "llm.cost.usd"
_SPAN_KIND_KEY = "agent.span.kind"


class CostAwareSampler(Sampler):  # type: ignore[misc]
    """Sample based on per-trace cost signals.

    Parameters
    ----------
    high_cost_threshold_usd:
        Traces whose accumulated cost exceeds this value are always sampled.
    low_cost_sample_rate:
        Fraction (0–1) of low-cost traces to keep.  Defaults to ``0.1``
        (10 % of cheap traces).
    always_sample_errors:
        When ``True`` (default), any span that carries an error status is
        unconditionally sampled.
    seed:
        Optional random seed for reproducible sampling in tests.
    """

    def __init__(
        self,
        high_cost_threshold_usd: float = 0.01,
        low_cost_sample_rate: float = 0.1,
        always_sample_errors: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not (0.0 <= low_cost_sample_rate <= 1.0):
            raise ValueError("low_cost_sample_rate must be between 0 and 1")
        self._high_cost_threshold = high_cost_threshold_usd
        self._low_cost_sample_rate = low_cost_sample_rate
        self._always_sample_errors = always_sample_errors
        self._rng = random.Random(seed)

    def should_sample(  # type: ignore[override]
        self,
        parent_context: Optional[object],
        trace_id: int,
        name: str,
        kind: Optional[object] = None,
        attributes: Optional[object] = None,
        links: Optional[Sequence[object]] = None,
        trace_state: Optional[object] = None,
    ) -> object:
        """Return an OTel SamplingResult (or a simple bool in no-SDK mode)."""
        if not _OTEL_SDK_AVAILABLE or SamplingResult is None or Decision is None:
            return self._simple_decision(attributes)

        decision = self._decide(attributes)
        return SamplingResult(
            decision=decision,
            attributes={},
            trace_state=trace_state,
        )

    def _simple_decision(self, attributes: Optional[object]) -> bool:
        """Fallback used when OTel SDK is not installed."""
        cost = self._extract_cost(attributes)
        if cost is None:
            return self._rng.random() < self._low_cost_sample_rate
        if cost >= self._high_cost_threshold:
            return True
        return self._rng.random() < self._low_cost_sample_rate

    def _decide(self, attributes: Optional[object]) -> object:
        assert Decision is not None
        cost = self._extract_cost(attributes)

        # Always sample when cost exceeds threshold
        if cost is not None and cost >= self._high_cost_threshold:
            return Decision.RECORD_AND_SAMPLE

        # Always sample errors when configured
        if self._always_sample_errors and self._has_error(attributes):
            return Decision.RECORD_AND_SAMPLE

        # Probabilistic sampling for low-cost / no-cost traces
        if self._rng.random() < self._low_cost_sample_rate:
            return Decision.RECORD_AND_SAMPLE

        return Decision.DROP

    @staticmethod
    def _extract_cost(attributes: Optional[object]) -> Optional[float]:
        if not isinstance(attributes, dict):
            return None
        raw = attributes.get(_COST_HINT_KEY)
        if isinstance(raw, (int, float)):
            return float(raw)
        return None

    @staticmethod
    def _has_error(attributes: Optional[object]) -> bool:
        if not isinstance(attributes, dict):
            return False
        kind = attributes.get(_SPAN_KIND_KEY, "")
        return str(kind) == "agent_error"

    def get_description(self) -> str:
        return (
            f"CostAwareSampler("
            f"high_cost_threshold={self._high_cost_threshold}, "
            f"low_cost_sample_rate={self._low_cost_sample_rate})"
        )
