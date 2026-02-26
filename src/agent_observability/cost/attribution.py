"""CostAttributor — simple per-span cost attribution with thread safety.

Provides a lightweight façade over the richer :mod:`agent_observability.cost.tracker`
system for callers that only need span-level cost attribution without the
full provider/model/session breakdown.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ── Provider pricing table ─────────────────────────────────────────────────────
# Tuple layout: (input_cost_per_1k_tokens, output_cost_per_1k_tokens) in USD.
# Prices approximate as of early 2026.

PROVIDER_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-haiku-4-5": (0.0008, 0.004),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-opus-4": (0.015, 0.075),
    # OpenAI
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.010, 0.030),
    # Google
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-1.5-pro": (0.00125, 0.005),
    # Mistral
    "mistral-large": (0.003, 0.009),
    # Meta / community
    "llama-3-70b": (0.00059, 0.00079),
    # DeepSeek
    "deepseek-v3": (0.00027, 0.0011),
}


@dataclass
class CostRecord:
    """A single recorded LLM cost event keyed by span_id."""

    span_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class CostSummary:
    """Aggregated cost data over a collection of :class:`CostRecord` instances."""

    total_cost: float
    by_model: dict[str, float]
    by_agent: dict[str, float]
    record_count: int


class CostAttributor:
    """Thread-safe per-span cost attribution.

    Example
    -------
    >>> attributor = CostAttributor()
    >>> record = attributor.record("span-abc", "gpt-4o", 1000, 200)
    >>> print(record.cost_usd)
    >>> summary = attributor.summary()
    """

    def __init__(self) -> None:
        self._records: list[CostRecord] = []
        self._lock = threading.Lock()

    # ── Pricing ───────────────────────────────────────────────────────────────

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return USD cost for the given token counts.

        Uses :data:`PROVIDER_PRICING` for a lookup; falls back to 0.0 for
        unknown models (callers should register custom pricing if needed).

        Parameters
        ----------
        model:
            Model name as it appears in :data:`PROVIDER_PRICING`.
        input_tokens:
            Number of prompt/input tokens.
        output_tokens:
            Number of completion/output tokens.

        Returns
        -------
        Computed USD cost rounded to 8 decimal places.
        """
        pricing = self._lookup_pricing(model)
        if pricing is None:
            return 0.0

        input_cost_per_k, output_cost_per_k = pricing
        cost = (input_tokens / 1_000) * input_cost_per_k + (output_tokens / 1_000) * output_cost_per_k
        return round(cost, 8)

    def _lookup_pricing(self, model: str) -> Optional[tuple[float, float]]:
        """Look up pricing with prefix-match fallback."""
        if model in PROVIDER_PRICING:
            return PROVIDER_PRICING[model]
        # Prefix match — allows "gpt-4o-2024-11-20" to resolve to "gpt-4o"
        for key, pricing in PROVIDER_PRICING.items():
            if model.startswith(key) or key.startswith(model):
                return pricing
        return None

    # ── Record ────────────────────────────────────────────────────────────────

    def record(
        self,
        span_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Record cost for a span and return the :class:`CostRecord`.

        Parameters
        ----------
        span_id:
            OTel span identifier (hex string or any unique label).
        model:
            Model name.
        input_tokens:
            Number of prompt tokens.
        output_tokens:
            Number of completion tokens.

        Returns
        -------
        The created :class:`CostRecord`.
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        rec = CostRecord(
            span_id=span_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self._records.append(rec)
        return rec

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, agent_id: Optional[str] = None) -> CostSummary:
        """Return aggregated cost summary.

        Parameters
        ----------
        agent_id:
            Ignored in this simple attributor (there is no per-agent partitioning
            at this level — use :class:`~agent_observability.cost.CostTracker`
            for full agent-level breakdowns).  Parameter kept for API compatibility.

        Returns
        -------
        :class:`CostSummary` with ``total_cost``, ``by_model``, ``by_agent``,
        and ``record_count``.
        """
        with self._lock:
            snapshot = list(self._records)

        by_model: dict[str, float] = {}
        by_agent: dict[str, float] = {}
        total_cost = 0.0

        for rec in snapshot:
            total_cost += rec.cost_usd
            by_model[rec.model] = by_model.get(rec.model, 0.0) + rec.cost_usd

        # agent breakdown not available at this level — provide empty dict
        return CostSummary(
            total_cost=round(total_cost, 8),
            by_model=by_model,
            by_agent=by_agent,
            record_count=len(snapshot),
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()
