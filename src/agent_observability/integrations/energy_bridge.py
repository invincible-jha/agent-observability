"""Energy budget bridge — forwards observability cost data to an energy budget tracker.

This module connects agent-observability span cost data to an external
agent-energy-budget tracker via Protocol-based dependency injection.
No direct import of agent-energy-budget is performed; the bridge accepts
any object that satisfies :class:`EnergyTrackerProtocol`.

Install both packages to enable the bridge::

    pip install aumos-agent-observability agent-energy-budget

Usage
-----
::

    from agent_observability.integrations.energy_bridge import (
        EnergyBudgetBridge,
        CostAttribution,
    )
    from agent_energy_budget.budget.tracker import BudgetTracker
    from agent_energy_budget.budget.config import BudgetConfig

    tracker = BudgetTracker(BudgetConfig(agent_id="my-agent", daily_limit=1.0))
    bridge = EnergyBudgetBridge(energy_tracker=tracker)

    # Forward a cost event directly
    bridge.on_cost_event(model="gpt-4o", tokens=1200, cost=0.003)

    # Or attach to an AgentSpan that already has cost/token attributes
    with tracer.llm_call(model="gpt-4o") as span:
        span.set_tokens(1000, 200)
        span.set_cost(0.0025)
    bridge.on_span_end(span)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Span attribute key constants (mirrors conventions.py to avoid circular import)
# ---------------------------------------------------------------------------

_ATTR_LLM_MODEL: str = "llm.model"
_ATTR_LLM_TOKENS_INPUT: str = "llm.tokens.input"
_ATTR_LLM_TOKENS_OUTPUT: str = "llm.tokens.output"
_ATTR_LLM_COST_USD: str = "llm.cost.usd"
_ATTR_AGENT_SPAN_KIND: str = "agent.span.kind"
_LLM_CALL_KIND: str = "llm_call"

# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class CostAttribution:
    """Cost data extracted from a completed span or supplied directly.

    Parameters
    ----------
    model:
        LLM model identifier.
    input_tokens:
        Number of input/prompt tokens.
    output_tokens:
        Number of output/completion tokens.
    estimated_cost_usd:
        Estimated cost in US dollars.
    timestamp:
        UTC timestamp when the attribution was recorded.
    """

    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


@dataclass
class BudgetStatus:
    """Snapshot of a budget tracker's current state.

    This type is used as the return value of
    :meth:`EnergyTrackerProtocol.check_budget` so callers can inspect
    consumption without importing agent-energy-budget directly.

    Parameters
    ----------
    budget_id:
        Identifier for the budget (typically agent_id).
    allocated_usd:
        Total USD budget allocated.
    consumed_usd:
        USD consumed so far.
    remaining_usd:
        USD remaining (allocated - consumed).
    usage_percentage:
        Percentage of the budget consumed (0–100+).
    is_exceeded:
        True when consumed_usd > allocated_usd.
    """

    budget_id: str
    allocated_usd: float
    consumed_usd: float
    remaining_usd: float
    usage_percentage: float
    is_exceeded: bool


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------


@runtime_checkable
class EnergyTrackerProtocol(Protocol):
    """Structural interface for any energy/budget tracker.

    Any object exposing these two methods can be used as the
    ``energy_tracker`` argument to :class:`EnergyBudgetBridge`.
    """

    def record_usage(
        self,
        model: str,
        tokens: int,
        cost: float,
    ) -> None:
        """Record a single LLM usage event.

        Parameters
        ----------
        model:
            The model identifier.
        tokens:
            Total tokens consumed (input + output).
        cost:
            Estimated cost in USD.
        """
        ...  # pragma: no cover

    def check_budget(self) -> BudgetStatus:
        """Return the current budget status snapshot.

        Returns
        -------
        BudgetStatus
            Current allocation, consumption, and excess information.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Bridge implementation
# ---------------------------------------------------------------------------


class EnergyBudgetBridge:
    """Forwards observability span cost data to an energy budget tracker.

    The bridge is deliberately decoupled from agent-energy-budget via
    :class:`EnergyTrackerProtocol`.  Passing ``energy_tracker=None`` is
    valid; all calls become no-ops, which preserves existing behaviour in
    environments where agent-energy-budget is not installed.

    Parameters
    ----------
    energy_tracker:
        Any object satisfying :class:`EnergyTrackerProtocol`, or ``None``
        to operate in no-op mode.

    Examples
    --------
    ::

        bridge = EnergyBudgetBridge(energy_tracker=my_tracker)
        bridge.on_cost_event("gpt-4o", tokens=1500, cost=0.003)
    """

    def __init__(
        self,
        energy_tracker: Optional[EnergyTrackerProtocol] = None,
    ) -> None:
        self._tracker = energy_tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_span_end(self, span: object) -> Optional[CostAttribution]:
        """Extract cost data from a finished span and forward it to the tracker.

        Attempts to read ``llm.model``, ``llm.tokens.input``,
        ``llm.tokens.output``, and ``llm.cost.usd`` from *span*.
        Only spans whose ``agent.span.kind`` attribute equals
        ``"llm_call"`` are forwarded (others are silently skipped).

        Parameters
        ----------
        span:
            A finished :class:`~agent_observability.spans.types.AgentSpan`
            or any object with ``_span._attributes`` (the internal
            ``_NoOpSpan``-based representation used in tests).

        Returns
        -------
        CostAttribution or None
            The extracted attribution, or ``None`` if the span carries no
            cost data or is not an LLM call span.
        """
        attribution = self._extract_attribution(span)
        if attribution is None:
            return None

        self._forward_attribution(attribution)
        return attribution

    def on_cost_event(
        self,
        model: str,
        tokens: int,
        cost: float,
    ) -> Optional[BudgetStatus]:
        """Record a raw cost event and return current budget status.

        This method is useful when cost data is available without a span
        (e.g. from a middleware or callback hook).

        Parameters
        ----------
        model:
            The model identifier.
        tokens:
            Total tokens consumed.
        cost:
            Estimated cost in USD.

        Returns
        -------
        BudgetStatus or None
            Current budget status from the tracker, or ``None`` when
            no tracker is configured.
        """
        if self._tracker is None:
            logger.debug(
                "EnergyBudgetBridge: no tracker configured; cost event dropped "
                "(model=%s tokens=%d cost=%.8f)",
                model,
                tokens,
                cost,
            )
            return None

        try:
            self._tracker.record_usage(model=model, tokens=tokens, cost=cost)
            status = self._tracker.check_budget()
            logger.debug(
                "EnergyBudgetBridge: forwarded cost event "
                "(model=%s tokens=%d cost=%.8f remaining=%.8f exceeded=%s)",
                model,
                tokens,
                cost,
                status.remaining_usd,
                status.is_exceeded,
            )
            return status
        except Exception as exc:
            logger.warning(
                "EnergyBudgetBridge: tracker call failed: %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_attribution(self, span: object) -> Optional[CostAttribution]:
        """Pull cost attributes from a span-like object.

        Supports both real OTel spans (via ``_span._attributes``) and the
        project's internal ``_NoOpSpan``.  Returns ``None`` when the span
        carries insufficient data.
        """
        attributes = self._get_span_attributes(span)
        if attributes is None:
            return None

        span_kind = attributes.get(_ATTR_AGENT_SPAN_KIND)
        if span_kind != _LLM_CALL_KIND:
            logger.debug(
                "EnergyBudgetBridge.on_span_end: skipping span kind=%r",
                span_kind,
            )
            return None

        model = str(attributes.get(_ATTR_LLM_MODEL, ""))
        input_tokens = int(attributes.get(_ATTR_LLM_TOKENS_INPUT, 0))
        output_tokens = int(attributes.get(_ATTR_LLM_TOKENS_OUTPUT, 0))
        cost_usd = float(attributes.get(_ATTR_LLM_COST_USD, 0.0))

        if not model and cost_usd == 0.0:
            logger.debug(
                "EnergyBudgetBridge.on_span_end: span has no cost data; skipping"
            )
            return None

        return CostAttribution(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost_usd,
        )

    def _get_span_attributes(
        self,
        span: object,
    ) -> Optional[dict[str, str | int | float | bool]]:
        """Extract the attributes dict from a span-like object.

        Checks for the ``_span._attributes`` pattern used by the internal
        ``_NoOpSpan``/``AgentSpan`` wrappers, then falls back to a direct
        ``_attributes`` attribute.
        """
        # AgentSpan wraps an inner _span
        inner_span = getattr(span, "_span", None)
        if inner_span is not None:
            attrs = getattr(inner_span, "_attributes", None)
            if isinstance(attrs, dict):
                return attrs  # type: ignore[return-value]

        # Direct _attributes (e.g. a _NoOpSpan used directly)
        attrs = getattr(span, "_attributes", None)
        if isinstance(attrs, dict):
            return attrs  # type: ignore[return-value]

        return None

    def _forward_attribution(self, attribution: CostAttribution) -> None:
        """Forward a :class:`CostAttribution` to the tracker."""
        if self._tracker is None:
            return

        total_tokens = attribution.input_tokens + attribution.output_tokens
        try:
            self._tracker.record_usage(
                model=attribution.model,
                tokens=total_tokens,
                cost=attribution.estimated_cost_usd,
            )
        except Exception as exc:
            logger.warning(
                "EnergyBudgetBridge: tracker.record_usage failed: %s", exc
            )
