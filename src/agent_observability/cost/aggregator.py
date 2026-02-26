"""CostAggregator — aggregate costs by agent, model, provider, task, time period."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from agent_observability.cost.tracker import CostRecord, CostSummary, CostTracker


class TimePeriod(str, Enum):
    """Predefined time window for cost aggregation."""

    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    ALL = "all"


def _period_start(period: TimePeriod) -> Optional[float]:
    """Return the Unix timestamp for the start of *period*, or ``None`` for ALL."""
    now = time.time()
    if period is TimePeriod.TODAY:
        import datetime

        today = datetime.datetime.now(tz=datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return today.timestamp()
    if period is TimePeriod.WEEK:
        return now - 7 * 86_400
    if period is TimePeriod.MONTH:
        return now - 30 * 86_400
    return None


@dataclass
class AggregatedCosts:
    """Multi-dimensional cost breakdown."""

    period: str
    total_usd: float
    total_tokens: int
    by_agent: dict[str, float]
    by_model: dict[str, float]
    by_provider: dict[str, float]
    by_task: dict[str, float]
    record_count: int


class CostAggregator:
    """Aggregate cost data from a :class:`CostTracker` across multiple dimensions.

    Parameters
    ----------
    tracker:
        The :class:`CostTracker` instance to read from.
    """

    def __init__(self, tracker: CostTracker) -> None:
        self._tracker = tracker

    def aggregate(
        self,
        period: TimePeriod = TimePeriod.ALL,
        agent_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AggregatedCosts:
        """Return a multi-dimensional cost breakdown.

        Parameters
        ----------
        period:
            Time window to include.
        agent_id:
            Filter to a single agent.
        provider:
            Filter to a single provider.
        model:
            Filter to a single model.
        """
        since = _period_start(period)
        records = self._tracker.records(
            agent_id=agent_id,
            provider=provider,
            model=model,
            since=since,
        )

        by_agent: dict[str, float] = {}
        by_model: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        by_task: dict[str, float] = {}
        total_usd = 0.0
        total_tokens = 0

        for rec in records:
            total_usd += rec.cost_usd
            total_tokens += rec.input_tokens + rec.output_tokens
            by_agent[rec.agent_id] = by_agent.get(rec.agent_id, 0.0) + rec.cost_usd
            by_model[rec.model] = by_model.get(rec.model, 0.0) + rec.cost_usd
            by_provider[rec.provider] = by_provider.get(rec.provider, 0.0) + rec.cost_usd
            task_key = rec.task_id or "(no task)"
            by_task[task_key] = by_task.get(task_key, 0.0) + rec.cost_usd

        return AggregatedCosts(
            period=period.value,
            total_usd=round(total_usd, 8),
            total_tokens=total_tokens,
            by_agent=by_agent,
            by_model=by_model,
            by_provider=by_provider,
            by_task=by_task,
            record_count=len(records),
        )

    def top_agents(
        self,
        n: int = 10,
        period: TimePeriod = TimePeriod.ALL,
    ) -> list[tuple[str, float]]:
        """Return the top-*n* agents by cost, descending."""
        agg = self.aggregate(period=period)
        return sorted(agg.by_agent.items(), key=lambda x: x[1], reverse=True)[:n]

    def top_models(
        self,
        n: int = 10,
        period: TimePeriod = TimePeriod.ALL,
    ) -> list[tuple[str, float]]:
        """Return the top-*n* models by cost, descending."""
        agg = self.aggregate(period=period)
        return sorted(agg.by_model.items(), key=lambda x: x[1], reverse=True)[:n]

    def daily_cost_series(
        self,
        days: int = 30,
        agent_id: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Return a list of ``(YYYY-MM-DD, cost_usd)`` tuples for the past *days*.

        Days with no activity appear with cost 0.0.
        """
        import datetime

        records = self._tracker.records(
            agent_id=agent_id,
            since=time.time() - days * 86_400,
        )

        day_costs: dict[str, float] = {}
        for rec in records:
            day_str = datetime.datetime.fromtimestamp(
                rec.timestamp, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d")
            day_costs[day_str] = day_costs.get(day_str, 0.0) + rec.cost_usd

        # Fill in zero days
        today = datetime.datetime.now(tz=datetime.timezone.utc)
        result: list[tuple[str, float]] = []
        for offset in range(days - 1, -1, -1):
            day = (today - datetime.timedelta(days=offset)).strftime("%Y-%m-%d")
            result.append((day, round(day_costs.get(day, 0.0), 8)))

        return result
