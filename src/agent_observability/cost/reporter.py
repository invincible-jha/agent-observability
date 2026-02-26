"""CostReporter — generate cost reports in CSV, JSON, and Markdown formats."""
from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Optional

from agent_observability.cost.aggregator import AggregatedCosts, CostAggregator, TimePeriod
from agent_observability.cost.tracker import CostTracker

logger = logging.getLogger(__name__)


class CostReporter:
    """Generate human- and machine-readable cost reports.

    Parameters
    ----------
    tracker:
        The :class:`CostTracker` instance to read raw records from.
    aggregator:
        Optional pre-built :class:`CostAggregator`.  One is created from
        *tracker* if not supplied.
    """

    def __init__(
        self,
        tracker: CostTracker,
        aggregator: Optional[CostAggregator] = None,
    ) -> None:
        self._tracker = tracker
        self._aggregator = aggregator or CostAggregator(tracker)

    # ── Format helpers ────────────────────────────────────────────────────────

    def to_csv(
        self,
        period: TimePeriod = TimePeriod.ALL,
        agent_id: Optional[str] = None,
    ) -> str:
        """Return raw cost records as a CSV string."""
        return self._tracker.export_csv(agent_id=agent_id)

    def to_json(
        self,
        period: TimePeriod = TimePeriod.ALL,
        agent_id: Optional[str] = None,
        indent: int = 2,
    ) -> str:
        """Return an aggregated cost report as a JSON string."""
        agg = self._aggregator.aggregate(period=period, agent_id=agent_id)
        daily = self._aggregator.daily_cost_series(
            days=30 if period != TimePeriod.TODAY else 1,
            agent_id=agent_id,
        )
        payload = {
            "period": agg.period,
            "total_usd": agg.total_usd,
            "total_tokens": agg.total_tokens,
            "record_count": agg.record_count,
            "by_agent": agg.by_agent,
            "by_model": agg.by_model,
            "by_provider": agg.by_provider,
            "by_task": agg.by_task,
            "daily_series": [{"date": d, "cost_usd": c} for d, c in daily],
        }
        return json.dumps(payload, indent=indent)

    def to_markdown(
        self,
        period: TimePeriod = TimePeriod.ALL,
        agent_id: Optional[str] = None,
    ) -> str:
        """Return a human-readable Markdown cost report."""
        agg = self._aggregator.aggregate(period=period, agent_id=agent_id)
        lines: list[str] = [
            f"# Agent Cost Report — {agg.period.upper()}",
            "",
            f"**Total cost:** ${agg.total_usd:.6f} USD",
            f"**Total tokens:** {agg.total_tokens:,}",
            f"**Records:** {agg.record_count:,}",
            "",
        ]

        def _table(header: str, data: dict[str, float]) -> None:
            lines.append(f"## {header}")
            lines.append("")
            lines.append("| Name | Cost (USD) |")
            lines.append("| ---- | ---------- |")
            for name, cost in sorted(data.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {name} | ${cost:.6f} |")
            lines.append("")

        _table("By Model", agg.by_model)
        _table("By Provider", agg.by_provider)
        _table("By Agent", agg.by_agent)
        _table("By Task", agg.by_task)

        return "\n".join(lines)

    # ── Write helpers ─────────────────────────────────────────────────────────

    def write(
        self,
        output_path: str,
        fmt: str = "json",
        period: TimePeriod = TimePeriod.ALL,
        agent_id: Optional[str] = None,
    ) -> None:
        """Write a report to *output_path* in the requested format.

        Parameters
        ----------
        output_path:
            File path to write to.
        fmt:
            One of ``"csv"``, ``"json"``, ``"md"`` (or ``"markdown"``).
        period:
            Time window to include.
        agent_id:
            Filter to a single agent.
        """
        if fmt == "csv":
            content = self.to_csv(period=period, agent_id=agent_id)
        elif fmt == "json":
            content = self.to_json(period=period, agent_id=agent_id)
        elif fmt in ("md", "markdown"):
            content = self.to_markdown(period=period, agent_id=agent_id)
        else:
            raise ValueError(f"Unknown report format: {fmt!r}. Choose csv, json, or md.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("CostReporter: wrote %s report to %s", fmt, output_path)
