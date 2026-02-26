"""CostTracker — per-operation cost tracking with provider pricing.

Thread-safe, in-memory accumulator.  Call :meth:`record` after each LLM
response, then :meth:`summary` to get aggregated totals.
"""
from __future__ import annotations

import csv
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from agent_observability.cost.pricing import estimate_cost

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """A single recorded LLM cost event."""

    timestamp: float
    agent_id: str
    session_id: str
    task_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    operation: str  # e.g. "llm_call", "embedding", etc.
    trace_id: str = ""
    span_id: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class CostSummary:
    """Aggregated cost summary over a set of records."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    record_count: int
    by_model: dict[str, float]
    by_provider: dict[str, float]
    by_agent: dict[str, float]
    by_operation: dict[str, float]


class CostTracker:
    """Thread-safe in-memory cost tracker.

    Parameters
    ----------
    agent_id:
        Default agent_id applied to records when not overridden.
    session_id:
        Default session_id applied to records when not overridden.
    """

    def __init__(self, agent_id: str = "", session_id: str = "") -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._records: list[CostRecord] = []
        self._lock = threading.Lock()

    # ── Record ────────────────────────────────────────────────────────────────

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cost_usd: Optional[float] = None,
        agent_id: str = "",
        session_id: str = "",
        task_id: str = "",
        operation: str = "llm_call",
        trace_id: str = "",
        span_id: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> CostRecord:
        """Record a single LLM usage event.

        Parameters
        ----------
        provider:
            LLM provider (``"openai"``, ``"anthropic"``, …).
        model:
            Model name as recognised by the provider.
        input_tokens:
            Number of prompt/input tokens.
        output_tokens:
            Number of completion/output tokens.
        cached_input_tokens:
            Cached prompt tokens (for providers that charge a reduced rate).
        cost_usd:
            Override the computed cost.  If ``None``, cost is computed from
            the pricing table.
        agent_id:
            Override the tracker-level agent_id for this record.
        session_id:
            Override the tracker-level session_id for this record.
        task_id:
            Task or run identifier.
        operation:
            Semantic label for the operation (``"llm_call"``, ``"embedding"``).
        trace_id:
            OTel trace ID hex string.
        span_id:
            OTel span ID hex string.
        tags:
            Arbitrary key/value metadata.

        Returns
        -------
        The created :class:`CostRecord`.
        """
        if cost_usd is None:
            cost_usd = estimate_cost(
                provider, model, input_tokens, output_tokens, cached_input_tokens
            )

        rec = CostRecord(
            timestamp=time.time(),
            agent_id=agent_id or self._agent_id,
            session_id=session_id or self._session_id,
            task_id=task_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=cost_usd,
            operation=operation,
            trace_id=trace_id,
            span_id=span_id,
            tags=tags or {},
        )

        with self._lock:
            self._records.append(rec)

        return rec

    # ── Query ─────────────────────────────────────────────────────────────────

    def records(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> list[CostRecord]:
        """Return matching records (a snapshot; thread-safe)."""
        with self._lock:
            snapshot = list(self._records)

        return [
            r for r in snapshot
            if (agent_id is None or r.agent_id == agent_id)
            and (session_id is None or r.session_id == session_id)
            and (provider is None or r.provider == provider)
            and (model is None or r.model == model)
            and (since is None or r.timestamp >= since)
            and (until is None or r.timestamp <= until)
        ]

    def summary(
        self,
        agent_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> CostSummary:
        """Return aggregated cost summary for the filtered records."""
        matching = self.records(agent_id=agent_id, since=since, until=until)

        by_model: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        by_agent: dict[str, float] = {}
        by_operation: dict[str, float] = {}
        total_cost = 0.0
        total_input = 0
        total_output = 0

        for rec in matching:
            total_cost += rec.cost_usd
            total_input += rec.input_tokens
            total_output += rec.output_tokens
            by_model[rec.model] = by_model.get(rec.model, 0.0) + rec.cost_usd
            by_provider[rec.provider] = by_provider.get(rec.provider, 0.0) + rec.cost_usd
            by_agent[rec.agent_id] = by_agent.get(rec.agent_id, 0.0) + rec.cost_usd
            by_operation[rec.operation] = by_operation.get(rec.operation, 0.0) + rec.cost_usd

        return CostSummary(
            total_cost_usd=round(total_cost, 8),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            record_count=len(matching),
            by_model=by_model,
            by_provider=by_provider,
            by_agent=by_agent,
            by_operation=by_operation,
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(
        self,
        agent_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> str:
        """Return a CSV string of the matching cost records."""
        matching = self.records(agent_id=agent_id, since=since, until=until)
        buf = io.StringIO()
        fieldnames = [
            "timestamp",
            "agent_id",
            "session_id",
            "task_id",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cost_usd",
            "operation",
            "trace_id",
            "span_id",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in matching:
            writer.writerow(
                {
                    "timestamp": rec.timestamp,
                    "agent_id": rec.agent_id,
                    "session_id": rec.session_id,
                    "task_id": rec.task_id,
                    "provider": rec.provider,
                    "model": rec.model,
                    "input_tokens": rec.input_tokens,
                    "output_tokens": rec.output_tokens,
                    "cached_input_tokens": rec.cached_input_tokens,
                    "cost_usd": rec.cost_usd,
                    "operation": rec.operation,
                    "trace_id": rec.trace_id,
                    "span_id": rec.span_id,
                }
            )
        return buf.getvalue()

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[CostRecord]:
        with self._lock:
            yield from list(self._records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def reset(self) -> None:
        """Clear all recorded data (mainly useful in tests)."""
        with self._lock:
            self._records.clear()
