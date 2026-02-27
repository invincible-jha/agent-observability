"""Route handler functions for the agent-observability HTTP server.

Each function accepts parsed request data and returns a tuple of
(status_code, response_dict). The HTTP handler in app.py calls these
functions and serializes the results to JSON.
"""
from __future__ import annotations

import time
import uuid

from agent_observability.cost.tracker import CostTracker
from agent_observability.server.models import (
    CostSummaryResponse,
    CreateTraceRequest,
    ErrorResponse,
    HealthResponse,
    TraceResponse,
)


# Module-level shared state (reset between tests via reset_state())
_cost_tracker: CostTracker = CostTracker()
_trace_store: dict[str, dict[str, object]] = {}


def reset_state() -> None:
    """Reset all shared state — used in tests and for clean restarts."""
    global _cost_tracker, _trace_store
    _cost_tracker = CostTracker()
    _trace_store = {}


def get_cost_tracker() -> CostTracker:
    """Return the module-level CostTracker instance."""
    return _cost_tracker


def handle_create_trace(body: dict[str, object]) -> tuple[int, dict[str, object]]:
    """Handle POST /traces.

    Validates the request, records a cost entry, and stores a trace record.

    Parameters
    ----------
    body:
        Parsed JSON request body.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    try:
        request = CreateTraceRequest.model_validate(body)
    except Exception as exc:
        return 422, ErrorResponse(error="Validation error", detail=str(exc)).model_dump()

    trace_id = str(uuid.uuid4())
    timestamp = time.time()

    # Record cost if provider/model are present
    if request.provider and request.model:
        _cost_tracker.record(
            provider=request.provider,
            model=request.model,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            cached_input_tokens=request.cached_input_tokens,
            cost_usd=request.cost_usd,
            agent_id=request.agent_id,
            session_id=request.session_id,
            task_id=request.task_id,
            operation=request.operation,
            trace_id=trace_id,
            tags=request.tags,
        )

    trace_record: dict[str, object] = {
        "trace_id": trace_id,
        "agent_id": request.agent_id,
        "session_id": request.session_id,
        "task_id": request.task_id,
        "service_name": request.service_name,
        "provider": request.provider,
        "model": request.model,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "cached_input_tokens": request.cached_input_tokens,
        "cost_usd": request.cost_usd or 0.0,
        "operation": request.operation,
        "tags": request.tags,
        "timestamp": timestamp,
        "spans": [],
    }
    _trace_store[trace_id] = trace_record

    response = TraceResponse(**trace_record)
    return 201, response.model_dump()


def handle_get_trace(trace_id: str) -> tuple[int, dict[str, object]]:
    """Handle GET /traces/{id}.

    Parameters
    ----------
    trace_id:
        The trace identifier from the URL path.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    trace_record = _trace_store.get(trace_id)
    if trace_record is None:
        return 404, ErrorResponse(
            error="Not found",
            detail=f"Trace {trace_id!r} not found.",
        ).model_dump()

    response = TraceResponse(**trace_record)
    return 200, response.model_dump()


def handle_list_traces(
    agent_id: str | None = None,
    limit: int = 100,
) -> tuple[int, dict[str, object]]:
    """Handle GET /traces with optional filtering.

    Parameters
    ----------
    agent_id:
        Optional agent_id filter.
    limit:
        Maximum number of traces to return.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    records = list(_trace_store.values())
    if agent_id:
        records = [r for r in records if r.get("agent_id") == agent_id]
    records = records[:limit]

    return 200, {
        "traces": [TraceResponse(**r).model_dump() for r in records],
        "total": len(records),
    }


def handle_get_costs(
    agent_id: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> tuple[int, dict[str, object]]:
    """Handle GET /costs.

    Returns an aggregated cost summary, optionally filtered.

    Parameters
    ----------
    agent_id:
        Optional agent_id filter.
    since:
        Optional Unix timestamp lower bound.
    until:
        Optional Unix timestamp upper bound.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    summary = _cost_tracker.summary(agent_id=agent_id, since=since, until=until)
    response = CostSummaryResponse(
        total_cost_usd=summary.total_cost_usd,
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        total_tokens=summary.total_tokens,
        record_count=summary.record_count,
        by_model=summary.by_model,
        by_provider=summary.by_provider,
        by_agent=summary.by_agent,
        by_operation=summary.by_operation,
    )
    return 200, response.model_dump()


def handle_health() -> tuple[int, dict[str, object]]:
    """Handle GET /health.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    response = HealthResponse(trace_count=len(_trace_store))
    return 200, response.model_dump()


__all__ = [
    "reset_state",
    "get_cost_tracker",
    "handle_create_trace",
    "handle_get_trace",
    "handle_list_traces",
    "handle_get_costs",
    "handle_health",
]
