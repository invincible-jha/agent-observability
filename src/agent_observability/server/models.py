"""Pydantic request/response models for the agent-observability HTTP server."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreateTraceRequest(BaseModel):
    """Request body for POST /traces."""

    agent_id: str = ""
    session_id: str = ""
    task_id: str = ""
    service_name: str = "agent"
    provider: str = ""
    model: str = ""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cost_usd: Optional[float] = None
    operation: str = "llm_call"
    tags: dict[str, str] = Field(default_factory=dict)


class SpanData(BaseModel):
    """Represents a single span within a trace response."""

    span_id: str
    name: str
    kind: str = ""
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    """Response body for GET /traces/{id} and POST /traces."""

    trace_id: str
    agent_id: str
    session_id: str
    task_id: str
    service_name: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    operation: str
    tags: dict[str, str] = Field(default_factory=dict)
    timestamp: float
    spans: list[SpanData] = Field(default_factory=list)


class CostSummaryResponse(BaseModel):
    """Response body for GET /costs."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    record_count: int
    by_model: dict[str, float] = Field(default_factory=dict)
    by_provider: dict[str, float] = Field(default_factory=dict)
    by_agent: dict[str, float] = Field(default_factory=dict)
    by_operation: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    service: str = "agent-observability"
    version: str = "0.1.0"
    trace_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str = ""


__all__ = [
    "CreateTraceRequest",
    "SpanData",
    "TraceResponse",
    "CostSummaryResponse",
    "HealthResponse",
    "ErrorResponse",
]
