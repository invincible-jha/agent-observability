"""Agent-semantic span types extending OpenTelemetry.

Provides:
- AgentSpanKind — enum of the 8 agent-semantic span kinds
- CostAnnotation — dataclass capturing token and cost data
- AgentSpan — fluent wrapper around an OTel Span
- AgentTracer — context-manager factory for all span kinds
"""
from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Generator, Optional

from agent_observability.spans.conventions import (
    AGENT_ERROR_RECOVERABLE,
    AGENT_ERROR_RETRY_COUNT,
    AGENT_ERROR_TYPE,
    AGENT_FRAMEWORK,
    AGENT_ID,
    AGENT_RUN_ID,
    AGENT_SESSION_ID,
    AGENT_SPAN_KIND,
    AGENT_TASK_ID,
    DELEGATION_STRATEGY,
    DELEGATION_SUCCESS,
    DELEGATION_TARGET_AGENT,
    DELEGATION_TASK_ID,
    HUMAN_APPROVAL_REQUESTED_BY,
    HUMAN_APPROVAL_STATUS,
    HUMAN_APPROVAL_TIMEOUT_SECONDS,
    LLM_COST_USD,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TOKENS_INPUT,
    LLM_TOKENS_OUTPUT,
    LLM_TOKENS_TOTAL,
    MEMORY_BACKEND,
    MEMORY_HIT,
    MEMORY_KEY,
    MEMORY_OPERATION,
    REASONING_CONFIDENCE,
    REASONING_STEP_INDEX,
    REASONING_STEP_TYPE,
    REASONING_STRATEGY,
    TOOL_ERROR_TYPE,
    TOOL_NAME,
    TOOL_SUCCESS,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Optional OTel import ───────────────────────────────────────────────────────
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Span, StatusCode, Tracer

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    otel_trace = None  # type: ignore[assignment]
    Span = None  # type: ignore[assignment,misc]
    StatusCode = None  # type: ignore[assignment]
    Tracer = None  # type: ignore[assignment]


class AgentSpanKind(str, Enum):
    """The 8 semantic span kinds for agent observability."""

    LLM_CALL = "llm_call"
    TOOL_INVOKE = "tool_invoke"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    REASONING_STEP = "reasoning_step"
    AGENT_DELEGATE = "agent_delegate"
    HUMAN_APPROVAL = "human_approval"
    AGENT_ERROR = "agent_error"


@dataclass
class CostAnnotation:
    """Token and cost data associated with an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class _NoOpSpan:
    """Fallback span used when OTel is not installed."""

    _attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    _ended: bool = False

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        self._attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        logger.debug("span.record_exception: %s", exc)

    def set_status(self, status_code: object, description: str = "") -> None:
        pass

    def end(self) -> None:
        self._ended = True


class AgentSpan:
    """Fluent wrapper around an OTel Span (or no-op fallback) adding agent semantics."""

    def __init__(
        self,
        span: "Span | _NoOpSpan",
        kind: AgentSpanKind,
    ) -> None:
        self._span = span
        self._kind = kind
        self._start_time: float = time.monotonic()
        span.set_attribute(AGENT_SPAN_KIND, kind.value)

    # ── Token / cost ──────────────────────────────────────────────────────────

    def set_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: Optional[int] = None,
    ) -> "AgentSpan":
        """Record token counts on the span."""
        self._span.set_attribute(LLM_TOKENS_INPUT, input_tokens)
        self._span.set_attribute(LLM_TOKENS_OUTPUT, output_tokens)
        self._span.set_attribute(
            LLM_TOKENS_TOTAL,
            total_tokens if total_tokens is not None else input_tokens + output_tokens,
        )
        return self

    def set_cost(self, cost_usd: float) -> "AgentSpan":
        """Record USD cost on the span."""
        self._span.set_attribute(LLM_COST_USD, cost_usd)
        return self

    def set_model(self, model: str, provider: str = "") -> "AgentSpan":
        """Record the LLM model and optional provider."""
        self._span.set_attribute(LLM_MODEL, model)
        if provider:
            self._span.set_attribute(LLM_PROVIDER, provider)
        return self

    # ── Tool ──────────────────────────────────────────────────────────────────

    def set_tool(
        self,
        name: str,
        success: bool = True,
        error_type: str = "",
    ) -> "AgentSpan":
        """Record tool invocation details."""
        self._span.set_attribute(TOOL_NAME, name)
        self._span.set_attribute(TOOL_SUCCESS, success)
        if error_type:
            self._span.set_attribute(TOOL_ERROR_TYPE, error_type)
        return self

    # ── Memory ────────────────────────────────────────────────────────────────

    def set_memory_key(
        self,
        key: str,
        operation: str = "read",
        backend: str = "",
        hit: Optional[bool] = None,
    ) -> "AgentSpan":
        """Record memory operation details."""
        self._span.set_attribute(MEMORY_KEY, key)
        self._span.set_attribute(MEMORY_OPERATION, operation)
        if backend:
            self._span.set_attribute(MEMORY_BACKEND, backend)
        if hit is not None:
            self._span.set_attribute(MEMORY_HIT, hit)
        return self

    # ── Reasoning ─────────────────────────────────────────────────────────────

    def set_reasoning(
        self,
        step_index: int = 0,
        step_type: str = "",
        confidence: float = 0.0,
        strategy: str = "",
    ) -> "AgentSpan":
        """Record reasoning step details."""
        self._span.set_attribute(REASONING_STEP_INDEX, step_index)
        if step_type:
            self._span.set_attribute(REASONING_STEP_TYPE, step_type)
        if confidence:
            self._span.set_attribute(REASONING_CONFIDENCE, confidence)
        if strategy:
            self._span.set_attribute(REASONING_STRATEGY, strategy)
        return self

    # ── Delegation ────────────────────────────────────────────────────────────

    def set_delegation(
        self,
        target_agent: str,
        task_id: str = "",
        strategy: str = "",
        success: Optional[bool] = None,
    ) -> "AgentSpan":
        """Record agent delegation details."""
        self._span.set_attribute(DELEGATION_TARGET_AGENT, target_agent)
        if task_id:
            self._span.set_attribute(DELEGATION_TASK_ID, task_id)
        if strategy:
            self._span.set_attribute(DELEGATION_STRATEGY, strategy)
        if success is not None:
            self._span.set_attribute(DELEGATION_SUCCESS, success)
        return self

    # ── Human approval ────────────────────────────────────────────────────────

    def set_human_approval(
        self,
        requested_by: str,
        status: str = "pending",
        timeout_seconds: Optional[int] = None,
    ) -> "AgentSpan":
        """Record human-in-the-loop approval details."""
        self._span.set_attribute(HUMAN_APPROVAL_REQUESTED_BY, requested_by)
        self._span.set_attribute(HUMAN_APPROVAL_STATUS, status)
        if timeout_seconds is not None:
            self._span.set_attribute(HUMAN_APPROVAL_TIMEOUT_SECONDS, timeout_seconds)
        return self

    # ── Error ─────────────────────────────────────────────────────────────────

    def set_error(
        self,
        error_type: str,
        recoverable: bool = True,
        retry_count: int = 0,
        exception: Optional[BaseException] = None,
    ) -> "AgentSpan":
        """Record agent error details and optionally record the exception."""
        self._span.set_attribute(AGENT_ERROR_TYPE, error_type)
        self._span.set_attribute(AGENT_ERROR_RECOVERABLE, recoverable)
        self._span.set_attribute(AGENT_ERROR_RETRY_COUNT, retry_count)
        if exception is not None:
            self._span.record_exception(exception)
            if _OTEL_AVAILABLE and StatusCode is not None:
                self._span.set_status(StatusCode.ERROR, str(exception))  # type: ignore[attr-defined]
        return self

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def finish(self) -> None:
        """End the underlying span."""
        self._span.end()

    def set_attribute(self, key: str, value: str | int | float | bool) -> "AgentSpan":
        """Set an arbitrary attribute on the underlying span."""
        self._span.set_attribute(key, value)
        return self

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since this span was created."""
        return time.monotonic() - self._start_time

    def _stable_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class AgentTracer:
    """Factory that creates AgentSpan context managers for each span kind.

    Parameters
    ----------
    tracer_name:
        Identifies this tracer (typically the agent ID or framework name).
    agent_id:
        Agent identifier attached to every span.
    session_id:
        Session identifier attached to every span.
    framework:
        Framework name (e.g. ``"langchain"``, ``"crewai"``).
    task_id:
        Optional task/run identifier.
    run_id:
        Optional run identifier.
    """

    def __init__(
        self,
        tracer_name: str = "agent-observability",
        agent_id: str = "",
        session_id: str = "",
        framework: str = "",
        task_id: str = "",
        run_id: str = "",
    ) -> None:
        self._tracer_name = tracer_name
        self._agent_id = agent_id
        self._session_id = session_id
        self._framework = framework
        self._task_id = task_id
        self._run_id = run_id

        if _OTEL_AVAILABLE and otel_trace is not None:
            self._tracer: "Tracer | None" = otel_trace.get_tracer(tracer_name)
        else:
            self._tracer = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _start_span(self, name: str, kind: AgentSpanKind) -> AgentSpan:
        if self._tracer is not None and _OTEL_AVAILABLE:
            raw_span = self._tracer.start_span(name)  # type: ignore[union-attr]
        else:
            raw_span = _NoOpSpan()

        agent_span = AgentSpan(raw_span, kind)

        # Attach identity attributes
        if self._agent_id:
            agent_span.set_attribute(AGENT_ID, self._agent_id)
        if self._session_id:
            agent_span.set_attribute(AGENT_SESSION_ID, self._session_id)
        if self._framework:
            agent_span.set_attribute(AGENT_FRAMEWORK, self._framework)
        if self._task_id:
            agent_span.set_attribute(AGENT_TASK_ID, self._task_id)
        if self._run_id:
            agent_span.set_attribute(AGENT_RUN_ID, self._run_id)

        return agent_span

    @contextmanager
    def _managed_span(
        self, name: str, kind: AgentSpanKind
    ) -> Generator[AgentSpan, None, None]:
        agent_span = self._start_span(name, kind)
        try:
            yield agent_span
        except Exception as exc:
            agent_span.set_error(
                error_type=type(exc).__name__,
                recoverable=False,
                exception=exc,
            )
            raise
        finally:
            agent_span.finish()

    # ── Public context managers ───────────────────────────────────────────────

    @contextmanager
    def llm_call(
        self, model: str = "", provider: str = "", name: str = "llm.call"
    ) -> Generator[AgentSpan, None, None]:
        """Trace a single LLM request/response cycle."""
        with self._managed_span(name, AgentSpanKind.LLM_CALL) as span:
            if model:
                span.set_model(model, provider)
            yield span

    @contextmanager
    def tool_invoke(
        self, tool_name: str, name: str = "tool.invoke"
    ) -> Generator[AgentSpan, None, None]:
        """Trace a tool invocation."""
        with self._managed_span(name, AgentSpanKind.TOOL_INVOKE) as span:
            span.set_attribute(TOOL_NAME, tool_name)
            yield span

    @contextmanager
    def memory_read(
        self, key: str, backend: str = "", name: str = "memory.read"
    ) -> Generator[AgentSpan, None, None]:
        """Trace a memory read operation."""
        with self._managed_span(name, AgentSpanKind.MEMORY_READ) as span:
            span.set_memory_key(key, operation="read", backend=backend)
            yield span

    @contextmanager
    def memory_write(
        self, key: str, backend: str = "", name: str = "memory.write"
    ) -> Generator[AgentSpan, None, None]:
        """Trace a memory write operation."""
        with self._managed_span(name, AgentSpanKind.MEMORY_WRITE) as span:
            span.set_memory_key(key, operation="write", backend=backend)
            yield span

    @contextmanager
    def reasoning_step(
        self,
        step_index: int = 0,
        step_type: str = "",
        name: str = "reasoning.step",
    ) -> Generator[AgentSpan, None, None]:
        """Trace a single reasoning step."""
        with self._managed_span(name, AgentSpanKind.REASONING_STEP) as span:
            span.set_reasoning(step_index=step_index, step_type=step_type)
            yield span

    @contextmanager
    def agent_delegate(
        self, target_agent: str, name: str = "agent.delegate"
    ) -> Generator[AgentSpan, None, None]:
        """Trace delegation of a task to another agent."""
        with self._managed_span(name, AgentSpanKind.AGENT_DELEGATE) as span:
            span.set_delegation(target_agent)
            yield span

    @contextmanager
    def human_approval(
        self,
        requested_by: str,
        timeout_seconds: Optional[int] = None,
        name: str = "human.approval",
    ) -> Generator[AgentSpan, None, None]:
        """Trace a human-in-the-loop approval gate."""
        with self._managed_span(name, AgentSpanKind.HUMAN_APPROVAL) as span:
            span.set_human_approval(requested_by, timeout_seconds=timeout_seconds)
            yield span

    @contextmanager
    def agent_error(
        self,
        error_type: str,
        recoverable: bool = True,
        name: str = "agent.error",
    ) -> Generator[AgentSpan, None, None]:
        """Trace an agent error event."""
        with self._managed_span(name, AgentSpanKind.AGENT_ERROR) as span:
            span.set_error(error_type=error_type, recoverable=recoverable)
            yield span
