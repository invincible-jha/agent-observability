"""AgentTracer — high-level tracer with 8 agent-semantic context managers.

Wraps the lower-level AgentSpan machinery from ``spans.types`` and adds:
- Local span collection for offline/batch export
- A serialise-to-dict ``export()`` helper
- A ``flush()`` method to clear collected spans

All framework imports are kept behind the existing OTel optional guards in
the span layer; this module has no hard OTel dependency of its own.
"""
from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from typing import Generator, Optional

from agent_observability.spans.conventions import (
    AGENT_ERROR_TYPE,
    DELEGATION_TARGET_AGENT,
    HUMAN_APPROVAL_REQUESTED_BY,
    HUMAN_APPROVAL_TIMEOUT_SECONDS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TOKENS_INPUT,
    MEMORY_BACKEND,
    MEMORY_KEY,
    MEMORY_OPERATION,
    REASONING_STEP_TYPE,
    TOOL_NAME,
)
from agent_observability.spans.types import (
    AgentSpan,
    AgentSpanKind,
    AgentTracer as _BaseAgentTracer,
)


class AgentTracer:
    """High-level tracer with local span collection and 8 semantic context managers.

    Parameters
    ----------
    service_name:
        Logical service / agent name included in every exported span dict.
    export_endpoint:
        Optional OTLP HTTP endpoint URL.  Stored for reference; actual
        network export is delegated to :class:`~agent_observability.exporter.otlp.OTLPExporter`.
    agent_id:
        Agent identifier forwarded to the underlying span factory.
    session_id:
        Session identifier forwarded to the underlying span factory.
    framework:
        Framework name (e.g. ``"langchain"``, ``"crewai"``).
    """

    def __init__(
        self,
        service_name: str = "agent",
        export_endpoint: Optional[str] = None,
        agent_id: str = "",
        session_id: str = "",
        framework: str = "",
    ) -> None:
        self.service_name = service_name
        self.export_endpoint = export_endpoint
        self._agent_id = agent_id
        self._session_id = session_id
        self._framework = framework

        self._base = _BaseAgentTracer(
            tracer_name=service_name,
            agent_id=agent_id,
            session_id=session_id,
            framework=framework,
        )

        # Local span collection — guarded by a lock for thread safety
        self._spans: list[AgentSpan] = []
        self._span_metas: list[dict[str, object]] = []
        self._lock = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record_span_meta(
        self,
        kind: AgentSpanKind,
        name: str,
        start_ns: int,
        end_ns: int,
        attributes: dict[str, object],
        error: Optional[str],
    ) -> None:
        """Store a plain-dict snapshot of a completed span."""
        meta: dict[str, object] = {
            "service_name": self.service_name,
            "name": name,
            "kind": kind.value,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "duration_ms": (end_ns - start_ns) / 1_000_000,
            "attributes": attributes,
            "error": error,
        }
        with self._lock:
            self._span_metas.append(meta)

    @contextmanager
    def _collect(
        self,
        kind: AgentSpanKind,
        name: str,
        attributes: dict[str, object],
    ) -> Generator[AgentSpan, None, None]:
        """Wraps the base tracer context manager and captures a dict snapshot."""
        start_ns = time.time_ns()
        error_msg: Optional[str] = None

        with self._base._managed_span(name, kind) as span:
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
            try:
                yield span
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                end_ns = time.time_ns()
                # Capture any attributes that were set during the context
                final_attrs: dict[str, object] = dict(attributes)
                self._record_span_meta(kind, name, start_ns, end_ns, final_attrs, error_msg)

    # ── Public context managers ───────────────────────────────────────────────

    @contextmanager
    def llm_call(
        self,
        model: str,
        provider: str,
        prompt_tokens: int = 0,
    ) -> Generator[AgentSpan, None, None]:
        """Trace a single LLM request/response cycle.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"claude-sonnet-4"``).
        provider:
            Provider name (e.g. ``"anthropic"``).
        prompt_tokens:
            Estimated prompt token count (updated on the span after yielding).
        """
        attrs: dict[str, object] = {
            LLM_MODEL: model,
            LLM_PROVIDER: provider,
            LLM_TOKENS_INPUT: prompt_tokens,
        }
        with self._collect(AgentSpanKind.LLM_CALL, "llm.call", attrs) as span:
            yield span

    @contextmanager
    def tool_invoke(
        self,
        tool_name: str,
        arguments: dict[str, object] | None = None,
    ) -> Generator[AgentSpan, None, None]:
        """Trace a tool invocation.

        Parameters
        ----------
        tool_name:
            Name of the tool being invoked.
        arguments:
            Keyword arguments passed to the tool (not recorded for PII safety;
            callers may explicitly set attributes on the yielded span).
        """
        attrs: dict[str, object] = {TOOL_NAME: tool_name}
        with self._collect(AgentSpanKind.TOOL_INVOKE, "tool.invoke", attrs) as span:
            yield span

    @contextmanager
    def memory_read(
        self,
        store: str,
        query: str = "",
    ) -> Generator[AgentSpan, None, None]:
        """Trace a memory read operation.

        Parameters
        ----------
        store:
            Backend/store identifier (e.g. ``"redis"``, ``"pinecone"``).
        query:
            The query or key used to look up memory.
        """
        attrs: dict[str, object] = {
            MEMORY_BACKEND: store,
            MEMORY_OPERATION: "read",
            MEMORY_KEY: query,
        }
        with self._collect(AgentSpanKind.MEMORY_READ, "memory.read", attrs) as span:
            yield span

    @contextmanager
    def memory_write(
        self,
        store: str,
        key: str = "",
    ) -> Generator[AgentSpan, None, None]:
        """Trace a memory write operation.

        Parameters
        ----------
        store:
            Backend/store identifier.
        key:
            The key under which data is written.
        """
        attrs: dict[str, object] = {
            MEMORY_BACKEND: store,
            MEMORY_OPERATION: "write",
            MEMORY_KEY: key,
        }
        with self._collect(AgentSpanKind.MEMORY_WRITE, "memory.write", attrs) as span:
            yield span

    @contextmanager
    def reasoning_step(
        self,
        step_name: str,
        description: str = "",
    ) -> Generator[AgentSpan, None, None]:
        """Trace a single reasoning step.

        Parameters
        ----------
        step_name:
            Semantic label for this step (e.g. ``"plan"``, ``"reflect"``).
        description:
            Human-readable description of what this step is doing.
        """
        attrs: dict[str, object] = {REASONING_STEP_TYPE: step_name}
        if description:
            attrs["reasoning.description"] = description
        with self._collect(AgentSpanKind.REASONING_STEP, "reasoning.step", attrs) as span:
            yield span

    @contextmanager
    def agent_delegate(
        self,
        target_agent: str,
        task: str = "",
    ) -> Generator[AgentSpan, None, None]:
        """Trace delegation of work to another agent.

        Parameters
        ----------
        target_agent:
            Identifier of the agent receiving the delegated task.
        task:
            Short description or ID of the delegated task.
        """
        attrs: dict[str, object] = {DELEGATION_TARGET_AGENT: target_agent}
        if task:
            attrs["delegation.task"] = task
        with self._collect(AgentSpanKind.AGENT_DELEGATE, "agent.delegate", attrs) as span:
            yield span

    @contextmanager
    def human_approval(
        self,
        action: str,
        timeout_seconds: float = 0,
    ) -> Generator[AgentSpan, None, None]:
        """Trace a human-in-the-loop approval gate.

        Parameters
        ----------
        action:
            Description of the action requiring human approval.
        timeout_seconds:
            How long to wait for approval before timing out (0 = no timeout).
        """
        attrs: dict[str, object] = {
            HUMAN_APPROVAL_REQUESTED_BY: action,
            "human_approval.action": action,
        }
        if timeout_seconds > 0:
            attrs[HUMAN_APPROVAL_TIMEOUT_SECONDS] = int(timeout_seconds)
        with self._collect(AgentSpanKind.HUMAN_APPROVAL, "human.approval", attrs) as span:
            yield span

    @contextmanager
    def agent_error(
        self,
        error_type: str,
        message: str = "",
    ) -> Generator[AgentSpan, None, None]:
        """Trace an agent error event.

        Parameters
        ----------
        error_type:
            Classifier for the error (e.g. ``"RateLimitError"``).
        message:
            Human-readable error message.
        """
        attrs: dict[str, object] = {AGENT_ERROR_TYPE: error_type}
        if message:
            attrs["agent.error.message"] = message
        with self._collect(AgentSpanKind.AGENT_ERROR, "agent.error", attrs) as span:
            yield span

    # ── Span collection helpers ───────────────────────────────────────────────

    def export(self) -> list[dict[str, object]]:
        """Return a serialised snapshot of all collected spans.

        Each dict is safe to JSON-serialise and contains at minimum:
        ``service_name``, ``name``, ``kind``, ``start_time_ns``,
        ``end_time_ns``, ``duration_ms``, ``attributes``, ``error``.
        """
        with self._lock:
            return list(self._span_metas)

    def flush(self) -> None:
        """Clear all stored span metadata from local collection."""
        with self._lock:
            self._span_metas.clear()
            self._spans.clear()
