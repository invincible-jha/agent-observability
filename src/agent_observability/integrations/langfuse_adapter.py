"""Langfuse adapter — enriches Langfuse traces with AumOS agent-semantic spans.

Wraps the Langfuse SDK's ``trace()`` and ``generation()``/``span()`` methods to
attach ``gen.ai.agent.*`` OpenTelemetry semantic attributes. Maps AumOS
AgentSpanKind values to the appropriate Langfuse observation type.

Install the extra to use this module::

    pip install aumos-agent-observability[langfuse]

Usage
-----
::

    from agent_observability.integrations.langfuse_adapter import LangfuseAgentTracer

    tracer = LangfuseAgentTracer(agent_id="planner-01", session_id="sess-abc")
    with tracer.trace_agent_session("plan-task") as trace_ctx:
        with tracer.trace_agent_action("decide-route", span_kind="decision") as span:
            span.update(metadata={"selected_tool": "web_search"})
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional

from agent_observability.spans.types import AgentSpanKind

try:
    import langfuse  # type: ignore[import-untyped]
    from langfuse import Langfuse  # type: ignore[import-untyped]
except ImportError as _import_error:
    raise ImportError(
        "Langfuse is required for this adapter. "
        "Install it with: pip install aumos-agent-observability[langfuse]"
    ) from _import_error

logger = logging.getLogger(__name__)

# Mapping from AumOS AgentSpanKind to Langfuse observation type.
# "generation" is for LLM calls that produce tokens; "span" for everything else.
_KIND_TO_LANGFUSE_TYPE: dict[AgentSpanKind, str] = {
    AgentSpanKind.LLM_CALL: "generation",
    AgentSpanKind.TOOL_INVOKE: "span",
    AgentSpanKind.MEMORY_READ: "span",
    AgentSpanKind.MEMORY_WRITE: "span",
    AgentSpanKind.REASONING_STEP: "span",
    AgentSpanKind.AGENT_DELEGATE: "span",
    AgentSpanKind.HUMAN_APPROVAL: "span",
    AgentSpanKind.AGENT_ERROR: "span",
}

# AumOS gen.ai.agent.* attribute namespace prefix
_AGENT_ATTR_PREFIX = "gen.ai.agent"


@dataclass
class SpanContext:
    """Lightweight context returned by LangfuseAgentTracer context managers.

    Parameters
    ----------
    langfuse_observation:
        The raw Langfuse span or generation object.
    span_kind:
        The AumOS semantic span kind assigned to this observation.
    observation_type:
        Either ``"generation"`` or ``"span"`` as mapped from span_kind.
    """

    langfuse_observation: Any
    span_kind: AgentSpanKind
    observation_type: str

    def update(self, **kwargs: Any) -> None:
        """Forward keyword arguments to the underlying Langfuse observation's update()."""
        update_fn = getattr(self.langfuse_observation, "update", None)
        if callable(update_fn):
            update_fn(**kwargs)

    def end(self, **kwargs: Any) -> None:
        """End the underlying Langfuse observation."""
        end_fn = getattr(self.langfuse_observation, "end", None)
        if callable(end_fn):
            end_fn(**kwargs)


class LangfuseAgentTracer:
    """Wraps Langfuse with AumOS agent-semantic span enrichment.

    All spans created through this tracer automatically receive
    ``gen.ai.agent.*`` metadata attributes aligning with the AumOS
    OpenTelemetry conventions.

    Parameters
    ----------
    agent_id:
        Identifier for the agent producing these traces.
    session_id:
        Session identifier attached to every trace.
    framework:
        Framework name (e.g. ``"crewai"``, ``"langchain"``).
    langfuse_client:
        Pre-configured Langfuse client. When omitted, a new client
        is created using environment variables for authentication.
    public_key:
        Langfuse public key (used only when ``langfuse_client`` is None).
    secret_key:
        Langfuse secret key (used only when ``langfuse_client`` is None).
    host:
        Langfuse host URL (used only when ``langfuse_client`` is None).

    Examples
    --------
    ::

        tracer = LangfuseAgentTracer(agent_id="agent-1", session_id="s-001")
        with tracer.trace_agent_session("research-task") as ctx:
            with tracer.trace_agent_action("search-web", span_kind="tool_invoke") as span:
                span.update(metadata={"query": "AI safety"})
    """

    def __init__(
        self,
        agent_id: str = "",
        session_id: str = "",
        framework: str = "",
        langfuse_client: Optional[Any] = None,
        public_key: str = "",
        secret_key: str = "",
        host: str = "",
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._framework = framework

        if langfuse_client is not None:
            self._client = langfuse_client
        else:
            init_kwargs: dict[str, str] = {}
            if public_key:
                init_kwargs["public_key"] = public_key
            if secret_key:
                init_kwargs["secret_key"] = secret_key
            if host:
                init_kwargs["host"] = host
            self._client = Langfuse(**init_kwargs)

        self._active_trace: Optional[Any] = None

    # ------------------------------------------------------------------
    # Core agent attribute builder
    # ------------------------------------------------------------------

    def _build_agent_metadata(
        self,
        span_kind: AgentSpanKind,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the metadata dict with gen.ai.agent.* attributes.

        Parameters
        ----------
        span_kind:
            The AumOS span kind for this observation.
        extra:
            Additional caller-supplied metadata merged into the result.

        Returns
        -------
        dict[str, Any]
            Metadata dict suitable for Langfuse span/generation ``metadata``
            parameter.
        """
        metadata: dict[str, Any] = {
            f"{_AGENT_ATTR_PREFIX}.span_kind": span_kind.value,
        }
        if self._agent_id:
            metadata[f"{_AGENT_ATTR_PREFIX}.agent_id"] = self._agent_id
        if self._session_id:
            metadata[f"{_AGENT_ATTR_PREFIX}.session_id"] = self._session_id
        if self._framework:
            metadata[f"{_AGENT_ATTR_PREFIX}.framework"] = self._framework
        if extra:
            metadata.update(extra)
        return metadata

    def _resolve_span_kind(self, span_kind_input: str | AgentSpanKind) -> AgentSpanKind:
        """Resolve a string or enum span kind to AgentSpanKind."""
        if isinstance(span_kind_input, AgentSpanKind):
            return span_kind_input
        try:
            return AgentSpanKind(span_kind_input)
        except ValueError:
            logger.warning(
                "Unknown span_kind %r — defaulting to REASONING_STEP", span_kind_input
            )
            return AgentSpanKind.REASONING_STEP

    # ------------------------------------------------------------------
    # Session-level trace
    # ------------------------------------------------------------------

    @contextmanager
    def trace_agent_session(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator["SpanContext", None, None]:
        """Open a Langfuse trace representing an agent session.

        Parameters
        ----------
        name:
            Human-readable name for this trace (e.g. ``"research-task"``).
        metadata:
            Additional metadata to attach to the trace.

        Yields
        ------
        SpanContext
            Context wrapping the Langfuse trace object.
        """
        combined_meta = self._build_agent_metadata(
            AgentSpanKind.REASONING_STEP, extra=metadata
        )
        trace_kwargs: dict[str, Any] = {"name": name, "metadata": combined_meta}
        if self._session_id:
            trace_kwargs["session_id"] = self._session_id

        trace = self._client.trace(**trace_kwargs)
        self._active_trace = trace
        ctx = SpanContext(
            langfuse_observation=trace,
            span_kind=AgentSpanKind.REASONING_STEP,
            observation_type="trace",
        )
        try:
            yield ctx
        finally:
            flush_fn = getattr(self._client, "flush", None)
            if callable(flush_fn):
                flush_fn()
            self._active_trace = None

    # ------------------------------------------------------------------
    # Action-level observation
    # ------------------------------------------------------------------

    @contextmanager
    def trace_agent_action(
        self,
        name: str,
        span_kind: str | AgentSpanKind = AgentSpanKind.REASONING_STEP,
        metadata: Optional[dict[str, Any]] = None,
        model: str = "",
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
    ) -> Generator[SpanContext, None, None]:
        """Trace a single agent action as a Langfuse span or generation.

        LLM_CALL span kinds are recorded as Langfuse ``generation`` observations
        (which support token counts and model metadata). All other kinds are
        recorded as ``span`` observations.

        Parameters
        ----------
        name:
            Human-readable name for this action.
        span_kind:
            AumOS semantic kind. Accepts string or AgentSpanKind enum.
        metadata:
            Additional metadata merged with agent attributes.
        model:
            Model name — attached to ``gen.ai.agent.model`` and to Langfuse
            generation ``model`` field when span_kind is LLM_CALL.
        input_data:
            Input payload for this span.
        output_data:
            Output payload for this span (can be updated in the ``with`` block).

        Yields
        ------
        SpanContext
            Context object with ``update()`` and ``end()`` methods.
        """
        resolved_kind = self._resolve_span_kind(span_kind)
        observation_type = _KIND_TO_LANGFUSE_TYPE.get(resolved_kind, "span")

        combined_meta = self._build_agent_metadata(resolved_kind, extra=metadata)
        if model:
            combined_meta[f"{_AGENT_ATTR_PREFIX}.model"] = model

        parent = self._active_trace
        observation: Any

        if observation_type == "generation":
            gen_kwargs: dict[str, Any] = {
                "name": name,
                "metadata": combined_meta,
            }
            if model:
                gen_kwargs["model"] = model
            if input_data is not None:
                gen_kwargs["input"] = input_data
            if output_data is not None:
                gen_kwargs["output"] = output_data

            if parent is not None:
                create_gen = getattr(parent, "generation", None)
                if callable(create_gen):
                    observation = create_gen(**gen_kwargs)
                else:
                    observation = self._client.generation(**gen_kwargs)
            else:
                observation = self._client.generation(**gen_kwargs)
        else:
            span_kwargs: dict[str, Any] = {
                "name": name,
                "metadata": combined_meta,
            }
            if input_data is not None:
                span_kwargs["input"] = input_data
            if output_data is not None:
                span_kwargs["output"] = output_data

            if parent is not None:
                create_span = getattr(parent, "span", None)
                if callable(create_span):
                    observation = create_span(**span_kwargs)
                else:
                    observation = self._client.span(**span_kwargs)
            else:
                observation = self._client.span(**span_kwargs)

        ctx = SpanContext(
            langfuse_observation=observation,
            span_kind=resolved_kind,
            observation_type=observation_type,
        )
        try:
            yield ctx
        finally:
            end_fn = getattr(observation, "end", None)
            if callable(end_fn):
                end_fn()

    # ------------------------------------------------------------------
    # Convenience methods for specific span kinds
    # ------------------------------------------------------------------

    @contextmanager
    def trace_llm_call(
        self,
        name: str,
        model: str,
        input_messages: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Trace an LLM call as a Langfuse generation with model metadata.

        Parameters
        ----------
        name:
            Name for the generation span.
        model:
            Model identifier (e.g. ``"gpt-4o"``).
        input_messages:
            OpenAI-format message list as the generation input.
        metadata:
            Additional metadata.

        Yields
        ------
        SpanContext
        """
        with self.trace_agent_action(
            name=name,
            span_kind=AgentSpanKind.LLM_CALL,
            metadata=metadata,
            model=model,
            input_data=input_messages,
        ) as ctx:
            yield ctx

    @contextmanager
    def trace_tool_call(
        self,
        tool_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Trace a tool invocation as a Langfuse span.

        Parameters
        ----------
        tool_name:
            Name of the tool being invoked.
        metadata:
            Additional metadata.

        Yields
        ------
        SpanContext
        """
        combined = {f"{_AGENT_ATTR_PREFIX}.tool_name": tool_name}
        if metadata:
            combined.update(metadata)
        with self.trace_agent_action(
            name=f"tool.{tool_name}",
            span_kind=AgentSpanKind.TOOL_INVOKE,
            metadata=combined,
        ) as ctx:
            yield ctx

    @contextmanager
    def trace_delegation(
        self,
        target_agent: str,
        task_description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """Trace delegation of a sub-task to another agent.

        Parameters
        ----------
        target_agent:
            Identifier of the agent receiving the delegation.
        task_description:
            Brief description of the delegated task.
        metadata:
            Additional metadata.

        Yields
        ------
        SpanContext
        """
        combined: dict[str, Any] = {
            f"{_AGENT_ATTR_PREFIX}.delegation_target": target_agent,
        }
        if task_description:
            combined[f"{_AGENT_ATTR_PREFIX}.task_description"] = task_description
        if metadata:
            combined.update(metadata)
        with self.trace_agent_action(
            name=f"delegate.{target_agent}",
            span_kind=AgentSpanKind.AGENT_DELEGATE,
            metadata=combined,
        ) as ctx:
            yield ctx

    def flush(self) -> None:
        """Flush any buffered Langfuse events to the server."""
        flush_fn = getattr(self._client, "flush", None)
        if callable(flush_fn):
            flush_fn()

    def __repr__(self) -> str:
        return (
            f"LangfuseAgentTracer("
            f"agent_id={self._agent_id!r}, "
            f"session_id={self._session_id!r}, "
            f"framework={self._framework!r})"
        )
