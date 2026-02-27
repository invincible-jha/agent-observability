"""Tests for agent_observability.integrations.langfuse_adapter.

The Langfuse package is mocked — it is not required in CI.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, call

import pytest


# ---------------------------------------------------------------------------
# Fixture: inject a fake langfuse module before the adapter is imported
# ---------------------------------------------------------------------------


def _make_mock_langfuse_module() -> types.ModuleType:
    """Build a minimal langfuse stub."""
    mod = types.ModuleType("langfuse")

    mock_trace = MagicMock()
    mock_trace.generation = MagicMock(return_value=MagicMock())
    mock_trace.span = MagicMock(return_value=MagicMock())

    mock_client = MagicMock()
    mock_client.trace = MagicMock(return_value=mock_trace)
    mock_client.generation = MagicMock(return_value=MagicMock())
    mock_client.span = MagicMock(return_value=MagicMock())
    mock_client.flush = MagicMock()

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def trace(self, **kwargs: Any) -> Any:
            return mock_client.trace(**kwargs)

        def generation(self, **kwargs: Any) -> Any:
            return mock_client.generation(**kwargs)

        def span(self, **kwargs: Any) -> Any:
            return mock_client.span(**kwargs)

        def flush(self) -> None:
            mock_client.flush()

    mod.Langfuse = FakeLangfuse  # type: ignore[attr-defined]
    mod._mock_client = mock_client  # type: ignore[attr-defined]
    mod._mock_trace = mock_trace  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def inject_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake langfuse before each test."""
    fake_langfuse = _make_mock_langfuse_module()
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    monkeypatch.delitem(
        sys.modules,
        "agent_observability.integrations.langfuse_adapter",
        raising=False,
    )


def _import_adapter() -> Any:
    from agent_observability.integrations import langfuse_adapter  # noqa: PLC0415

    return langfuse_adapter


def _make_tracer(**kwargs: Any) -> Any:
    adapter = _import_adapter()
    return adapter.LangfuseAgentTracer(**kwargs)


# ---------------------------------------------------------------------------
# LangfuseAgentTracer — construction
# ---------------------------------------------------------------------------


class TestLangfuseAgentTracerConstruction:
    def test_default_construction(self) -> None:
        tracer = _make_tracer()
        assert tracer is not None

    def test_agent_id_stored(self) -> None:
        tracer = _make_tracer(agent_id="agent-42")
        assert tracer._agent_id == "agent-42"

    def test_session_id_stored(self) -> None:
        tracer = _make_tracer(session_id="sess-001")
        assert tracer._session_id == "sess-001"

    def test_framework_stored(self) -> None:
        tracer = _make_tracer(framework="crewai")
        assert tracer._framework == "crewai"

    def test_repr_contains_class_name(self) -> None:
        tracer = _make_tracer(agent_id="a1")
        assert "LangfuseAgentTracer" in repr(tracer)

    def test_custom_client_accepted(self) -> None:
        mock_client = MagicMock()
        tracer = _make_tracer(langfuse_client=mock_client)
        assert tracer._client is mock_client


# ---------------------------------------------------------------------------
# _build_agent_metadata
# ---------------------------------------------------------------------------


class TestBuildAgentMetadata:
    def test_span_kind_in_metadata(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer(agent_id="a1")
        meta = tracer._build_agent_metadata(AgentSpanKind.LLM_CALL)
        assert "gen.ai.agent.span_kind" in meta
        assert meta["gen.ai.agent.span_kind"] == "llm_call"

    def test_agent_id_in_metadata_when_set(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer(agent_id="agent-99")
        meta = tracer._build_agent_metadata(AgentSpanKind.TOOL_INVOKE)
        assert meta.get("gen.ai.agent.agent_id") == "agent-99"

    def test_session_id_in_metadata_when_set(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer(session_id="s-42")
        meta = tracer._build_agent_metadata(AgentSpanKind.MEMORY_READ)
        assert meta.get("gen.ai.agent.session_id") == "s-42"

    def test_extra_metadata_merged(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer()
        meta = tracer._build_agent_metadata(
            AgentSpanKind.REASONING_STEP,
            extra={"custom_key": "custom_value"},
        )
        assert meta["custom_key"] == "custom_value"

    def test_no_agent_id_key_when_empty(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer()
        meta = tracer._build_agent_metadata(AgentSpanKind.TOOL_INVOKE)
        assert "gen.ai.agent.agent_id" not in meta


# ---------------------------------------------------------------------------
# _resolve_span_kind
# ---------------------------------------------------------------------------


class TestResolveSpanKind:
    def test_resolves_string_to_enum(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer()
        result = tracer._resolve_span_kind("llm_call")
        assert result == AgentSpanKind.LLM_CALL

    def test_enum_passthrough(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer()
        result = tracer._resolve_span_kind(AgentSpanKind.TOOL_INVOKE)
        assert result == AgentSpanKind.TOOL_INVOKE

    def test_unknown_string_defaults_to_reasoning_step(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        tracer = _make_tracer()
        result = tracer._resolve_span_kind("unknown_kind_xyz")
        assert result == AgentSpanKind.REASONING_STEP


# ---------------------------------------------------------------------------
# trace_agent_session context manager
# ---------------------------------------------------------------------------


class TestTraceAgentSession:
    def test_yields_span_context(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_agent_session("my-task") as ctx:
            assert isinstance(ctx, adapter.SpanContext)

    def test_active_trace_set_within_context(self) -> None:
        tracer = _make_tracer()
        with tracer.trace_agent_session("task") as _ctx:
            assert tracer._active_trace is not None

    def test_active_trace_cleared_after_context(self) -> None:
        tracer = _make_tracer()
        with tracer.trace_agent_session("task"):
            pass
        assert tracer._active_trace is None

    def test_span_context_has_correct_observation_type(self) -> None:
        tracer = _make_tracer()
        with tracer.trace_agent_session("task") as ctx:
            assert ctx.observation_type == "trace"


# ---------------------------------------------------------------------------
# trace_agent_action context manager
# ---------------------------------------------------------------------------


class TestTraceAgentAction:
    def test_llm_call_uses_generation_type(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_agent_action(
            "gen-call", span_kind="llm_call"
        ) as ctx:
            assert isinstance(ctx, adapter.SpanContext)
            assert ctx.observation_type == "generation"

    def test_tool_invoke_uses_span_type(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_agent_action(
            "tool-call", span_kind="tool_invoke"
        ) as ctx:
            assert ctx.observation_type == "span"

    def test_reasoning_step_uses_span_type(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_agent_action(
            "think", span_kind="reasoning_step"
        ) as ctx:
            assert ctx.observation_type == "span"

    def test_delegation_uses_span_type(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_agent_action(
            "delegate", span_kind="agent_delegate"
        ) as ctx:
            assert ctx.observation_type == "span"

    def test_update_called_on_span_context(self) -> None:
        tracer = _make_tracer()
        with tracer.trace_agent_action("search", span_kind="tool_invoke") as ctx:
            # Should not raise
            ctx.update(metadata={"query": "test"})

    def test_model_metadata_attached_for_llm_call(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_agent_action(
            "llm-gen",
            span_kind="llm_call",
            model="gpt-4o",
        ) as ctx:
            assert isinstance(ctx, adapter.SpanContext)


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    def test_trace_llm_call_yields_span_context(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_llm_call("llm-op", model="gpt-4o") as ctx:
            assert isinstance(ctx, adapter.SpanContext)
            assert ctx.observation_type == "generation"

    def test_trace_tool_call_yields_span_context(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_tool_call("web_search") as ctx:
            assert isinstance(ctx, adapter.SpanContext)
            assert ctx.observation_type == "span"

    def test_trace_delegation_yields_span_context(self) -> None:
        adapter = _import_adapter()
        tracer = _make_tracer()
        with tracer.trace_delegation("sub-agent-1") as ctx:
            assert isinstance(ctx, adapter.SpanContext)
            assert ctx.observation_type == "span"

    def test_flush_called_without_error(self) -> None:
        tracer = _make_tracer()
        # Should not raise
        tracer.flush()


# ---------------------------------------------------------------------------
# SpanContext
# ---------------------------------------------------------------------------


class TestSpanContext:
    def test_update_delegates_to_observation(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        adapter = _import_adapter()
        mock_obs = MagicMock()
        ctx = adapter.SpanContext(
            langfuse_observation=mock_obs,
            span_kind=AgentSpanKind.LLM_CALL,
            observation_type="generation",
        )
        ctx.update(output="result")
        mock_obs.update.assert_called_once_with(output="result")

    def test_end_delegates_to_observation(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        adapter = _import_adapter()
        mock_obs = MagicMock()
        ctx = adapter.SpanContext(
            langfuse_observation=mock_obs,
            span_kind=AgentSpanKind.TOOL_INVOKE,
            observation_type="span",
        )
        ctx.end()
        mock_obs.end.assert_called_once()

    def test_update_safe_when_no_update_method(self) -> None:
        from agent_observability.spans.types import AgentSpanKind  # noqa: PLC0415

        adapter = _import_adapter()
        obs_without_update = object()
        ctx = adapter.SpanContext(
            langfuse_observation=obs_without_update,
            span_kind=AgentSpanKind.REASONING_STEP,
            observation_type="span",
        )
        # Should not raise — graceful fallback
        ctx.update(metadata={})
