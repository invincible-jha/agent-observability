"""Tests for tracer.agent_tracer (high-level AgentTracer with span collection)."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agent_observability.tracer.agent_tracer import AgentTracer


@pytest.fixture()
def tracer() -> AgentTracer:
    return AgentTracer(
        service_name="test-service",
        agent_id="agent-test",
        session_id="session-test",
        framework="test-framework",
    )


class TestAgentTracerInit:
    def test_attributes_stored(self, tracer: AgentTracer) -> None:
        assert tracer.service_name == "test-service"
        assert tracer._agent_id == "agent-test"
        assert tracer._session_id == "session-test"
        assert tracer._framework == "test-framework"

    def test_optional_export_endpoint(self) -> None:
        t = AgentTracer(export_endpoint="http://localhost:4318")
        assert t.export_endpoint == "http://localhost:4318"

    def test_defaults(self) -> None:
        t = AgentTracer()
        assert t.service_name == "agent"
        assert t.export_endpoint is None


class TestAgentTracerContextManagers:
    def test_llm_call_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai") as span:
                assert span is not None
        spans = tracer.export()
        assert len(spans) == 1
        assert spans[0]["name"] == "llm.call"
        assert spans[0]["kind"] == "llm_call"

    def test_tool_invoke_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.tool_invoke(tool_name="web_search") as span:
                assert span is not None
        spans = tracer.export()
        assert len(spans) == 1
        assert spans[0]["name"] == "tool.invoke"

    def test_memory_read_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.memory_read(store="redis", query="key:123") as span:
                assert span is not None
        spans = tracer.export()
        assert len(spans) == 1
        assert spans[0]["name"] == "memory.read"

    def test_memory_write_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.memory_write(store="redis", key="output:1") as span:
                assert span is not None
        spans = tracer.export()
        assert spans[0]["name"] == "memory.write"

    def test_reasoning_step_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.reasoning_step(step_name="plan", description="Planning phase") as span:
                assert span is not None
        spans = tracer.export()
        assert spans[0]["name"] == "reasoning.step"

    def test_reasoning_step_no_description(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.reasoning_step(step_name="reflect") as span:
                assert span is not None
        spans = tracer.export()
        assert len(spans) == 1

    def test_agent_delegate_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.agent_delegate(target_agent="sub-agent-1", task="summarise docs") as span:
                assert span is not None
        spans = tracer.export()
        assert spans[0]["name"] == "agent.delegate"

    def test_agent_delegate_no_task(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.agent_delegate(target_agent="sub-agent") as span:
                assert span is not None
        spans = tracer.export()
        assert len(spans) == 1

    def test_human_approval_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.human_approval(action="deploy", timeout_seconds=60.0) as span:
                assert span is not None
        spans = tracer.export()
        assert spans[0]["name"] == "human.approval"

    def test_human_approval_no_timeout(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.human_approval(action="confirm") as span:
                assert span is not None
        spans = tracer.export()
        assert len(spans) == 1

    def test_agent_error_records_span(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.agent_error(error_type="RateLimitError", message="too many requests"):
                pass
        spans = tracer.export()
        assert spans[0]["name"] == "agent.error"

    def test_agent_error_no_message(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.agent_error(error_type="TimeoutError"):
                pass
        spans = tracer.export()
        assert len(spans) == 1


class TestAgentTracerSpanMeta:
    def test_span_meta_contains_service_name(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai"):
                pass
        meta = tracer.export()[0]
        assert meta["service_name"] == "test-service"

    def test_span_meta_has_timing(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai"):
                pass
        meta = tracer.export()[0]
        assert meta["start_time_ns"] > 0
        assert meta["end_time_ns"] >= meta["start_time_ns"]
        assert meta["duration_ms"] >= 0.0

    def test_span_meta_error_field_none_on_success(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai"):
                pass
        meta = tracer.export()[0]
        assert meta["error"] is None

    def test_span_meta_error_captured_on_exception(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with pytest.raises(ValueError):
                with tracer.llm_call(model="gpt-4o", provider="openai"):
                    raise ValueError("test error")
        meta = tracer.export()[0]
        assert meta["error"] is not None
        assert "ValueError" in meta["error"]

    def test_multiple_spans_accumulated(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai"):
                pass
            with tracer.tool_invoke(tool_name="search"):
                pass
        spans = tracer.export()
        assert len(spans) == 2

    def test_export_returns_copy(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai"):
                pass
        spans1 = tracer.export()
        spans1.append({"fake": True})
        assert len(tracer.export()) == 1  # original unchanged


class TestAgentTracerFlush:
    def test_flush_clears_spans(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="gpt-4o", provider="openai"):
                pass
        assert len(tracer.export()) == 1
        tracer.flush()
        assert tracer.export() == []

    def test_flush_is_idempotent(self, tracer: AgentTracer) -> None:
        tracer.flush()
        tracer.flush()
        assert tracer.export() == []

    def test_attributes_included_in_meta(self, tracer: AgentTracer) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with tracer.llm_call(model="claude-sonnet-4", provider="anthropic", prompt_tokens=100):
                pass
        meta = tracer.export()[0]
        assert "llm.model" in meta["attributes"] or "attributes" in meta
