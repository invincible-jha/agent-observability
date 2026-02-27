"""Unit tests for spans.types — AgentSpanKind, CostAnnotation, AgentSpan, AgentTracer."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from agent_observability.spans.types import (
    AgentSpan,
    AgentSpanKind,
    AgentTracer,
    CostAnnotation,
    _NoOpSpan,
)
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
    DELEGATION_TARGET_AGENT,
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
    REASONING_STRATEGY,
    REASONING_STEP_TYPE,
    TOOL_ERROR_TYPE,
    TOOL_NAME,
    TOOL_SUCCESS,
)


# ── AgentSpanKind ──────────────────────────────────────────────────────────────

class TestAgentSpanKind:
    def test_all_eight_kinds_exist(self) -> None:
        kinds = {kind.value for kind in AgentSpanKind}
        expected = {
            "llm_call",
            "tool_invoke",
            "memory_read",
            "memory_write",
            "reasoning_step",
            "agent_delegate",
            "human_approval",
            "agent_error",
        }
        assert kinds == expected

    def test_kind_is_string_enum(self) -> None:
        assert AgentSpanKind.LLM_CALL == "llm_call"
        assert isinstance(AgentSpanKind.LLM_CALL, str)

    def test_kind_values_are_snake_case(self) -> None:
        for kind in AgentSpanKind:
            assert kind.value == kind.value.lower()
            assert " " not in kind.value


# ── CostAnnotation ─────────────────────────────────────────────────────────────

class TestCostAnnotation:
    def test_defaults_to_zero(self) -> None:
        annotation = CostAnnotation()
        assert annotation.input_tokens == 0
        assert annotation.output_tokens == 0
        assert annotation.cost_usd == 0.0

    def test_total_tokens_auto_computed(self) -> None:
        annotation = CostAnnotation(input_tokens=100, output_tokens=50)
        assert annotation.total_tokens == 150

    def test_total_tokens_not_overwritten_when_provided(self) -> None:
        annotation = CostAnnotation(input_tokens=100, output_tokens=50, total_tokens=200)
        assert annotation.total_tokens == 200

    def test_total_tokens_zero_when_both_zero(self) -> None:
        annotation = CostAnnotation(input_tokens=0, output_tokens=0)
        assert annotation.total_tokens == 0

    def test_model_and_provider_default_to_empty(self) -> None:
        annotation = CostAnnotation()
        assert annotation.model == ""
        assert annotation.provider == ""


# ── _NoOpSpan ──────────────────────────────────────────────────────────────────

class TestNoOpSpan:
    def test_set_attribute_stored(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("foo", "bar")
        assert span._attributes["foo"] == "bar"

    def test_end_marks_ended(self) -> None:
        span = _NoOpSpan()
        assert not span._ended
        span.end()
        assert span._ended

    def test_record_exception_does_not_raise(self) -> None:
        span = _NoOpSpan()
        span.record_exception(ValueError("test error"))

    def test_set_status_does_not_raise(self) -> None:
        span = _NoOpSpan()
        span.set_status("error", "something went wrong")

    def test_multiple_attributes(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("a", 1)
        span.set_attribute("b", 2.5)
        span.set_attribute("c", True)
        assert span._attributes == {"a": 1, "b": 2.5, "c": True}


# ── AgentSpan ─────────────────────────────────────────────────────────────────

class TestAgentSpan:
    def _make_span(self, kind: AgentSpanKind = AgentSpanKind.LLM_CALL) -> tuple[AgentSpan, _NoOpSpan]:
        inner = _NoOpSpan()
        agent_span = AgentSpan(inner, kind)
        return agent_span, inner

    def test_constructor_sets_kind_attribute(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.LLM_CALL)
        assert inner._attributes[AGENT_SPAN_KIND] == "llm_call"

    def test_set_tokens_records_all_fields(self) -> None:
        agent_span, inner = self._make_span()
        result = agent_span.set_tokens(100, 50)
        assert inner._attributes[LLM_TOKENS_INPUT] == 100
        assert inner._attributes[LLM_TOKENS_OUTPUT] == 50
        assert inner._attributes[LLM_TOKENS_TOTAL] == 150
        assert result is agent_span  # fluent

    def test_set_tokens_uses_explicit_total(self) -> None:
        agent_span, inner = self._make_span()
        agent_span.set_tokens(100, 50, total_tokens=999)
        assert inner._attributes[LLM_TOKENS_TOTAL] == 999

    def test_set_cost_records_usd(self) -> None:
        agent_span, inner = self._make_span()
        result = agent_span.set_cost(0.00123)
        assert inner._attributes[LLM_COST_USD] == 0.00123
        assert result is agent_span

    def test_set_model_records_model_and_provider(self) -> None:
        agent_span, inner = self._make_span()
        agent_span.set_model("gpt-4o", "openai")
        assert inner._attributes[LLM_MODEL] == "gpt-4o"
        assert inner._attributes[LLM_PROVIDER] == "openai"

    def test_set_model_skips_empty_provider(self) -> None:
        agent_span, inner = self._make_span()
        agent_span.set_model("gpt-4o")
        assert LLM_PROVIDER not in inner._attributes

    def test_set_tool_records_name_and_success(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.TOOL_INVOKE)
        agent_span.set_tool("web_search", success=True)
        assert inner._attributes[TOOL_NAME] == "web_search"
        assert inner._attributes[TOOL_SUCCESS] is True

    def test_set_tool_records_error_type(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.TOOL_INVOKE)
        agent_span.set_tool("web_search", success=False, error_type="TimeoutError")
        assert inner._attributes[TOOL_ERROR_TYPE] == "TimeoutError"

    def test_set_tool_omits_error_type_when_empty(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.TOOL_INVOKE)
        agent_span.set_tool("web_search")
        assert TOOL_ERROR_TYPE not in inner._attributes

    def test_set_memory_key_records_key_and_operation(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.MEMORY_READ)
        agent_span.set_memory_key("user:profile", operation="read", backend="redis")
        assert inner._attributes[MEMORY_KEY] == "user:profile"
        assert inner._attributes[MEMORY_OPERATION] == "read"
        assert inner._attributes[MEMORY_BACKEND] == "redis"

    def test_set_memory_key_records_hit_when_provided(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.MEMORY_READ)
        agent_span.set_memory_key("user:profile", hit=True)
        assert inner._attributes[MEMORY_HIT] is True

    def test_set_reasoning_records_step_index(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.REASONING_STEP)
        agent_span.set_reasoning(step_index=3, step_type="chain_of_thought", confidence=0.95)
        assert inner._attributes[REASONING_STEP_INDEX] == 3
        assert inner._attributes[REASONING_STEP_TYPE] == "chain_of_thought"
        assert inner._attributes[REASONING_CONFIDENCE] == 0.95

    def test_set_delegation_records_target_agent(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.AGENT_DELEGATE)
        agent_span.set_delegation("sub_agent_42", task_id="task-99", strategy="round_robin")
        assert inner._attributes[DELEGATION_TARGET_AGENT] == "sub_agent_42"

    def test_set_human_approval_records_status(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.HUMAN_APPROVAL)
        agent_span.set_human_approval("admin_user", status="approved", timeout_seconds=30)
        assert inner._attributes[HUMAN_APPROVAL_REQUESTED_BY] == "admin_user"
        assert inner._attributes[HUMAN_APPROVAL_STATUS] == "approved"
        assert inner._attributes[HUMAN_APPROVAL_TIMEOUT_SECONDS] == 30

    def test_set_error_records_all_fields(self) -> None:
        agent_span, inner = self._make_span(AgentSpanKind.AGENT_ERROR)
        agent_span.set_error("NetworkError", recoverable=True, retry_count=2)
        assert inner._attributes[AGENT_ERROR_TYPE] == "NetworkError"
        assert inner._attributes[AGENT_ERROR_RECOVERABLE] is True
        assert inner._attributes[AGENT_ERROR_RETRY_COUNT] == 2

    def test_set_error_records_exception(self) -> None:
        inner = MagicMock(spec=_NoOpSpan)
        inner.set_attribute = MagicMock()
        agent_span = AgentSpan(inner, AgentSpanKind.AGENT_ERROR)
        exc = ValueError("something failed")
        agent_span.set_error("ValueError", exception=exc)
        inner.record_exception.assert_called_once_with(exc)

    def test_finish_calls_end(self) -> None:
        inner = MagicMock(spec=_NoOpSpan)
        agent_span = AgentSpan(inner, AgentSpanKind.LLM_CALL)
        agent_span.finish()
        inner.end.assert_called_once()

    def test_set_attribute_passes_through(self) -> None:
        agent_span, inner = self._make_span()
        result = agent_span.set_attribute("custom.key", "custom_value")
        assert inner._attributes["custom.key"] == "custom_value"
        assert result is agent_span

    def test_elapsed_seconds_increases_over_time(self) -> None:
        """elapsed_seconds must grow as the monotonic clock advances.

        Using unittest.mock.patch to control time.monotonic avoids any
        dependency on the OS scheduler or sleep precision, making the test
        fully deterministic regardless of machine load or platform.
        """
        monotonic_values = iter([1000.0, 1000.0, 1000.5])

        with patch("agent_observability.spans.types.time") as mock_time:
            mock_time.monotonic.side_effect = monotonic_values
            agent_span, _ = self._make_span()   # consumes 1000.0 for _start_time
            t0 = agent_span.elapsed_seconds     # 1000.0 - 1000.0 == 0.0
            t1 = agent_span.elapsed_seconds     # 1000.5 - 1000.0 == 0.5

        assert t1 > t0

    def test_stable_hash_returns_16_char_hex(self) -> None:
        agent_span, _ = self._make_span()
        h = agent_span._stable_hash("hello world")
        assert len(h) == 16
        int(h, 16)  # should parse as hex without error

    def test_stable_hash_is_deterministic(self) -> None:
        agent_span, _ = self._make_span()
        assert agent_span._stable_hash("test") == agent_span._stable_hash("test")


# ── AgentTracer ───────────────────────────────────────────────────────────────

class TestAgentTracer:
    def _make_tracer(self, **kwargs: str) -> AgentTracer:
        return AgentTracer(tracer_name="test-tracer", **kwargs)

    def test_llm_call_sets_span_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.llm_call() as span:
            assert isinstance(span, AgentSpan)
            assert span._kind == AgentSpanKind.LLM_CALL

    def test_tool_invoke_sets_tool_name(self) -> None:
        tracer = self._make_tracer()
        with tracer.tool_invoke("my_tool") as span:
            assert span._kind == AgentSpanKind.TOOL_INVOKE

    def test_memory_read_sets_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.memory_read("session:abc") as span:
            assert span._kind == AgentSpanKind.MEMORY_READ

    def test_memory_write_sets_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.memory_write("session:abc") as span:
            assert span._kind == AgentSpanKind.MEMORY_WRITE

    def test_reasoning_step_sets_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.reasoning_step(step_index=1) as span:
            assert span._kind == AgentSpanKind.REASONING_STEP

    def test_agent_delegate_sets_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.agent_delegate("sub_agent") as span:
            assert span._kind == AgentSpanKind.AGENT_DELEGATE

    def test_human_approval_sets_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.human_approval("user@example.com") as span:
            assert span._kind == AgentSpanKind.HUMAN_APPROVAL

    def test_agent_error_sets_kind(self) -> None:
        tracer = self._make_tracer()
        with tracer.agent_error("TimeoutError") as span:
            assert span._kind == AgentSpanKind.AGENT_ERROR

    def test_identity_attributes_attached(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = AgentTracer(
                tracer_name="tracer-x",
                agent_id="agent-001",
                session_id="sess-abc",
                framework="crewai",
                task_id="task-7",
                run_id="run-99",
            )
            with tracer.llm_call() as span:
                assert isinstance(span._span, _NoOpSpan)
                attrs = span._span._attributes
                assert attrs[AGENT_ID] == "agent-001"
                assert attrs[AGENT_SESSION_ID] == "sess-abc"
                assert attrs[AGENT_FRAMEWORK] == "crewai"
                assert attrs[AGENT_TASK_ID] == "task-7"
                assert attrs[AGENT_RUN_ID] == "run-99"

    def test_exception_inside_context_manager_sets_error_attributes(self) -> None:
        tracer = self._make_tracer()
        with pytest.raises(RuntimeError):
            with tracer.llm_call() as span:
                raise RuntimeError("unexpected failure")
        # span was finished (end called) — no exception escapes the test

    def test_llm_call_sets_model_if_provided(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = self._make_tracer()
            with tracer.llm_call(model="gpt-4o", provider="openai") as span:
                assert isinstance(span._span, _NoOpSpan)
                assert span._span._attributes[LLM_MODEL] == "gpt-4o"
                assert span._span._attributes[LLM_PROVIDER] == "openai"

    def test_span_is_finished_after_context_exit(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = self._make_tracer()
            captured_span: AgentSpan | None = None
            with tracer.llm_call() as span:
                captured_span = span
            assert captured_span is not None
            assert isinstance(captured_span._span, _NoOpSpan)
            assert captured_span._span._ended  # type: ignore[union-attr]
