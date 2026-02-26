"""Tests for auto_instrument.generic (GenericInstrumentor)."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from agent_observability.auto_instrument.generic import GenericInstrumentor, _safe_set
from agent_observability.spans.types import AgentSpan, AgentSpanKind, AgentTracer


@pytest.fixture()
def instrumentor() -> GenericInstrumentor:
    with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
        tracer = AgentTracer()
        return GenericInstrumentor(tracer=tracer)


class TestGenericInstrumentorSync:
    def test_instrument_wraps_function(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="my-step")
        def my_func(x: int) -> int:
            return x * 2

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = my_func(5)
        assert result == 10

    def test_instrument_preserves_function_name(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument()
        def my_named_func() -> None:
            pass

        assert my_named_func.__name__ == "my_named_func"

    def test_instrument_default_name_uses_qualified_name(
        self, instrumentor: GenericInstrumentor
    ) -> None:
        @instrumentor.instrument()
        def default_name_func() -> str:
            return "ok"

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = default_name_func()
        assert result == "ok"

    def test_instrument_exception_propagates(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="will-fail")
        def failing_func() -> None:
            raise RuntimeError("deliberate error")

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="deliberate error"):
                failing_func()

    def test_instrument_capture_args(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="args-capture", capture_args=True)
        def func_with_args(a: int, b: str) -> str:
            return f"{a}{b}"

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = func_with_args(1, "x")
        assert result == "1x"

    def test_instrument_capture_result(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="result-capture", capture_result=True)
        def func_returning_value() -> int:
            return 42

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = func_returning_value()
        assert result == 42

    def test_llm_call_decorator(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.llm_call(name="my-llm", model="gpt-4o", provider="openai")
        def call_llm(prompt: str) -> str:
            return "response"

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = call_llm("hello")
        assert result == "response"

    def test_llm_call_default_name(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.llm_call()
        def llm_default() -> str:
            return "ok"

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = llm_default()
        assert result == "ok"

    def test_tool_invoke_decorator_success(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.tool_invoke(tool_name="web_search")
        def do_search(query: str) -> str:
            return f"results for {query}"

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = do_search("python testing")
        assert "results" in result

    def test_tool_invoke_decorator_on_exception(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.tool_invoke(tool_name="failing_tool")
        def do_fail() -> None:
            raise ValueError("tool error")

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with pytest.raises(ValueError):
                do_fail()

    def test_tool_invoke_custom_name(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.tool_invoke(tool_name="my_tool", name="custom-span-name")
        def invoke_tool() -> str:
            return "result"

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = invoke_tool()
        assert result == "result"

    def test_instrument_different_kinds(self, instrumentor: GenericInstrumentor) -> None:
        for kind in (AgentSpanKind.LLM_CALL, AgentSpanKind.TOOL_INVOKE, AgentSpanKind.MEMORY_READ):
            @instrumentor.instrument(name=f"test-{kind.value}", kind=kind)
            def func() -> str:
                return "ok"

            with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
                result = func()
            assert result == "ok"


class TestGenericInstrumentorAsync:
    def test_async_instrument_wraps_coroutine(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="async-step")
        async def async_func(x: int) -> int:
            return x + 1

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = asyncio.get_event_loop().run_until_complete(async_func(5))
        assert result == 6

    def test_async_instrument_exception_propagates(
        self, instrumentor: GenericInstrumentor
    ) -> None:
        @instrumentor.instrument(name="async-fail")
        async def async_failing() -> None:
            raise TypeError("async error")

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            with pytest.raises(TypeError, match="async error"):
                asyncio.get_event_loop().run_until_complete(async_failing())

    def test_async_instrument_capture_args(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="async-args", capture_args=True)
        async def async_with_args(a: int) -> int:
            return a

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = asyncio.get_event_loop().run_until_complete(async_with_args(42))
        assert result == 42

    def test_async_instrument_capture_result(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument(name="async-result", capture_result=True)
        async def async_result_func() -> bool:
            return True

        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            result = asyncio.get_event_loop().run_until_complete(async_result_func())
        assert result is True

    def test_async_preserves_function_name(self, instrumentor: GenericInstrumentor) -> None:
        @instrumentor.instrument()
        async def my_async_func() -> None:
            pass

        assert my_async_func.__name__ == "my_async_func"


class TestDefaultInstrumentorTracer:
    def test_default_tracer_created_when_none_provided(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            instrumentor = GenericInstrumentor()
            assert instrumentor._tracer is not None


class TestSafeSet:
    def test_safe_set_string(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = AgentTracer()
            with tracer._managed_span("test", AgentSpanKind.REASONING_STEP) as span:
                _safe_set(span, "key", "value")
                assert span._span._attributes.get("key") == "value"

    def test_safe_set_int(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = AgentTracer()
            with tracer._managed_span("test", AgentSpanKind.REASONING_STEP) as span:
                _safe_set(span, "count", 42)
                assert span._span._attributes.get("count") == 42

    def test_safe_set_bool(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = AgentTracer()
            with tracer._managed_span("test", AgentSpanKind.REASONING_STEP) as span:
                _safe_set(span, "flag", True)
                assert span._span._attributes.get("flag") is True

    def test_safe_set_non_primitive_stringified(self) -> None:
        with patch("agent_observability.spans.types._OTEL_AVAILABLE", False):
            tracer = AgentTracer()
            with tracer._managed_span("test", AgentSpanKind.REASONING_STEP) as span:
                _safe_set(span, "obj", {"nested": "dict"})
                val = span._span._attributes.get("obj")
                assert isinstance(val, str)
