"""Test that the 3-line quickstart API works for agent-observability."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_observability import Tracer

    tracer = Tracer()
    assert tracer is not None


def test_quickstart_trace_context_manager() -> None:
    from agent_observability import Tracer

    tracer = Tracer()
    with tracer.trace("test-operation") as span:
        assert span is not None


def test_quickstart_trace_llm() -> None:
    from agent_observability import Tracer

    tracer = Tracer()
    with tracer.trace_llm("claude-sonnet-4", "anthropic") as span:
        assert span is not None


def test_quickstart_export() -> None:
    from agent_observability import Tracer

    tracer = Tracer()
    with tracer.trace("operation-a"):
        pass

    spans = tracer.export()
    assert isinstance(spans, list)
    assert len(spans) >= 1


def test_quickstart_flush() -> None:
    from agent_observability import Tracer

    tracer = Tracer()
    with tracer.trace("to-be-flushed"):
        pass

    tracer.flush()
    assert tracer.export() == []


def test_quickstart_repr() -> None:
    from agent_observability import Tracer

    tracer = Tracer(service_name="test-service", agent_id="agent-123")
    text = repr(tracer)
    assert "Tracer" in text
    assert "test-service" in text
