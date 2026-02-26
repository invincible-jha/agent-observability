"""Tests for tracing.exporter (JsonLinesExporter, ConsoleSpanExporter,
AgentSpanExporter, build_otlp_exporter) and tracing.tracer
(AgentTracerProvider, setup_tracing, _NoOpTracer, _NoOpSpanHandle)."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.tracing.exporter import (
    AgentSpanExporter,
    ConsoleSpanExporter,
    JsonLinesExporter,
    build_otlp_exporter,
)
from agent_observability.tracing.tracer import (
    AgentTracerProvider,
    _NoOpSpanHandle,
    _NoOpTracer,
    setup_tracing,
)


# ── Helper ─────────────────────────────────────────────────────────────────────


def _make_mock_span(name: str = "test-span") -> MagicMock:
    """Build a mock ReadableSpan-like object for exporter tests."""
    ctx = MagicMock()
    ctx.trace_id = 0xABCD1234ABCD1234ABCD1234ABCD1234
    ctx.span_id = 0x1234ABCD1234ABCD
    span = MagicMock()
    span.name = name
    span.context = ctx
    span.parent = None
    span.start_time = 1_000_000_000
    span.end_time = 1_100_000_000
    span.status = MagicMock()
    span.status.status_code = "OK"
    span.attributes = {"agent.id": "agent-1"}
    span.events = []
    return span


# ── JsonLinesExporter ──────────────────────────────────────────────────────────


class TestJsonLinesExporter:
    def test_export_to_file(self, tmp_path: Path) -> None:
        output_file = str(tmp_path / "spans.jsonl")
        exporter = JsonLinesExporter(path=output_file)
        mock_span = _make_mock_span("my-span")

        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", True):
            mock_result = MagicMock()
            mock_result.SUCCESS = "SUCCESS"
            with patch("agent_observability.tracing.exporter.SpanExportResult", mock_result):
                exporter.export([mock_span])

        assert Path(output_file).exists()
        content = Path(output_file).read_text(encoding="utf-8")
        assert "my-span" in content

    def test_export_no_otel_returns_none(self) -> None:
        exporter = JsonLinesExporter()
        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", False):
            result = exporter.export([_make_mock_span()])
        assert result is None

    def test_export_to_stdout(self, capsys: Any) -> None:
        exporter = JsonLinesExporter(path="-")
        mock_span = _make_mock_span("stdout-span")

        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", True):
            mock_result = MagicMock()
            mock_result.SUCCESS = "SUCCESS"
            with patch("agent_observability.tracing.exporter.SpanExportResult", mock_result):
                exporter.export([mock_span])

        captured = capsys.readouterr()
        assert "stdout-span" in captured.out

    def test_export_empty_path_is_stdout(self) -> None:
        exporter = JsonLinesExporter(path="")
        assert exporter._to_stdout is True

    def test_shutdown_noop(self) -> None:
        exporter = JsonLinesExporter()
        exporter.shutdown()  # should not raise

    def test_force_flush_returns_true(self) -> None:
        exporter = JsonLinesExporter()
        assert exporter.force_flush() is True

    def test_export_appends_to_existing_file(self, tmp_path: Path) -> None:
        output_file = str(tmp_path / "spans.jsonl")
        Path(output_file).write_text("existing\n", encoding="utf-8")
        exporter = JsonLinesExporter(path=output_file)
        mock_span = _make_mock_span("new-span")

        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", True):
            mock_result = MagicMock()
            mock_result.SUCCESS = "SUCCESS"
            with patch("agent_observability.tracing.exporter.SpanExportResult", mock_result):
                exporter.export([mock_span])

        content = Path(output_file).read_text(encoding="utf-8")
        assert "existing" in content
        assert "new-span" in content


# ── ConsoleSpanExporter ────────────────────────────────────────────────────────


class TestConsoleSpanExporter:
    def test_export_no_otel_returns_none(self) -> None:
        exporter = ConsoleSpanExporter()
        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", False):
            result = exporter.export([_make_mock_span()])
        assert result is None

    def test_export_logs_spans(self) -> None:
        exporter = ConsoleSpanExporter()
        mock_span = _make_mock_span("logged-span")

        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", True):
            mock_result = MagicMock()
            mock_result.SUCCESS = "SUCCESS"
            with patch("agent_observability.tracing.exporter.SpanExportResult", mock_result):
                with patch("agent_observability.tracing.exporter.logger") as mock_logger:
                    exporter.export([mock_span])
                    mock_logger.info.assert_called()

    def test_shutdown_noop(self) -> None:
        exporter = ConsoleSpanExporter()
        exporter.shutdown()  # should not raise

    def test_force_flush_returns_true(self) -> None:
        exporter = ConsoleSpanExporter()
        assert exporter.force_flush() is True


# ── AgentSpanExporter ──────────────────────────────────────────────────────────


class TestAgentSpanExporter:
    def test_default_creates_console_exporter(self) -> None:
        exporter = AgentSpanExporter()
        assert len(exporter._exporters) == 1
        assert isinstance(exporter._exporters[0], ConsoleSpanExporter)

    def test_add_exporter(self) -> None:
        exporter = AgentSpanExporter(exporters=[ConsoleSpanExporter()])
        new_exp = ConsoleSpanExporter()
        exporter.add_exporter(new_exp)
        assert len(exporter._exporters) == 2

    def test_export_no_otel_returns_none(self) -> None:
        exporter = AgentSpanExporter()
        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", False):
            result = exporter.export([_make_mock_span()])
        assert result is None

    def test_export_forwards_to_downstream(self) -> None:
        mock_downstream = MagicMock()
        mock_downstream.export.return_value = "SUCCESS"
        exporter = AgentSpanExporter(exporters=[mock_downstream])

        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", True):
            mock_result = MagicMock()
            mock_result.SUCCESS = "SUCCESS"
            with patch("agent_observability.tracing.exporter.SpanExportResult", mock_result):
                exporter.export([_make_mock_span()])
                mock_downstream.export.assert_called_once()

    def test_export_handles_downstream_exception(self) -> None:
        mock_downstream = MagicMock()
        mock_downstream.export.side_effect = RuntimeError("exporter failed")
        exporter = AgentSpanExporter(exporters=[mock_downstream])

        with patch("agent_observability.tracing.exporter._OTEL_SDK_AVAILABLE", True):
            mock_result = MagicMock()
            mock_result.SUCCESS = "SUCCESS"
            mock_result.FAILURE = "FAILURE"
            with patch("agent_observability.tracing.exporter.SpanExportResult", mock_result):
                result = exporter.export([_make_mock_span()])
                assert result == "FAILURE"

    def test_shutdown_calls_all_exporters(self) -> None:
        mock_a = MagicMock()
        mock_b = MagicMock()
        exporter = AgentSpanExporter(exporters=[mock_a, mock_b])
        exporter.shutdown()
        mock_a.shutdown.assert_called_once()
        mock_b.shutdown.assert_called_once()

    def test_shutdown_handles_exporter_exception(self) -> None:
        mock_exp = MagicMock()
        mock_exp.shutdown.side_effect = RuntimeError("shutdown error")
        exporter = AgentSpanExporter(exporters=[mock_exp])
        exporter.shutdown()  # should not raise

    def test_force_flush_all_success(self) -> None:
        mock_a = MagicMock()
        mock_a.force_flush.return_value = True
        mock_b = MagicMock()
        mock_b.force_flush.return_value = True
        exporter = AgentSpanExporter(exporters=[mock_a, mock_b])
        assert exporter.force_flush() is True

    def test_force_flush_partial_failure(self) -> None:
        mock_a = MagicMock()
        mock_a.force_flush.return_value = True
        mock_b = MagicMock()
        mock_b.force_flush.return_value = False
        exporter = AgentSpanExporter(exporters=[mock_a, mock_b])
        assert exporter.force_flush() is False

    def test_force_flush_exception_returns_false(self) -> None:
        mock_exp = MagicMock()
        mock_exp.force_flush.side_effect = RuntimeError("flush failed")
        exporter = AgentSpanExporter(exporters=[mock_exp])
        assert exporter.force_flush() is False

    def test_agent_id_session_id_stored(self) -> None:
        exporter = AgentSpanExporter(agent_id="agent-1", session_id="session-1")
        assert exporter._agent_id == "agent-1"
        assert exporter._session_id == "session-1"


# ── build_otlp_exporter ────────────────────────────────────────────────────────


class TestBuildOtlpExporter:
    def test_returns_console_when_no_otlp_available(self) -> None:
        with patch("agent_observability.tracing.exporter._OTLP_GRPC_AVAILABLE", False):
            with patch("agent_observability.tracing.exporter._OTLP_HTTP_AVAILABLE", False):
                result = build_otlp_exporter()
        assert isinstance(result, ConsoleSpanExporter)

    def test_returns_http_when_available(self) -> None:
        mock_http_cls = MagicMock()
        mock_http_instance = MagicMock()
        mock_http_cls.return_value = mock_http_instance

        with patch("agent_observability.tracing.exporter._OTLP_HTTP_AVAILABLE", True):
            with patch("agent_observability.tracing.exporter._HttpOTLP", mock_http_cls):
                result = build_otlp_exporter(endpoint="http://localhost:4318", use_http=True)
        assert result is mock_http_instance

    def test_returns_grpc_when_available(self) -> None:
        mock_grpc_cls = MagicMock()
        mock_grpc_instance = MagicMock()
        mock_grpc_cls.return_value = mock_grpc_instance

        with patch("agent_observability.tracing.exporter._OTLP_GRPC_AVAILABLE", True):
            with patch("agent_observability.tracing.exporter._GrpcOTLP", mock_grpc_cls):
                result = build_otlp_exporter(endpoint="http://localhost:4317", use_http=False)
        assert result is mock_grpc_instance


# ── _NoOpTracer / _NoOpSpanHandle ──────────────────────────────────────────────


class TestNoOpTracer:
    def test_start_span_returns_handle(self) -> None:
        tracer = _NoOpTracer()
        handle = tracer.start_span("test-span")
        assert isinstance(handle, _NoOpSpanHandle)
        assert handle.name == "test-span"

    def test_start_as_current_span_is_context_manager(self) -> None:
        tracer = _NoOpTracer()
        ctx = tracer.start_as_current_span("test-span")
        with ctx as span:
            assert isinstance(span, _NoOpSpanHandle)


class TestNoOpSpanHandle:
    def test_set_attribute_noop(self) -> None:
        handle = _NoOpSpanHandle("test")
        handle.set_attribute("key", "value")  # should not raise

    def test_record_exception_noop(self) -> None:
        handle = _NoOpSpanHandle("test")
        handle.record_exception(RuntimeError("err"))  # should not raise

    def test_set_status_noop(self) -> None:
        handle = _NoOpSpanHandle("test")
        handle.set_status("OK")  # should not raise

    def test_end_noop(self) -> None:
        handle = _NoOpSpanHandle("test")
        handle.end()  # should not raise

    def test_context_manager(self) -> None:
        handle = _NoOpSpanHandle("test")
        with handle as span:
            assert span is handle


# ── AgentTracerProvider ────────────────────────────────────────────────────────


class TestAgentTracerProvider:
    def test_build_no_otel_returns_none(self) -> None:
        provider = AgentTracerProvider(service_name="test-svc")
        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", False):
            result = provider.build()
        assert result is None

    def test_get_tracer_no_otel_returns_noop(self) -> None:
        provider = AgentTracerProvider()
        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", False):
            tracer = provider.get_tracer()
        assert isinstance(tracer, _NoOpTracer)

    def test_get_tracer_with_otel(self) -> None:
        provider = AgentTracerProvider()
        mock_trace = MagicMock()
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", True):
            with patch("agent_observability.tracing.tracer.otel_trace", mock_trace):
                tracer = provider.get_tracer("my-lib")
        assert tracer is mock_tracer

    def test_shutdown_no_provider(self) -> None:
        provider = AgentTracerProvider()
        provider.shutdown()  # should not raise when _provider is None

    def test_shutdown_calls_provider_shutdown(self) -> None:
        provider = AgentTracerProvider()
        mock_otel_provider = MagicMock()
        provider._provider = mock_otel_provider
        provider.shutdown()
        mock_otel_provider.shutdown.assert_called_once()

    def test_build_with_otel_sdk(self) -> None:
        mock_resource = MagicMock()
        mock_resource.create.return_value = MagicMock()
        mock_provider_cls = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_cls.return_value = mock_provider_instance
        mock_batch_processor = MagicMock()
        mock_simple_processor = MagicMock()
        mock_otel_trace = MagicMock()

        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", True):
            with patch("agent_observability.tracing.tracer.Resource", mock_resource):
                with patch("agent_observability.tracing.tracer.TracerProvider", mock_provider_cls):
                    with patch("agent_observability.tracing.tracer.BatchSpanProcessor", mock_batch_processor):
                        with patch("agent_observability.tracing.tracer.SimpleSpanProcessor", mock_simple_processor):
                            with patch("agent_observability.tracing.tracer.otel_trace", mock_otel_trace):
                                provider = AgentTracerProvider(
                                    service_name="test-svc",
                                    agent_id="agent-1",
                                )
                                mock_exporter = MagicMock()
                                result = provider.build(exporter=mock_exporter, batch=True)
        assert result is mock_provider_instance


# ── setup_tracing ──────────────────────────────────────────────────────────────


class TestSetupTracing:
    def test_returns_agent_tracer_provider(self) -> None:
        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", False):
            result = setup_tracing(service_name="test")
        assert isinstance(result, AgentTracerProvider)

    def test_console_exporter_when_no_downstream(self) -> None:
        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", False):
            provider = setup_tracing(service_name="test")
        # With no OTel SDK, build returns None but provider is created
        assert provider._service_name == "test"

    def test_jsonl_path_creates_jsonl_exporter(self) -> None:
        from agent_observability.tracing.exporter import JsonLinesExporter

        created_exporters: list[Any] = []

        original_init = AgentSpanExporter.__init__

        def capturing_init(self: AgentSpanExporter, exporters: Any = None, **kwargs: Any) -> None:
            if exporters:
                created_exporters.extend(exporters)
            original_init(self, exporters=exporters, **kwargs)

        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", False):
            with patch.object(AgentSpanExporter, "__init__", capturing_init):
                setup_tracing(service_name="test", jsonl_path="/tmp/spans.jsonl")

        jsonl_exporters = [e for e in created_exporters if isinstance(e, JsonLinesExporter)]
        assert len(jsonl_exporters) == 1

    def test_console_flag_adds_console_exporter(self) -> None:
        created_exporters: list[Any] = []

        original_init = AgentSpanExporter.__init__

        def capturing_init(self: AgentSpanExporter, exporters: Any = None, **kwargs: Any) -> None:
            if exporters:
                created_exporters.extend(exporters)
            original_init(self, exporters=exporters, **kwargs)

        with patch("agent_observability.tracing.tracer._OTEL_SDK_AVAILABLE", False):
            with patch.object(AgentSpanExporter, "__init__", capturing_init):
                setup_tracing(service_name="test", console=True)

        console_exporters = [e for e in created_exporters if isinstance(e, ConsoleSpanExporter)]
        assert len(console_exporters) >= 1
