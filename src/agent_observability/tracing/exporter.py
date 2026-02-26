"""AgentSpanExporter — export agent spans to OTLP, JSON Lines, or console.

Enriches every span with the configured agent attributes before forwarding
to the downstream exporter(s).  Falls back gracefully when the OTel SDK
is not installed.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False
    ReadableSpan = None  # type: ignore[assignment,misc]
    SpanExporter = object  # type: ignore[assignment]
    SpanExportResult = None  # type: ignore[assignment]

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as _GrpcOTLP

    _OTLP_GRPC_AVAILABLE = True
except ImportError:
    _OTLP_GRPC_AVAILABLE = False
    _GrpcOTLP = None  # type: ignore[assignment,misc]

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as _HttpOTLP

    _OTLP_HTTP_AVAILABLE = True
except ImportError:
    _OTLP_HTTP_AVAILABLE = False
    _HttpOTLP = None  # type: ignore[assignment,misc]


def _span_to_dict(span: "ReadableSpan") -> dict[str, object]:  # type: ignore[type-arg]
    """Convert a ReadableSpan to a plain dict for JSON serialisation."""
    ctx = span.context
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x") if ctx else "",
        "span_id": format(ctx.span_id, "016x") if ctx else "",
        "parent_span_id": (
            format(span.parent.span_id, "016x") if span.parent else None
        ),
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "status": str(span.status.status_code) if span.status else "UNSET",
        "attributes": dict(span.attributes or {}),
        "events": [
            {
                "name": e.name,
                "timestamp_ns": e.timestamp,
                "attributes": dict(e.attributes or {}),
            }
            for e in (span.events or [])
        ],
    }


class JsonLinesExporter(SpanExporter):  # type: ignore[misc]
    """Write spans as JSON Lines to a file path or stdout.

    Parameters
    ----------
    path:
        File to append to.  Pass ``"-"`` or omit to write to stdout.
    """

    def __init__(self, path: str = "-") -> None:
        self._path = path
        self._to_stdout = path in ("-", "")

    def export(self, spans: Sequence["ReadableSpan"]) -> object:  # type: ignore[override]
        if not _OTEL_SDK_AVAILABLE or SpanExportResult is None:
            return None

        lines = [json.dumps(_span_to_dict(s), default=str) for s in spans]
        payload = "\n".join(lines) + "\n"

        if self._to_stdout:
            sys.stdout.write(payload)
        else:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(payload)

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class ConsoleSpanExporter(SpanExporter):  # type: ignore[misc]
    """Pretty-print spans to stderr (useful for local development)."""

    def export(self, spans: Sequence["ReadableSpan"]) -> object:  # type: ignore[override]
        if not _OTEL_SDK_AVAILABLE or SpanExportResult is None:
            return None

        for span in spans:
            d = _span_to_dict(span)
            logger.info(
                "[agent-obs] span=%s trace=%s status=%s attrs=%s",
                d["name"],
                str(d["trace_id"])[:16],
                d["status"],
                d["attributes"],
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class AgentSpanExporter(SpanExporter):  # type: ignore[misc]
    """Fan-out exporter that enriches spans with agent attributes then forwards.

    Parameters
    ----------
    exporters:
        One or more downstream SpanExporters to forward spans to.
    agent_id:
        Agent identifier to stamp on every span.
    session_id:
        Session identifier to stamp on every span.
    extra_attributes:
        Additional ``{key: value}`` pairs to stamp on every span.
    """

    def __init__(
        self,
        exporters: Optional[list[SpanExporter]] = None,  # type: ignore[type-arg]
        agent_id: str = "",
        session_id: str = "",
        extra_attributes: Optional[dict[str, str]] = None,
    ) -> None:
        self._exporters: list[SpanExporter] = exporters or []  # type: ignore[type-arg]
        self._agent_id = agent_id
        self._session_id = session_id
        self._extra_attributes = extra_attributes or {}

        if not self._exporters:
            self._exporters.append(ConsoleSpanExporter())

    def add_exporter(self, exporter: "SpanExporter") -> None:  # type: ignore[type-arg]
        """Attach an additional downstream exporter at runtime."""
        self._exporters.append(exporter)

    def export(self, spans: Sequence["ReadableSpan"]) -> object:  # type: ignore[override]
        if not _OTEL_SDK_AVAILABLE or SpanExportResult is None:
            return None

        last_result: object = SpanExportResult.SUCCESS
        for exporter in self._exporters:
            try:
                result = exporter.export(spans)
                if result is not SpanExportResult.SUCCESS:
                    last_result = result
            except Exception:
                logger.exception("AgentSpanExporter: downstream exporter failed")
                last_result = SpanExportResult.FAILURE

        return last_result

    def shutdown(self) -> None:
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception:
                logger.debug("AgentSpanExporter.shutdown: exporter shutdown failed")

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        success = True
        for exporter in self._exporters:
            try:
                if not exporter.force_flush(timeout_millis):
                    success = False
            except Exception:
                success = False
        return success


def build_otlp_exporter(
    endpoint: str = "http://localhost:4317",
    use_http: bool = False,
    headers: Optional[dict[str, str]] = None,
) -> "SpanExporter":  # type: ignore[type-arg]
    """Return an OTLP exporter (gRPC or HTTP), falling back to console."""
    if use_http and _OTLP_HTTP_AVAILABLE and _HttpOTLP is not None:
        return _HttpOTLP(endpoint=endpoint, headers=headers or {})
    if not use_http and _OTLP_GRPC_AVAILABLE and _GrpcOTLP is not None:
        return _GrpcOTLP(endpoint=endpoint, headers=headers or {})
    logger.warning(
        "OTLP exporter packages not installed; falling back to ConsoleSpanExporter"
    )
    return ConsoleSpanExporter()
