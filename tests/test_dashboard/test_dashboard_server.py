"""Tests for agent_observability.dashboard.server."""
from __future__ import annotations

import io
import json
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from agent_observability.dashboard.server import (
    DashboardDataSource,
    DashboardServer,
    _build_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> DashboardDataSource:
    source = DashboardDataSource()
    source.add_trace({
        "agent_id": "agent-1",
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.002,
    })
    source.add_span({
        "trace_id": "trace-001",
        "name": "llm_call",
        "duration_ms": 250.0,
    })
    source.add_span({
        "trace_id": "trace-001",
        "name": "retrieval",
        "duration_ms": 80.0,
    })
    source.add_cost_record({
        "model": "gpt-4o",
        "cost_usd": 0.002,
    })
    return source


def _call_get(path: str, source: DashboardDataSource | None = None) -> bytes:
    """Execute do_GET for *path* and return raw response bytes."""
    if source is None:
        source = _make_source()
    handler_cls = _build_handler(source)
    output = io.BytesIO()
    request = MagicMock()
    srv = MagicMock()
    srv.server_address = ("127.0.0.1", 8081)
    handler = handler_cls.__new__(handler_cls)
    handler.request = request
    handler.client_address = ("127.0.0.1", 9999)
    handler.server = srv
    handler.rfile = io.BytesIO(b"")
    handler.wfile = output
    handler.path = path
    # Required by BaseHTTPRequestHandler.send_response / send_header
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.close_connection = True
    handler.do_GET()
    return output.getvalue()


def _parse_json(path: str, source: DashboardDataSource | None = None) -> dict[str, object]:
    raw = _call_get(path, source)
    body_start = raw.find(b"\r\n\r\n")
    body = raw[body_start + 4:] if body_start != -1 else raw[raw.find(b"\n\n") + 2:]
    return json.loads(body)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DashboardDataSource unit tests
# ---------------------------------------------------------------------------


class TestDashboardDataSource:
    def test_add_trace_assigns_id(self) -> None:
        source = DashboardDataSource()
        trace_id = source.add_trace({"agent_id": "a1"})
        assert trace_id
        assert source.trace_count == 1

    def test_add_span_assigns_id(self) -> None:
        source = DashboardDataSource()
        span_id = source.add_span({"name": "test"})
        assert span_id
        assert source.span_count == 1

    def test_get_traces_returns_correct_count(self) -> None:
        source = _make_source()
        traces = source.get_traces(limit=10)
        assert len(traces) == 1

    def test_get_spans_filter_by_trace_id(self) -> None:
        source = _make_source()
        spans = source.get_spans(trace_id="trace-001")
        assert len(spans) == 2
        assert all(s["trace_id"] == "trace-001" for s in spans)

    def test_get_cost_summary_totals(self) -> None:
        source = _make_source()
        summary = source.get_cost_summary()
        assert summary["total_usd"] > 0
        assert "gpt-4o" in summary["by_model"]

    def test_latency_histogram_populated(self) -> None:
        source = _make_source()
        hist = source.get_latency_histogram(buckets=5)
        assert "buckets" in hist
        assert "counts" in hist
        assert hist["p95"] > 0

    def test_max_traces_eviction(self) -> None:
        source = DashboardDataSource(max_traces=3)
        for i in range(5):
            source.add_trace({"agent_id": f"a{i}"})
        assert source.trace_count == 3


# ---------------------------------------------------------------------------
# HTTP handler — root / static files
# ---------------------------------------------------------------------------


class TestStaticServing:
    def test_root_returns_html(self) -> None:
        raw = _call_get("/")
        assert b"text/html" in raw or b"<!DOCTYPE" in raw or b"<html" in raw

    def test_index_html_explicit(self) -> None:
        raw = _call_get("/index.html")
        # Returns the HTML file content
        assert b"html" in raw.lower()

    def test_styles_css_content_type(self) -> None:
        raw = _call_get("/styles.css")
        assert b"text/css" in raw or b"--bg" in raw

    def test_app_js_content_type(self) -> None:
        raw = _call_get("/app.js")
        assert b"javascript" in raw or b"function" in raw or b"'use strict'" in raw


# ---------------------------------------------------------------------------
# HTTP handler — health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_status_ok(self) -> None:
        data = _parse_json("/health")
        assert data["status"] == "ok"

    def test_health_service_name(self) -> None:
        data = _parse_json("/health")
        assert data["service"] == "agent-observability-dashboard"


# ---------------------------------------------------------------------------
# HTTP handler — API endpoints
# ---------------------------------------------------------------------------


class TestApiTraces:
    def test_traces_endpoint_returns_list(self) -> None:
        data = _parse_json("/api/traces")
        assert "traces" in data
        assert isinstance(data["traces"], list)

    def test_traces_count_is_correct(self) -> None:
        data = _parse_json("/api/traces")
        assert data["count"] == 1

    def test_traces_filter_by_agent_id(self) -> None:
        source = _make_source()
        source.add_trace({"agent_id": "agent-2"})
        data = _parse_json("/api/traces?agent_id=agent-2", source)
        assert data["count"] == 1
        assert data["traces"][0]["agent_id"] == "agent-2"


class TestApiSpans:
    def test_spans_endpoint_returns_list(self) -> None:
        data = _parse_json("/api/spans")
        assert "spans" in data
        assert data["count"] == 2

    def test_spans_filter_by_trace_id(self) -> None:
        data = _parse_json("/api/spans?trace_id=trace-001")
        assert data["count"] == 2

    def test_spans_filter_nonexistent_trace(self) -> None:
        data = _parse_json("/api/spans?trace_id=no-such-trace")
        assert data["count"] == 0


class TestApiCosts:
    def test_costs_returns_total(self) -> None:
        data = _parse_json("/api/costs")
        assert "total_usd" in data
        assert data["total_usd"] > 0

    def test_costs_by_model_populated(self) -> None:
        data = _parse_json("/api/costs")
        assert "by_model" in data
        assert "gpt-4o" in data["by_model"]


class TestApiLatency:
    def test_latency_returns_histogram(self) -> None:
        data = _parse_json("/api/latency")
        assert "buckets" in data
        assert "counts" in data
        assert "p95" in data

    def test_latency_custom_buckets(self) -> None:
        data = _parse_json("/api/latency?buckets=5")
        assert len(data["buckets"]) == 5


class TestNotFound:
    def test_unknown_path_is_404(self) -> None:
        raw = _call_get("/api/does-not-exist")
        assert b"404" in raw


# ---------------------------------------------------------------------------
# DashboardServer — server instantiation
# ---------------------------------------------------------------------------


class TestDashboardServer:
    def test_server_instantiation(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source)
        assert server.address == "127.0.0.1:8081"

    def test_build_server_returns_http_server(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source, port=0)
        http_server = server.build_server()
        try:
            assert isinstance(http_server, HTTPServer)
        finally:
            http_server.server_close()

    def test_custom_host_and_port(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source, host="0.0.0.0", port=9090)
        assert server.address == "0.0.0.0:9090"

    def test_shutdown_when_no_server_built(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source)
        server.shutdown()  # should not raise
