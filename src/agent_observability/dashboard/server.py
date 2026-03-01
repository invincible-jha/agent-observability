"""HTTP dashboard server for agent-observability.

Serves a single-page web dashboard (trace timeline, span tree, cost
breakdown, latency histogram) using only the Python standard library.
No external web frameworks are required.

Usage
-----
::

    from agent_observability.dashboard.server import DashboardServer, DashboardDataSource

    source = DashboardDataSource()
    server = DashboardServer(data_source=source, host="127.0.0.1", port=8081)
    server.start()  # blocks; Ctrl-C to stop
"""
from __future__ import annotations

import json
import mimetypes
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    pass

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# In-memory data store
# ---------------------------------------------------------------------------


class DashboardDataSource:
    """Thread-safe in-memory store for traces, spans, and cost records.

    Parameters
    ----------
    max_traces:
        Maximum number of trace records to retain (FIFO eviction).
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self._max_traces = max_traces
        self._traces: list[dict[str, object]] = []
        self._spans: list[dict[str, object]] = []
        self._cost_records: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def add_trace(self, trace: dict[str, object]) -> str:
        """Store a trace record, returning its assigned ID."""
        trace_id = str(trace.get("trace_id") or uuid.uuid4())
        record = dict(trace)
        record["trace_id"] = trace_id
        record.setdefault("timestamp", time.time())
        self._traces.append(record)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces :]
        return trace_id

    def add_span(self, span: dict[str, object]) -> str:
        """Store a span record, returning its assigned ID."""
        span_id = str(span.get("span_id") or uuid.uuid4())
        record = dict(span)
        record["span_id"] = span_id
        record.setdefault("timestamp", time.time())
        self._spans.append(record)
        return span_id

    def add_cost_record(self, record: dict[str, object]) -> None:
        """Store a cost record."""
        self._cost_records.append(dict(record))

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_traces(
        self,
        limit: int = 100,
        agent_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return recent traces, optionally filtered by agent_id."""
        traces = self._traces
        if agent_id is not None:
            traces = [t for t in traces if t.get("agent_id") == agent_id]
        return traces[-limit:]

    def get_spans(
        self,
        trace_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        """Return spans, optionally filtered by trace_id."""
        spans = self._spans
        if trace_id is not None:
            spans = [s for s in spans if s.get("trace_id") == trace_id]
        return spans[-limit:]

    def get_cost_summary(self) -> dict[str, object]:
        """Return cost totals grouped by model."""
        by_model: dict[str, float] = {}
        total_usd = 0.0
        for record in self._cost_records:
            model = str(record.get("model") or "unknown")
            cost = float(record.get("cost_usd") or 0.0)
            by_model[model] = by_model.get(model, 0.0) + cost
            total_usd += cost
        return {
            "total_usd": round(total_usd, 6),
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "record_count": len(self._cost_records),
        }

    def get_latency_histogram(self, buckets: int = 10) -> dict[str, object]:
        """Return latency histogram data from span durations."""
        durations: list[float] = [
            float(s["duration_ms"])
            for s in self._spans
            if "duration_ms" in s
        ]
        if not durations:
            return {"buckets": [], "counts": [], "min_ms": 0, "max_ms": 0, "p50": 0, "p95": 0}

        durations_sorted = sorted(durations)
        min_val = durations_sorted[0]
        max_val = durations_sorted[-1]
        bucket_width = max(1.0, (max_val - min_val) / buckets)

        counts: list[int] = [0] * buckets
        edges: list[float] = []
        for i in range(buckets):
            edges.append(round(min_val + i * bucket_width, 2))

        for d in durations:
            index = min(int((d - min_val) / bucket_width), buckets - 1)
            counts[index] += 1

        n = len(durations_sorted)
        p50 = durations_sorted[n // 2]
        p95 = durations_sorted[min(int(n * 0.95), n - 1)]

        return {
            "buckets": edges,
            "counts": counts,
            "min_ms": round(min_val, 2),
            "max_ms": round(max_val, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
        }

    @property
    def trace_count(self) -> int:
        """Total number of stored traces."""
        return len(self._traces)

    @property
    def span_count(self) -> int:
        """Total number of stored spans."""
        return len(self._spans)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


def _build_handler(data_source: DashboardDataSource) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP request handler bound to *data_source*."""

    class _Handler(BaseHTTPRequestHandler):
        _source = data_source

        def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
            pass

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            params = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path == "/app.js":
                self._serve_static("app.js")
            elif path == "/styles.css":
                self._serve_static("styles.css")
            elif path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "service": "agent-observability-dashboard",
                    "traces": self._source.trace_count,
                    "spans": self._source.span_count,
                })
            elif path == "/api/traces":
                limit = int((params.get("limit") or ["100"])[0])
                agent_id = (params.get("agent_id") or [None])[0]
                traces = self._source.get_traces(limit=limit, agent_id=agent_id)
                self._send_json(200, {"traces": traces, "count": len(traces)})
            elif path == "/api/spans":
                trace_id = (params.get("trace_id") or [None])[0]
                limit = int((params.get("limit") or ["500"])[0])
                spans = self._source.get_spans(trace_id=trace_id, limit=limit)
                self._send_json(200, {"spans": spans, "count": len(spans)})
            elif path == "/api/costs":
                self._send_json(200, self._source.get_cost_summary())
            elif path == "/api/latency":
                buckets = int((params.get("buckets") or ["10"])[0])
                self._send_json(200, self._source.get_latency_histogram(buckets=buckets))
            else:
                self._send_json(404, {"error": "Not found", "path": path})

        def _serve_static(self, filename: str) -> None:
            file_path = _STATIC_DIR / filename
            if not file_path.exists():
                self._send_json(404, {"error": f"Static file not found: {filename}"})
                return
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, data: dict[str, object]) -> None:
            body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


# ---------------------------------------------------------------------------
# Server wrapper
# ---------------------------------------------------------------------------


class DashboardServer:
    """Agent-observability web dashboard server.

    Parameters
    ----------
    data_source:
        The data source to serve dashboard data from.
    host:
        Bind host (default ``"127.0.0.1"``).
    port:
        Bind port (default ``8081``).
    """

    def __init__(
        self,
        data_source: DashboardDataSource,
        host: str = "127.0.0.1",
        port: int = 8081,
    ) -> None:
        self._data_source = data_source
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None

    def build_server(self) -> HTTPServer:
        """Build and return the underlying ``HTTPServer`` without starting it."""
        handler_cls = _build_handler(self._data_source)
        server = HTTPServer((self._host, self._port), handler_cls)
        self._server = server
        return server

    def start(self) -> None:
        """Start the HTTP server and block until interrupted."""
        server = self.build_server()
        try:
            server.serve_forever()
        finally:
            server.server_close()

    def shutdown(self) -> None:
        """Stop the server if it is running."""
        if self._server is not None:
            self._server.shutdown()

    @property
    def address(self) -> str:
        """Return the server's bind address as ``host:port``."""
        return f"{self._host}:{self._port}"


__all__ = [
    "DashboardServer",
    "DashboardDataSource",
]
