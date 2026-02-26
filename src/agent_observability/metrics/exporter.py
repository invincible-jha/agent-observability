"""PrometheusExporter — expose agent metrics at a /metrics HTTP endpoint.

Uses ``prometheus_client`` if available; falls back to a minimal plain-text
Prometheus exposition format implemented in pure Python.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from agent_observability.metrics.collector import (
    AgentMetricCollector,
    CounterValue,
    GaugeValue,
    HistogramValue,
)

logger = logging.getLogger(__name__)

try:
    import prometheus_client
    from prometheus_client import REGISTRY, CollectorRegistry, generate_latest, make_server

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    prometheus_client = None  # type: ignore[assignment]
    REGISTRY = None  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment,misc]
    generate_latest = None  # type: ignore[assignment]
    make_server = None  # type: ignore[assignment]


def _labels_to_str(labels: dict[str, str]) -> str:
    """Render a label dict as Prometheus label selector syntax."""
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


def _render_plain_text(collector: AgentMetricCollector) -> str:
    """Render all metrics in plain Prometheus exposition format.

    This does NOT use ``prometheus_client`` — it is the pure-Python fallback.
    """
    snapshot = collector.snapshot()
    lines: list[str] = []

    # Counters
    seen_counter_names: set[str] = set()
    for entry in snapshot.get("counters", []):
        name: str = entry["name"]  # type: ignore[assignment]
        labels: dict[str, str] = entry["labels"]  # type: ignore[assignment]
        value: float = entry["value"]  # type: ignore[assignment]
        if name not in seen_counter_names:
            lines.append(f"# TYPE {name} counter")
            seen_counter_names.add(name)
        lines.append(f"{name}{_labels_to_str(labels)} {value}")

    # Gauges
    seen_gauge_names: set[str] = set()
    for entry in snapshot.get("gauges", []):
        name = entry["name"]  # type: ignore[assignment]
        labels = entry["labels"]  # type: ignore[assignment]
        value = entry["value"]  # type: ignore[assignment]
        if name not in seen_gauge_names:
            lines.append(f"# TYPE {name} gauge")
            seen_gauge_names.add(name)
        lines.append(f"{name}{_labels_to_str(labels)} {value}")

    # Histograms
    seen_hist_names: set[str] = set()
    for entry in snapshot.get("histograms", []):
        name = entry["name"]  # type: ignore[assignment]
        labels = entry["labels"]  # type: ignore[assignment]
        count: int = entry["count"]  # type: ignore[assignment]
        total_sum: float = entry["sum"]  # type: ignore[assignment]
        buckets: list[tuple[float, int]] = entry["buckets"]  # type: ignore[assignment]
        if name not in seen_hist_names:
            lines.append(f"# TYPE {name} histogram")
            seen_hist_names.add(name)
        label_str = _labels_to_str(labels)
        for upper_bound, cumulative_count in buckets:
            le_label = _labels_to_str({**labels, "le": str(upper_bound)})
            lines.append(f"{name}_bucket{le_label} {cumulative_count}")
        # +Inf bucket
        inf_label = _labels_to_str({**labels, "le": "+Inf"})
        lines.append(f"{name}_bucket{inf_label} {count}")
        lines.append(f"{name}_count{label_str} {count}")
        lines.append(f"{name}_sum{label_str} {total_sum}")

    return "\n".join(lines) + "\n"


class PrometheusExporter:
    """Expose agent metrics at an HTTP ``/metrics`` endpoint.

    Parameters
    ----------
    collector:
        The :class:`AgentMetricCollector` to read metrics from.
    port:
        TCP port for the HTTP server.
    host:
        Bind address (defaults to ``"0.0.0.0"``).
    use_prometheus_client:
        Prefer ``prometheus_client`` if available.  Set to ``False`` to force
        the plain-text fallback.
    """

    def __init__(
        self,
        collector: AgentMetricCollector,
        port: int = 9090,
        host: str = "0.0.0.0",
        use_prometheus_client: bool = True,
    ) -> None:
        self._collector = collector
        self._port = port
        self._host = host
        self._use_prometheus_client = use_prometheus_client and _PROMETHEUS_AVAILABLE
        self._server: Optional[object] = None
        self._thread: Optional[threading.Thread] = None

    # ── Text export (no HTTP server) ──────────────────────────────────────────

    def render(self) -> str:
        """Return the current metrics as a Prometheus-format text string."""
        return _render_plain_text(self._collector)

    # ── HTTP server ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the HTTP metrics server in a daemon thread."""
        if self._use_prometheus_client and make_server is not None:
            self._start_prometheus_client_server()
        else:
            self._start_builtin_server()

    def _start_prometheus_client_server(self) -> None:
        """Use prometheus_client's built-in WSGI server."""
        assert make_server is not None

        # Disable default process metrics to avoid conflicts
        try:
            import prometheus_client as pc  # type: ignore[import]

            pc.REGISTRY.unregister(pc.GC_COLLECTOR)
        except Exception:
            pass

        logger.info(
            "PrometheusExporter: starting prometheus_client server on %s:%d",
            self._host,
            self._port,
        )
        self._thread = threading.Thread(
            target=lambda: prometheus_client.start_http_server(  # type: ignore[union-attr]
                self._port, self._host
            ),
            daemon=True,
        )
        self._thread.start()

    def _start_builtin_server(self) -> None:
        """Start a minimal HTTP server serving the plain-text metrics."""
        import http.server

        collector = self._collector
        render_fn = _render_plain_text

        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    body = render_fn(collector).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt: str, *args: object) -> None:
                logger.debug("PrometheusExporter: " + fmt, *args)

        server = http.server.HTTPServer((self._host, self._port), MetricsHandler)
        self._server = server
        logger.info(
            "PrometheusExporter: starting built-in server on %s:%d",
            self._host,
            self._port,
        )
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the HTTP server if running."""
        if self._server is not None and hasattr(self._server, "shutdown"):
            self._server.shutdown()  # type: ignore[union-attr]
            self._server = None
