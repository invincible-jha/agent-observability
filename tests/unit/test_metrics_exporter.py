"""Tests for metrics.exporter (PrometheusExporter, _render_plain_text, _labels_to_str)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.metrics.collector import AgentMetricCollector
from agent_observability.metrics.exporter import (
    PrometheusExporter,
    _labels_to_str,
    _render_plain_text,
)


# ── _labels_to_str ─────────────────────────────────────────────────────────────


class TestLabelsToStr:
    def test_empty_labels_returns_empty_string(self) -> None:
        assert _labels_to_str({}) == ""

    def test_single_label(self) -> None:
        result = _labels_to_str({"model": "gpt-4o"})
        assert result == '{model="gpt-4o"}'

    def test_multiple_labels_sorted(self) -> None:
        result = _labels_to_str({"le": "0.5", "model": "gpt-4o"})
        # Keys are sorted alphabetically
        assert result.index("le") < result.index("model")
        assert 'le="0.5"' in result
        assert 'model="gpt-4o"' in result

    def test_special_chars_in_value_preserved(self) -> None:
        result = _labels_to_str({"status": "ok/200"})
        assert 'status="ok/200"' in result


# ── _render_plain_text ──────────────────────────────────────────────────────────


def _make_collector_with_counter(
    name: str = "test_counter",
    labels: dict[str, str] | None = None,
    value: float = 5.0,
) -> AgentMetricCollector:
    """Build a minimal collector whose snapshot has one counter entry."""
    collector = MagicMock(spec=AgentMetricCollector)
    collector.snapshot.return_value = {
        "counters": [{"name": name, "labels": labels or {}, "value": value}],
        "gauges": [],
        "histograms": [],
    }
    return collector


def _make_collector_with_gauge(
    name: str = "test_gauge",
    labels: dict[str, str] | None = None,
    value: float = 3.0,
) -> AgentMetricCollector:
    collector = MagicMock(spec=AgentMetricCollector)
    collector.snapshot.return_value = {
        "counters": [],
        "gauges": [{"name": name, "labels": labels or {}, "value": value}],
        "histograms": [],
    }
    return collector


def _make_collector_with_histogram(
    name: str = "test_histogram",
    labels: dict[str, str] | None = None,
    count: int = 2,
    total_sum: float = 1.5,
    buckets: list[tuple[float, int]] | None = None,
) -> AgentMetricCollector:
    collector = MagicMock(spec=AgentMetricCollector)
    collector.snapshot.return_value = {
        "counters": [],
        "gauges": [],
        "histograms": [
            {
                "name": name,
                "labels": labels or {},
                "count": count,
                "sum": total_sum,
                "buckets": buckets or [(0.5, 1), (1.0, 2)],
            }
        ],
    }
    return collector


class TestRenderPlainText:
    def test_counter_type_header_emitted(self) -> None:
        collector = _make_collector_with_counter()
        output = _render_plain_text(collector)
        assert "# TYPE test_counter counter" in output

    def test_counter_value_line_emitted(self) -> None:
        collector = _make_collector_with_counter(value=42.0)
        output = _render_plain_text(collector)
        assert "test_counter 42.0" in output

    def test_counter_with_labels(self) -> None:
        collector = _make_collector_with_counter(labels={"model": "gpt-4o"}, value=3.0)
        output = _render_plain_text(collector)
        assert 'model="gpt-4o"' in output

    def test_gauge_type_header_emitted(self) -> None:
        collector = _make_collector_with_gauge()
        output = _render_plain_text(collector)
        assert "# TYPE test_gauge gauge" in output

    def test_gauge_value_line_emitted(self) -> None:
        collector = _make_collector_with_gauge(value=7.0)
        output = _render_plain_text(collector)
        assert "test_gauge 7.0" in output

    def test_histogram_type_header_emitted(self) -> None:
        collector = _make_collector_with_histogram()
        output = _render_plain_text(collector)
        assert "# TYPE test_histogram histogram" in output

    def test_histogram_bucket_lines_emitted(self) -> None:
        collector = _make_collector_with_histogram(
            buckets=[(0.5, 1), (1.0, 2)]
        )
        output = _render_plain_text(collector)
        assert "_bucket" in output

    def test_histogram_inf_bucket_emitted(self) -> None:
        collector = _make_collector_with_histogram(count=5, buckets=[(0.5, 3)])
        output = _render_plain_text(collector)
        assert '+Inf"' in output or "le=+Inf" in output or "+Inf" in output

    def test_histogram_count_and_sum_lines(self) -> None:
        collector = _make_collector_with_histogram(count=3, total_sum=2.7)
        output = _render_plain_text(collector)
        assert "_count" in output
        assert "_sum" in output
        assert "3" in output

    def test_counter_type_header_appears_once_for_same_name(self) -> None:
        collector = MagicMock(spec=AgentMetricCollector)
        collector.snapshot.return_value = {
            "counters": [
                {"name": "my_counter", "labels": {"model": "a"}, "value": 1.0},
                {"name": "my_counter", "labels": {"model": "b"}, "value": 2.0},
            ],
            "gauges": [],
            "histograms": [],
        }
        output = _render_plain_text(collector)
        assert output.count("# TYPE my_counter counter") == 1

    def test_gauge_type_header_appears_once_for_same_name(self) -> None:
        collector = MagicMock(spec=AgentMetricCollector)
        collector.snapshot.return_value = {
            "counters": [],
            "gauges": [
                {"name": "my_gauge", "labels": {"region": "us"}, "value": 1.0},
                {"name": "my_gauge", "labels": {"region": "eu"}, "value": 2.0},
            ],
            "histograms": [],
        }
        output = _render_plain_text(collector)
        assert output.count("# TYPE my_gauge gauge") == 1

    def test_output_ends_with_newline(self) -> None:
        collector = _make_collector_with_counter()
        output = _render_plain_text(collector)
        assert output.endswith("\n")

    def test_empty_collector(self) -> None:
        collector = MagicMock(spec=AgentMetricCollector)
        collector.snapshot.return_value = {"counters": [], "gauges": [], "histograms": []}
        output = _render_plain_text(collector)
        # Should produce just a newline (empty lines joined + newline)
        assert isinstance(output, str)

    def test_histogram_with_labels(self) -> None:
        collector = _make_collector_with_histogram(
            labels={"model": "gpt-4o"},
            buckets=[(0.5, 1)],
        )
        output = _render_plain_text(collector)
        assert "model=" in output


# ── PrometheusExporter ─────────────────────────────────────────────────────────


@pytest.fixture()
def real_collector() -> AgentMetricCollector:
    """A real AgentMetricCollector for integration tests."""
    return AgentMetricCollector()


class TestPrometheusExporterInit:
    def test_stores_collector(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector, port=9090)
        assert exporter._collector is real_collector

    def test_stores_port(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector, port=8888)
        assert exporter._port == 8888

    def test_stores_host(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector, host="127.0.0.1")
        assert exporter._host == "127.0.0.1"

    def test_default_host(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector)
        assert exporter._host == "0.0.0.0"

    def test_default_port(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector)
        assert exporter._port == 9090

    def test_use_prometheus_client_false_when_unavailable(
        self, real_collector: AgentMetricCollector
    ) -> None:
        with patch("agent_observability.metrics.exporter._PROMETHEUS_AVAILABLE", False):
            exporter = PrometheusExporter(collector=real_collector, use_prometheus_client=True)
        # Even if requested, not available
        assert exporter._use_prometheus_client is False

    def test_use_prometheus_client_false_when_disabled(
        self, real_collector: AgentMetricCollector
    ) -> None:
        exporter = PrometheusExporter(collector=real_collector, use_prometheus_client=False)
        assert exporter._use_prometheus_client is False


class TestPrometheusExporterRender:
    def test_render_returns_string(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector)
        output = exporter.render()
        assert isinstance(output, str)

    def test_render_reflects_collector_data(self, real_collector: AgentMetricCollector) -> None:
        real_collector.increment_counter("test_count", labels={"env": "prod"})
        exporter = PrometheusExporter(collector=real_collector)
        output = exporter.render()
        assert "test_count" in output


class TestPrometheusExporterStop:
    def test_stop_no_server_does_not_raise(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector)
        exporter.stop()  # _server is None, should be a no-op

    def test_stop_calls_server_shutdown(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector)
        mock_server = MagicMock()
        mock_server.shutdown = MagicMock()
        exporter._server = mock_server
        exporter.stop()
        mock_server.shutdown.assert_called_once()
        assert exporter._server is None

    def test_stop_server_without_shutdown_attr(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(collector=real_collector)
        # Object without shutdown attribute
        exporter._server = object()
        exporter.stop()  # should not raise


class TestPrometheusExporterStartBuiltin:
    def test_start_builtin_launches_thread(self, real_collector: AgentMetricCollector) -> None:
        exporter = PrometheusExporter(
            collector=real_collector,
            port=0,  # OS assigns ephemeral port
            use_prometheus_client=False,
        )
        with patch("agent_observability.metrics.exporter._PROMETHEUS_AVAILABLE", False):
            try:
                exporter._start_builtin_server()
                assert exporter._thread is not None
                assert exporter._thread.is_alive()
            finally:
                exporter.stop()

    def test_start_routes_to_builtin_when_client_unavailable(
        self, real_collector: AgentMetricCollector
    ) -> None:
        exporter = PrometheusExporter(
            collector=real_collector,
            port=0,
            use_prometheus_client=False,
        )
        with patch("agent_observability.metrics.exporter._PROMETHEUS_AVAILABLE", False):
            with patch.object(exporter, "_start_builtin_server") as mock_builtin:
                exporter.start()
                mock_builtin.assert_called_once()

    def test_start_routes_to_prometheus_client_when_available(
        self, real_collector: AgentMetricCollector
    ) -> None:
        exporter = PrometheusExporter(
            collector=real_collector,
            port=9090,
        )
        # Force _use_prometheus_client=True and make_server truthy
        exporter._use_prometheus_client = True
        with patch("agent_observability.metrics.exporter.make_server", MagicMock()):
            with patch.object(exporter, "_start_prometheus_client_server") as mock_prom:
                exporter.start()
                mock_prom.assert_called_once()
