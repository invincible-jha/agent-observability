"""GrafanaDashboardGenerator — programmatic Grafana dashboard JSON generation.

Produces valid Grafana v8+ dashboard JSON without any Grafana API dependency.
Dashboards can be imported directly into Grafana or provisioned via the
``dashboards`` provisioning directory.

Three built-in dashboard templates are provided:

1. ``generate_overview()`` — main agent observability overview
2. ``generate_cost_dashboard()`` — LLM cost tracking
3. ``generate_drift_dashboard()`` — behavioral drift monitoring
"""
from __future__ import annotations

import json
import time
from pathlib import Path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _uid_from_title(title: str) -> str:
    """Derive a stable Grafana dashboard UID from its title."""
    import hashlib
    return hashlib.sha1(title.encode()).hexdigest()[:9]


def _panel_id_counter() -> "PanelIdCounter":
    return PanelIdCounter()


class PanelIdCounter:
    """Simple incrementing panel ID counter."""

    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        self._value += 1
        return self._value


def _timeseries_panel(
    panel_id: int,
    title: str,
    targets: list[dict[str, object]],
    grid_pos: dict[str, int],
    unit: str = "short",
    description: str = "",
) -> dict[str, object]:
    """Produce a Grafana time series panel dict."""
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": "timeseries",
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "palette-classic"},
            },
            "overrides": [],
        },
        "options": {
            "tooltip": {"mode": "single"},
            "legend": {"displayMode": "list", "placement": "bottom"},
        },
        "targets": targets,
    }


def _stat_panel(
    panel_id: int,
    title: str,
    targets: list[dict[str, object]],
    grid_pos: dict[str, int],
    unit: str = "short",
) -> dict[str, object]:
    """Produce a Grafana stat panel dict."""
    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 80},
                        {"color": "red", "value": 95},
                    ],
                },
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "background",
        },
        "targets": targets,
    }


def _prom_target(expr: str, legend_format: str = "") -> dict[str, object]:
    """Produce a Prometheus datasource target."""
    return {
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "expr": expr,
        "legendFormat": legend_format or expr,
        "refId": "A",
    }


def _dashboard_skeleton(
    title: str,
    panels: list[dict[str, object]],
    description: str = "",
    tags: list[str] | None = None,
) -> dict[str, object]:
    """Return a minimal valid Grafana dashboard JSON skeleton."""
    return {
        "id": None,
        "uid": _uid_from_title(title),
        "title": title,
        "description": description,
        "tags": tags or ["agent-observability"],
        "schemaVersion": 37,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "panels": panels,
        "templating": {
            "list": [
                {
                    "current": {},
                    "hide": 0,
                    "includeAll": False,
                    "multi": False,
                    "name": "datasource",
                    "options": [],
                    "query": "prometheus",
                    "refresh": 1,
                    "type": "datasource",
                    "label": "Prometheus",
                }
            ]
        },
        "annotations": {"list": []},
        "links": [],
    }


# ── Main class ─────────────────────────────────────────────────────────────────

class GrafanaDashboardGenerator:
    """Generate Grafana dashboard JSON for agent observability.

    Example
    -------
    >>> gen = GrafanaDashboardGenerator()
    >>> overview = gen.generate_overview()
    >>> path = gen.save(overview, Path("/tmp/overview.json"))
    """

    def generate_overview(self) -> dict[str, object]:
        """Generate the main agent observability overview dashboard.

        Panels
        ------
        * LLM calls per second
        * Tool invocations per second
        * Active spans gauge
        * Error rate
        * LLM p50/p95/p99 latency
        * Tool p50/p95 latency
        """
        counter = _panel_id_counter()

        panels: list[dict[str, object]] = [
            # Row 1 — Stats
            _stat_panel(
                panel_id=counter.next(),
                title="LLM Calls (last 5m)",
                targets=[_prom_target("sum(rate(agent_llm_calls_total[5m]))", "calls/s")],
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                unit="reqps",
            ),
            _stat_panel(
                panel_id=counter.next(),
                title="Tool Invocations (last 5m)",
                targets=[_prom_target("sum(rate(agent_tool_invocations_total[5m]))", "calls/s")],
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                unit="reqps",
            ),
            _stat_panel(
                panel_id=counter.next(),
                title="Active Spans",
                targets=[_prom_target("agent_active_spans", "active")],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                unit="short",
            ),
            _stat_panel(
                panel_id=counter.next(),
                title="Error Rate (last 5m)",
                targets=[_prom_target("sum(rate(agent_errors_total[5m]))", "errors/s")],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                unit="reqps",
            ),
            # Row 2 — LLM latency timeseries
            _timeseries_panel(
                panel_id=counter.next(),
                title="LLM Latency (p50 / p95 / p99)",
                description="Per-model LLM call latency percentiles",
                targets=[
                    {**_prom_target(
                        'histogram_quantile(0.50, sum(rate(agent_llm_latency_seconds_bucket[5m])) by (le, model))',
                        "p50 {{model}}",
                    ), "refId": "A"},
                    {**_prom_target(
                        'histogram_quantile(0.95, sum(rate(agent_llm_latency_seconds_bucket[5m])) by (le, model))',
                        "p95 {{model}}",
                    ), "refId": "B"},
                    {**_prom_target(
                        'histogram_quantile(0.99, sum(rate(agent_llm_latency_seconds_bucket[5m])) by (le, model))',
                        "p99 {{model}}",
                    ), "refId": "C"},
                ],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
                unit="s",
            ),
            # Row 2 — Tool latency timeseries
            _timeseries_panel(
                panel_id=counter.next(),
                title="Tool Latency (p50 / p95)",
                description="Per-tool invocation latency percentiles",
                targets=[
                    {**_prom_target(
                        'histogram_quantile(0.50, sum(rate(agent_tool_latency_seconds_bucket[5m])) by (le, tool_name))',
                        "p50 {{tool_name}}",
                    ), "refId": "A"},
                    {**_prom_target(
                        'histogram_quantile(0.95, sum(rate(agent_tool_latency_seconds_bucket[5m])) by (le, tool_name))',
                        "p95 {{tool_name}}",
                    ), "refId": "B"},
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
                unit="s",
            ),
            # Row 3 — Error breakdown
            _timeseries_panel(
                panel_id=counter.next(),
                title="Errors by Type",
                description="Agent errors broken down by error_type label",
                targets=[_prom_target(
                    'sum(rate(agent_errors_total[5m])) by (error_type)',
                    "{{error_type}}",
                )],
                grid_pos={"x": 0, "y": 12, "w": 24, "h": 6},
                unit="short",
            ),
        ]

        return _dashboard_skeleton(
            title="Agent Observability — Overview",
            description="High-level view of LLM calls, tool invocations, latency, and errors",
            panels=panels,
            tags=["agent-observability", "overview"],
        )

    def generate_cost_dashboard(self) -> dict[str, object]:
        """Generate the LLM cost tracking dashboard.

        Panels
        ------
        * Total accumulated cost
        * Cost per minute by model
        * Cumulative cost over time
        * Token throughput by model
        """
        counter = _panel_id_counter()

        panels: list[dict[str, object]] = [
            # Stat — total cost
            _stat_panel(
                panel_id=counter.next(),
                title="Total Accumulated Cost (USD)",
                targets=[_prom_target("agent_total_cost_usd", "total cost")],
                grid_pos={"x": 0, "y": 0, "w": 8, "h": 4},
                unit="currencyUSD",
            ),
            # Stat — LLM calls total
            _stat_panel(
                panel_id=counter.next(),
                title="Total LLM Calls",
                targets=[_prom_target("sum(agent_llm_calls_total)", "calls")],
                grid_pos={"x": 8, "y": 0, "w": 8, "h": 4},
                unit="short",
            ),
            # Stat — calls per second
            _stat_panel(
                panel_id=counter.next(),
                title="LLM Call Rate (last 5m)",
                targets=[_prom_target("sum(rate(agent_llm_calls_total[5m]))", "calls/s")],
                grid_pos={"x": 16, "y": 0, "w": 8, "h": 4},
                unit="reqps",
            ),
            # Timeseries — LLM calls by model
            _timeseries_panel(
                panel_id=counter.next(),
                title="LLM Call Rate by Model",
                description="Calls per second broken down by model",
                targets=[_prom_target(
                    'sum(rate(agent_llm_calls_total[5m])) by (model)',
                    "{{model}}",
                )],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
                unit="reqps",
            ),
            # Timeseries — total cost gauge over time
            _timeseries_panel(
                panel_id=counter.next(),
                title="Cumulative Cost (USD)",
                description="Running total LLM spend",
                targets=[_prom_target("agent_total_cost_usd", "total USD")],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
                unit="currencyUSD",
            ),
        ]

        return _dashboard_skeleton(
            title="Agent Observability — Cost Tracking",
            description="LLM cost attribution and spending trends",
            panels=panels,
            tags=["agent-observability", "cost"],
        )

    def generate_drift_dashboard(self) -> dict[str, object]:
        """Generate the behavioral drift monitoring dashboard.

        Panels
        ------
        * Current max Z-score per agent
        * Drift events per hour
        * Z-score trend by metric
        * Drift severity distribution
        """
        counter = _panel_id_counter()

        panels: list[dict[str, object]] = [
            # Stat — drift events
            _stat_panel(
                panel_id=counter.next(),
                title="Drift Events (last hour)",
                targets=[_prom_target(
                    'sum(increase(agent_drift_events_total[1h]))',
                    "events",
                )],
                grid_pos={"x": 0, "y": 0, "w": 8, "h": 4},
                unit="short",
            ),
            # Stat — max z-score
            _stat_panel(
                panel_id=counter.next(),
                title="Max Z-Score (current)",
                targets=[_prom_target(
                    'max(agent_drift_z_score)',
                    "max z",
                )],
                grid_pos={"x": 8, "y": 0, "w": 8, "h": 4},
                unit="short",
            ),
            # Stat — drifted agents
            _stat_panel(
                panel_id=counter.next(),
                title="Agents with Active Drift",
                targets=[_prom_target(
                    'count(agent_drift_z_score > 3)',
                    "agents",
                )],
                grid_pos={"x": 16, "y": 0, "w": 8, "h": 4},
                unit="short",
            ),
            # Timeseries — Z-score per metric over time
            _timeseries_panel(
                panel_id=counter.next(),
                title="Drift Z-Score by Metric",
                description="Z-score trend for each monitored behavioral metric",
                targets=[_prom_target(
                    'agent_drift_z_score',
                    "{{agent_id}} / {{metric_name}}",
                )],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
                unit="short",
            ),
            # Timeseries — drift events rate
            _timeseries_panel(
                panel_id=counter.next(),
                title="Drift Events Rate",
                description="Rate of drift events per hour",
                targets=[_prom_target(
                    'sum(rate(agent_drift_events_total[5m])) by (severity)',
                    "{{severity}}",
                )],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
                unit="short",
            ),
            # Timeseries — latency drift
            _timeseries_panel(
                panel_id=counter.next(),
                title="LLM Latency vs Baseline",
                description="How much current LLM latency deviates from the baseline average",
                targets=[
                    {**_prom_target(
                        'histogram_quantile(0.95, sum(rate(agent_llm_latency_seconds_bucket[5m])) by (le))',
                        "current p95",
                    ), "refId": "A"},
                ],
                grid_pos={"x": 0, "y": 12, "w": 24, "h": 6},
                unit="s",
            ),
        ]

        return _dashboard_skeleton(
            title="Agent Observability — Behavioral Drift",
            description="Monitor for deviations from baseline agent behavior",
            panels=panels,
            tags=["agent-observability", "drift"],
        )

    def save(
        self,
        dashboard: dict[str, object],
        output_path: Path,
    ) -> Path:
        """Serialise *dashboard* to JSON and write it to *output_path*.

        Parameters
        ----------
        dashboard:
            A dashboard dict as returned by one of the ``generate_*`` methods.
        output_path:
            Destination file path.  Parent directories are created if absent.

        Returns
        -------
        The resolved absolute path that was written.
        """
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(dashboard, fh, indent=2, ensure_ascii=False)

        return output_path
