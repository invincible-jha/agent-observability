"""OTel GenAI convention-aligned Grafana dashboard generators.

Provides four dashboard generators whose PromQL expressions and panel
descriptions reference the official OTel GenAI semantic attributes
(``gen_ai.*``) and AumOS extension attributes (``aumos.*``).

Functions
---------
generate_agent_waterfall_dashboard
    Trace waterfall view using OTel GenAI span conventions.
generate_cost_attribution_dashboard
    Cost treemap keyed on ``aumos.cost.usd``.
generate_drift_timeline_dashboard
    Behavioral drift using ``aumos.drift.score``.
generate_security_heatmap_dashboard
    Security events heatmap.
export_dashboard_json
    Helper to serialise a dashboard dict to a JSON file.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from agent_observability.conventions.semantic import AumOSAttributes, GenAIAttributes

# ── Internal helpers ──────────────────────────────────────────────────────────


def _uid_from_title(title: str) -> str:
    """Derive a stable Grafana dashboard UID from its title."""
    return hashlib.sha1(title.encode()).hexdigest()[:9]


class _PanelIdCounter:
    """Simple auto-incrementing panel ID counter."""

    def __init__(self) -> None:
        self._value: int = 0

    def next(self) -> int:
        self._value += 1
        return self._value


def _prom_target(
    expr: str,
    legend_format: str = "",
    ref_id: str = "A",
) -> dict[str, object]:
    """Produce a Prometheus datasource target dict."""
    return {
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "expr": expr,
        "legendFormat": legend_format or expr,
        "refId": ref_id,
    }


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
    description: str = "",
) -> dict[str, object]:
    """Produce a Grafana stat panel dict."""
    return {
        "id": panel_id,
        "title": title,
        "description": description,
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


def _heatmap_panel(
    panel_id: int,
    title: str,
    targets: list[dict[str, object]],
    grid_pos: dict[str, int],
    description: str = "",
) -> dict[str, object]:
    """Produce a Grafana heatmap panel dict."""
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": "heatmap",
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "fieldConfig": {
            "defaults": {},
            "overrides": [],
        },
        "options": {
            "calculate": False,
            "cellGap": 2,
            "color": {
                "exponent": 0.5,
                "fill": "dark-orange",
                "mode": "scheme",
                "reverse": False,
                "scale": "exponential",
                "scheme": "Oranges",
                "steps": 64,
            },
            "tooltip": {"show": True, "yHistogram": False},
            "yAxis": {"axisPlacement": "left", "unit": "short"},
        },
        "targets": targets,
    }


def _base_dashboard(
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
        "tags": tags or ["agent-observability", "otel-genai"],
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


# ── PromQL expression constants ───────────────────────────────────────────────

_CHAT_RATE = "sum(rate(otel_gen_ai_chat_total[5m]))"
_TOOL_RATE = "sum(rate(otel_gen_ai_execute_tool_total[5m]))"
_AGENT_RATE = "sum(rate(otel_invoke_agent_total[5m]))"
_ALL_SPANS_RATE = "sum(rate(otel_gen_ai_spans_total[5m]))"

_INPUT_TOKENS_BY_MODEL = (
    "sum(rate(otel_gen_ai_usage_input_tokens_total[5m])) by (gen_ai_request_model)"
)
_OUTPUT_TOKENS_BY_MODEL = (
    "sum(rate(otel_gen_ai_usage_output_tokens_total[5m])) by (gen_ai_response_model)"
)

_DURATION_P50 = (
    "histogram_quantile(0.50, "
    "sum(rate(otel_gen_ai_duration_seconds_bucket[5m])) by (le, span_name))"
)
_DURATION_P95 = (
    "histogram_quantile(0.95, "
    "sum(rate(otel_gen_ai_duration_seconds_bucket[5m])) by (le, span_name))"
)
_DURATION_P99 = (
    "histogram_quantile(0.99, "
    "sum(rate(otel_gen_ai_duration_seconds_bucket[5m])) by (le, span_name))"
)

_CHAT_LATENCY_P95 = (
    "histogram_quantile(0.95, "
    'sum(rate(otel_gen_ai_duration_seconds_bucket{span_name="gen_ai.chat"}[5m])) by (le))'
)

_APPROVAL_LATENCY_P50 = (
    "histogram_quantile(0.50, "
    "sum(rate(otel_gen_ai_duration_seconds_bucket"
    '{span_name="gen_ai.agent.human_approval"}[5m])) by (le))'
)
_APPROVAL_LATENCY_P95 = (
    "histogram_quantile(0.95, "
    "sum(rate(otel_gen_ai_duration_seconds_bucket"
    '{span_name="gen_ai.agent.human_approval"}[5m])) by (le))'
)

_COST_PER_1K_TOKENS = (
    "sum(rate(aumos_cost_usd_total[5m])) by (gen_ai_request_model) / "
    "(sum(rate(otel_gen_ai_usage_input_tokens_total[5m])) by (gen_ai_request_model) + "
    "sum(rate(otel_gen_ai_usage_output_tokens_total[5m])) by (gen_ai_request_model))"
    " * 1000"
)


# ── Public dashboard generators ───────────────────────────────────────────────


def generate_agent_waterfall_dashboard() -> dict[str, object]:
    """Generate a trace waterfall dashboard using OTel GenAI span conventions.

    Panels reference ``gen_ai.*`` span names and attributes as defined in
    :class:`~agent_observability.conventions.semantic.GenAIAttributes`.

    Returns
    -------
    dict[str, object]
        Valid Grafana dashboard JSON dict ready for import or provisioning.
    """
    counter = _PanelIdCounter()

    # Attribute references used in panel descriptions for traceability
    model_attr = GenAIAttributes.REQUEST_MODEL
    input_tokens_attr = GenAIAttributes.USAGE_INPUT_TOKENS
    output_tokens_attr = GenAIAttributes.USAGE_OUTPUT_TOKENS
    agent_name_attr = GenAIAttributes.AGENT_NAME

    panels: list[dict[str, object]] = [
        # Row 1 — Span counts by OTel GenAI span type
        _stat_panel(
            panel_id=counter.next(),
            title="gen_ai.chat Spans (last 5m)",
            description=f"Count of gen_ai.chat spans. Model attributed via {model_attr}.",
            targets=[_prom_target(_CHAT_RATE, "chat spans/s")],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            unit="reqps",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="gen_ai.execute_tool Spans (last 5m)",
            description="Count of gen_ai.execute_tool spans per second.",
            targets=[_prom_target(_TOOL_RATE, "tool spans/s")],
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            unit="reqps",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="invoke_agent Spans (last 5m)",
            description=(
                f"Agent delegation spans. Agent identity via {agent_name_attr}."
            ),
            targets=[_prom_target(_AGENT_RATE, "agent spans/s")],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            unit="reqps",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Total OTel GenAI Spans (last 5m)",
            description="All gen_ai.* spans combined.",
            targets=[_prom_target(_ALL_SPANS_RATE, "all spans/s")],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
            unit="reqps",
        ),
        # Row 2 — Token throughput (gen_ai.usage.input_tokens / output_tokens)
        _timeseries_panel(
            panel_id=counter.next(),
            title="Token Throughput by Model",
            description=(
                f"Input/output token rate keyed on {input_tokens_attr} "
                f"and {output_tokens_attr}. Model label from {model_attr}."
            ),
            targets=[
                _prom_target(
                    _INPUT_TOKENS_BY_MODEL,
                    "input {{gen_ai_request_model}}",
                    ref_id="A",
                ),
                _prom_target(
                    _OUTPUT_TOKENS_BY_MODEL,
                    "output {{gen_ai_response_model}}",
                    ref_id="B",
                ),
            ],
            grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
            unit="short",
        ),
        # Row 2 — Span latency waterfall (p50/p95/p99 by span type)
        _timeseries_panel(
            panel_id=counter.next(),
            title="GenAI Span Latency (p50 / p95 / p99)",
            description="Latency distribution across all OTel GenAI span types.",
            targets=[
                _prom_target(_DURATION_P50, "p50 {{span_name}}", ref_id="A"),
                _prom_target(_DURATION_P95, "p95 {{span_name}}", ref_id="B"),
                _prom_target(_DURATION_P99, "p99 {{span_name}}", ref_id="C"),
            ],
            grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
            unit="s",
        ),
        # Row 3 — Error rate by span type
        _timeseries_panel(
            panel_id=counter.next(),
            title="GenAI Span Error Rate",
            description="Error rate broken down by OTel GenAI span name.",
            targets=[_prom_target(
                'sum(rate(otel_gen_ai_errors_total[5m])) by (span_name)',
                "{{span_name}}",
            )],
            grid_pos={"x": 0, "y": 12, "w": 24, "h": 6},
            unit="short",
        ),
    ]

    return _base_dashboard(
        title="Agent Observability — GenAI Span Waterfall (OTel v1.37)",
        description=(
            "Trace waterfall view aligned to OTel GenAI semantic conventions v1.37. "
            "Span names: gen_ai.chat, gen_ai.execute_tool, invoke_agent."
        ),
        panels=panels,
        tags=["agent-observability", "otel-genai", "waterfall"],
    )


def generate_cost_attribution_dashboard() -> dict[str, object]:
    """Generate a cost attribution dashboard keyed on ``aumos.cost.usd``.

    Uses :attr:`~agent_observability.conventions.semantic.AumOSAttributes.COST_USD`
    as the primary metric label and ``gen_ai.request.model`` for breakdown.

    Returns
    -------
    dict[str, object]
        Valid Grafana dashboard JSON dict.
    """
    counter = _PanelIdCounter()

    cost_attr = AumOSAttributes.COST_USD
    model_attr = GenAIAttributes.REQUEST_MODEL

    panels: list[dict[str, object]] = [
        # Stat row — headline cost figures
        _stat_panel(
            panel_id=counter.next(),
            title="Total Cost (USD)",
            description=f"Accumulated spend tracked via {cost_attr}.",
            targets=[_prom_target("sum(aumos_cost_usd_total)", "total USD")],
            grid_pos={"x": 0, "y": 0, "w": 8, "h": 4},
            unit="currencyUSD",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Cost Rate (last 5m, USD/min)",
            description=f"Spend rate per minute. Attribute: {cost_attr}.",
            targets=[_prom_target(
                "sum(rate(aumos_cost_usd_total[5m])) * 60",
                "USD/min",
            )],
            grid_pos={"x": 8, "y": 0, "w": 8, "h": 4},
            unit="currencyUSD",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Models Tracked",
            description=(
                f"Number of distinct models reporting cost. Label: {model_attr}."
            ),
            targets=[_prom_target(
                "count(count by (gen_ai_request_model)(aumos_cost_usd_total))",
                "model count",
            )],
            grid_pos={"x": 16, "y": 0, "w": 8, "h": 4},
            unit="short",
        ),
        # Cost breakdown timeseries
        _timeseries_panel(
            panel_id=counter.next(),
            title="Cost Rate by Model",
            description=(
                f"USD/min broken down by {model_attr}. "
                f"Cost reported via {cost_attr}."
            ),
            targets=[_prom_target(
                "sum(rate(aumos_cost_usd_total[5m])) by (gen_ai_request_model) * 60",
                "{{gen_ai_request_model}}",
            )],
            grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
            unit="currencyUSD",
        ),
        # Cumulative cost timeseries
        _timeseries_panel(
            panel_id=counter.next(),
            title="Cumulative Cost Over Time (USD)",
            description=f"Running total spend. Attribute: {cost_attr}.",
            targets=[_prom_target("sum(aumos_cost_usd_total)", "cumulative USD")],
            grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
            unit="currencyUSD",
        ),
        # Token-to-cost efficiency
        _timeseries_panel(
            panel_id=counter.next(),
            title="Cost per 1K Tokens by Model",
            description=(
                f"USD per 1,000 tokens. Correlates {cost_attr} with "
                f"{GenAIAttributes.USAGE_INPUT_TOKENS} + "
                f"{GenAIAttributes.USAGE_OUTPUT_TOKENS}."
            ),
            targets=[_prom_target(
                _COST_PER_1K_TOKENS,
                "USD/1K tokens {{gen_ai_request_model}}",
            )],
            grid_pos={"x": 0, "y": 12, "w": 24, "h": 6},
            unit="currencyUSD",
        ),
    ]

    return _base_dashboard(
        title="Agent Observability — Cost Attribution (aumos.cost.usd)",
        description=(
            f"LLM cost attribution using {cost_attr}. "
            f"Model breakdown via {model_attr}."
        ),
        panels=panels,
        tags=["agent-observability", "cost", "aumos"],
    )


def generate_drift_timeline_dashboard() -> dict[str, object]:
    """Generate a behavioral drift timeline using ``aumos.drift.score``.

    Uses :attr:`~agent_observability.conventions.semantic.AumOSAttributes.DRIFT_SCORE`
    as the primary drift signal alongside OTel GenAI span names for context.

    Returns
    -------
    dict[str, object]
        Valid Grafana dashboard JSON dict.
    """
    counter = _PanelIdCounter()

    drift_attr = AumOSAttributes.DRIFT_SCORE
    agent_attr = GenAIAttributes.AGENT_NAME

    panels: list[dict[str, object]] = [
        # Headline stats
        _stat_panel(
            panel_id=counter.next(),
            title="Max Drift Score (current)",
            description=f"Peak {drift_attr} across all agents.",
            targets=[_prom_target("max(aumos_drift_score)", "max drift")],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Agents Drifting (score > 0.7)",
            description=f"Count of agents with {drift_attr} above threshold.",
            targets=[_prom_target(
                "count(aumos_drift_score > 0.7)",
                "drifting agents",
            )],
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Drift Events (last hour)",
            description="Number of drift threshold crossings in the past hour.",
            targets=[_prom_target(
                "sum(increase(aumos_drift_events_total[1h]))",
                "events/h",
            )],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Agents Monitored",
            description=f"Total distinct agents reporting {drift_attr}.",
            targets=[_prom_target(
                "count(count by (gen_ai_agent_name)(aumos_drift_score))",
                "agents",
            )],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        # Drift score timeline per agent
        _timeseries_panel(
            panel_id=counter.next(),
            title="Drift Score Timeline by Agent",
            description=(
                f"{drift_attr} over time per agent ({agent_attr}). "
                "Score 0 = baseline; 1 = maximum deviation."
            ),
            targets=[_prom_target("aumos_drift_score", "{{gen_ai_agent_name}}")],
            grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
            unit="short",
        ),
        # Drift event rate by severity
        _timeseries_panel(
            panel_id=counter.next(),
            title="Drift Event Rate by Severity",
            description=(
                "Rate of drift threshold crossings broken down by severity label."
            ),
            targets=[_prom_target(
                "sum(rate(aumos_drift_events_total[5m])) by (severity)",
                "{{severity}}",
            )],
            grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
            unit="short",
        ),
        # Correlated LLM latency drift (gen_ai spans)
        _timeseries_panel(
            panel_id=counter.next(),
            title="LLM Latency vs Drift Score (gen_ai.chat)",
            description=(
                "Overlays gen_ai.chat p95 latency with drift score to surface "
                "latency-induced behavioral drift."
            ),
            targets=[
                _prom_target(_CHAT_LATENCY_P95, "chat p95 latency", ref_id="A"),
                _prom_target("avg(aumos_drift_score)", "avg drift score", ref_id="B"),
            ],
            grid_pos={"x": 0, "y": 12, "w": 24, "h": 6},
            unit="short",
        ),
    ]

    return _base_dashboard(
        title="Agent Observability — Behavioral Drift Timeline (aumos.drift.score)",
        description=(
            f"Behavioral drift monitoring using {drift_attr}. "
            f"Agent identity via {agent_attr}."
        ),
        panels=panels,
        tags=["agent-observability", "drift", "aumos"],
    )


def generate_security_heatmap_dashboard() -> dict[str, object]:
    """Generate a security events heatmap dashboard.

    Surfaces security-relevant signals including approval events
    (:attr:`~agent_observability.conventions.semantic.AumOSAttributes.APPROVAL_REQUIRED`),
    error categories
    (:attr:`~agent_observability.conventions.semantic.AumOSAttributes.ERROR_CATEGORY`),
    and human-in-the-loop spans (``gen_ai.agent.human_approval``).

    Returns
    -------
    dict[str, object]
        Valid Grafana dashboard JSON dict.
    """
    counter = _PanelIdCounter()

    approval_required_attr = AumOSAttributes.APPROVAL_REQUIRED
    approval_granted_attr = AumOSAttributes.APPROVAL_GRANTED
    error_category_attr = AumOSAttributes.ERROR_CATEGORY

    panels: list[dict[str, object]] = [
        # Headline security stats
        _stat_panel(
            panel_id=counter.next(),
            title="Approval Requests (last hour)",
            description=(
                f"Spans where {approval_required_attr} = true in the past hour."
            ),
            targets=[_prom_target(
                "sum(increase(aumos_approval_required_total[1h]))",
                "approvals requested",
            )],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Approvals Denied (last hour)",
            description=(
                f"Spans where {approval_granted_attr} = false in the past hour."
            ),
            targets=[_prom_target(
                "sum(increase(aumos_approval_denied_total[1h]))",
                "denied",
            )],
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Security Errors (last hour)",
            description=f"Agent errors with {error_category_attr} = 'security'.",
            targets=[_prom_target(
                'sum(increase(aumos_agent_errors_total{error_category="security"}[1h]))',
                "security errors",
            )],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            unit="short",
        ),
        _stat_panel(
            panel_id=counter.next(),
            title="Approval Grant Rate (%)",
            description=(
                f"Ratio of granted approvals. "
                f"Attributes: {approval_required_attr}, {approval_granted_attr}."
            ),
            targets=[_prom_target(
                (
                    "sum(aumos_approval_granted_total) / "
                    "sum(aumos_approval_required_total) * 100"
                ),
                "grant rate %",
            )],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
            unit="percent",
        ),
        # Security events heatmap (approval requests over time by agent)
        _heatmap_panel(
            panel_id=counter.next(),
            title="Approval Requests Heatmap (by Agent)",
            description=(
                f"Density of human approval requests per agent over time. "
                f"Attribute: {approval_required_attr}."
            ),
            targets=[_prom_target(
                "sum(rate(aumos_approval_required_total[5m])) by (gen_ai_agent_name)",
                "{{gen_ai_agent_name}}",
            )],
            grid_pos={"x": 0, "y": 4, "w": 24, "h": 8},
        ),
        # Error category breakdown
        _timeseries_panel(
            panel_id=counter.next(),
            title="Agent Errors by Category",
            description=f"Error rate broken down by {error_category_attr} label.",
            targets=[_prom_target(
                "sum(rate(aumos_agent_errors_total[5m])) by (error_category)",
                "{{error_category}}",
            )],
            grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
            unit="short",
        ),
        # Human approval span latency (gen_ai.agent.human_approval)
        _timeseries_panel(
            panel_id=counter.next(),
            title="Human Approval Span Latency (p50 / p95)",
            description=(
                "Latency of gen_ai.agent.human_approval spans — "
                "how long human reviewers take to respond."
            ),
            targets=[
                _prom_target(
                    _APPROVAL_LATENCY_P50,
                    "p50 approval latency",
                    ref_id="A",
                ),
                _prom_target(
                    _APPROVAL_LATENCY_P95,
                    "p95 approval latency",
                    ref_id="B",
                ),
            ],
            grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
            unit="s",
        ),
    ]

    return _base_dashboard(
        title="Agent Observability — Security Heatmap",
        description=(
            f"Security monitoring: approval requests ({approval_required_attr}), "
            f"denials ({approval_granted_attr}), "
            f"and error categories ({error_category_attr})."
        ),
        panels=panels,
        tags=["agent-observability", "security", "aumos"],
    )


# ── Export helper ─────────────────────────────────────────────────────────────


def export_dashboard_json(
    dashboard: dict[str, object],
    output_path: str,
) -> pathlib.Path:
    """Serialise *dashboard* to JSON and write it to *output_path*.

    Parameters
    ----------
    dashboard:
        A dashboard dict as returned by one of the ``generate_*`` functions.
    output_path:
        Destination file path as a string. Parent directories are created
        if absent.

    Returns
    -------
    pathlib.Path
        The resolved absolute path that was written.
    """
    resolved = pathlib.Path(output_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w", encoding="utf-8") as file_handle:
        json.dump(dashboard, file_handle, indent=2, ensure_ascii=False)

    return resolved
