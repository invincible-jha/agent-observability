"""Tests for OTel GenAI convention-aligned Grafana dashboard generators."""
from __future__ import annotations

import json
import pathlib

import pytest

from agent_observability.conventions.semantic import AumOSAttributes, GenAIAttributes
from agent_observability.dashboards.generator import (
    export_dashboard_json,
    generate_agent_waterfall_dashboard,
    generate_cost_attribution_dashboard,
    generate_drift_timeline_dashboard,
    generate_security_heatmap_dashboard,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _assert_valid_grafana_schema(dashboard: dict[str, object]) -> None:
    """Assert minimal Grafana dashboard schema fields are present."""
    required_keys = {"id", "uid", "title", "panels", "schemaVersion", "templating"}
    missing = required_keys - set(dashboard.keys())
    assert not missing, f"Dashboard missing required keys: {missing}"
    assert isinstance(dashboard["panels"], list)
    assert isinstance(dashboard["uid"], str) and len(dashboard["uid"]) > 0
    assert isinstance(dashboard["title"], str) and len(dashboard["title"]) > 0


def _assert_panels_have_required_fields(panels: list[dict[str, object]]) -> None:
    """Assert each panel has id, title, type, gridPos, and targets."""
    for panel in panels:
        assert "id" in panel, f"Panel missing 'id': {panel.get('title')}"
        assert "title" in panel, f"Panel {panel.get('id')} missing 'title'"
        assert "type" in panel, f"Panel {panel.get('id')} missing 'type'"
        assert "gridPos" in panel, f"Panel {panel.get('id')} missing 'gridPos'"
        assert "targets" in panel, f"Panel {panel.get('id')} missing 'targets'"


def _assert_no_duplicate_panel_ids(panels: list[dict[str, object]]) -> None:
    """Assert no two panels share an ID."""
    ids = [p["id"] for p in panels]
    assert len(ids) == len(set(ids)), f"Duplicate panel IDs found: {ids}"


# ── Waterfall dashboard tests ─────────────────────────────────────────────────


class TestGenerateAgentWaterfallDashboard:
    def test_returns_dict(self) -> None:
        result = generate_agent_waterfall_dashboard()
        assert isinstance(result, dict)

    def test_valid_grafana_schema(self) -> None:
        _assert_valid_grafana_schema(generate_agent_waterfall_dashboard())

    def test_title_contains_otel_reference(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        assert "OTel" in str(dashboard["title"]) or "GenAI" in str(dashboard["title"])

    def test_has_minimum_panel_count(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        assert len(panels) >= 4, f"Expected at least 4 panels, got {len(panels)}"

    def test_panels_have_required_fields(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_panels_have_required_fields(panels)

    def test_no_duplicate_panel_ids(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_no_duplicate_panel_ids(panels)

    def test_references_gen_ai_request_model_attribute(self) -> None:
        """Panel descriptions or targets should reference gen_ai.request.model."""
        dashboard = generate_agent_waterfall_dashboard()
        dashboard_str = json.dumps(dashboard)
        assert GenAIAttributes.REQUEST_MODEL in dashboard_str

    def test_references_usage_tokens_attributes(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        dashboard_str = json.dumps(dashboard)
        assert GenAIAttributes.USAGE_INPUT_TOKENS in dashboard_str
        assert GenAIAttributes.USAGE_OUTPUT_TOKENS in dashboard_str

    def test_references_gen_ai_chat_span(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        dashboard_str = json.dumps(dashboard)
        assert "gen_ai.chat" in dashboard_str

    def test_tags_include_otel_genai(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        tags = dashboard.get("tags", [])
        assert isinstance(tags, list)
        assert "otel-genai" in tags

    def test_uid_is_deterministic(self) -> None:
        """Calling the function twice should produce the same UID."""
        uid1 = generate_agent_waterfall_dashboard()["uid"]
        uid2 = generate_agent_waterfall_dashboard()["uid"]
        assert uid1 == uid2

    def test_schema_version_is_37(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        assert dashboard["schemaVersion"] == 37

    def test_datasource_template_variable_present(self) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        template_list = dashboard["templating"]["list"]  # type: ignore[index]
        assert isinstance(template_list, list)
        datasource_vars = [t for t in template_list if t.get("name") == "datasource"]
        assert len(datasource_vars) == 1


# ── Cost attribution dashboard tests ─────────────────────────────────────────


class TestGenerateCostAttributionDashboard:
    def test_returns_dict(self) -> None:
        assert isinstance(generate_cost_attribution_dashboard(), dict)

    def test_valid_grafana_schema(self) -> None:
        _assert_valid_grafana_schema(generate_cost_attribution_dashboard())

    def test_has_minimum_panel_count(self) -> None:
        dashboard = generate_cost_attribution_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        assert len(panels) >= 4

    def test_panels_have_required_fields(self) -> None:
        dashboard = generate_cost_attribution_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_panels_have_required_fields(panels)

    def test_no_duplicate_panel_ids(self) -> None:
        dashboard = generate_cost_attribution_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_no_duplicate_panel_ids(panels)

    def test_references_aumos_cost_usd(self) -> None:
        dashboard_str = json.dumps(generate_cost_attribution_dashboard())
        assert AumOSAttributes.COST_USD in dashboard_str

    def test_references_gen_ai_request_model(self) -> None:
        dashboard_str = json.dumps(generate_cost_attribution_dashboard())
        assert GenAIAttributes.REQUEST_MODEL in dashboard_str

    def test_has_currency_unit_panel(self) -> None:
        """At least one panel should use currencyUSD unit."""
        dashboard = generate_cost_attribution_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        currency_panels = [
            p for p in panels
            if p.get("fieldConfig", {}).get("defaults", {}).get("unit") == "currencyUSD"
        ]
        assert len(currency_panels) >= 1

    def test_tags_include_cost(self) -> None:
        dashboard = generate_cost_attribution_dashboard()
        assert "cost" in dashboard.get("tags", [])

    def test_tags_include_aumos(self) -> None:
        dashboard = generate_cost_attribution_dashboard()
        assert "aumos" in dashboard.get("tags", [])


# ── Drift timeline dashboard tests ────────────────────────────────────────────


class TestGenerateDriftTimelineDashboard:
    def test_returns_dict(self) -> None:
        assert isinstance(generate_drift_timeline_dashboard(), dict)

    def test_valid_grafana_schema(self) -> None:
        _assert_valid_grafana_schema(generate_drift_timeline_dashboard())

    def test_has_minimum_panel_count(self) -> None:
        dashboard = generate_drift_timeline_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        assert len(panels) >= 4

    def test_panels_have_required_fields(self) -> None:
        dashboard = generate_drift_timeline_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_panels_have_required_fields(panels)

    def test_no_duplicate_panel_ids(self) -> None:
        dashboard = generate_drift_timeline_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_no_duplicate_panel_ids(panels)

    def test_references_aumos_drift_score(self) -> None:
        dashboard_str = json.dumps(generate_drift_timeline_dashboard())
        assert AumOSAttributes.DRIFT_SCORE in dashboard_str

    def test_references_gen_ai_agent_name(self) -> None:
        dashboard_str = json.dumps(generate_drift_timeline_dashboard())
        assert GenAIAttributes.AGENT_NAME in dashboard_str

    def test_references_gen_ai_chat_span_in_correlation(self) -> None:
        """Drift dashboard should correlate with gen_ai.chat latency."""
        dashboard_str = json.dumps(generate_drift_timeline_dashboard())
        assert "gen_ai.chat" in dashboard_str

    def test_tags_include_drift(self) -> None:
        dashboard = generate_drift_timeline_dashboard()
        assert "drift" in dashboard.get("tags", [])


# ── Security heatmap dashboard tests ─────────────────────────────────────────


class TestGenerateSecurityHeatmapDashboard:
    def test_returns_dict(self) -> None:
        assert isinstance(generate_security_heatmap_dashboard(), dict)

    def test_valid_grafana_schema(self) -> None:
        _assert_valid_grafana_schema(generate_security_heatmap_dashboard())

    def test_has_minimum_panel_count(self) -> None:
        dashboard = generate_security_heatmap_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        assert len(panels) >= 5

    def test_panels_have_required_fields(self) -> None:
        dashboard = generate_security_heatmap_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_panels_have_required_fields(panels)

    def test_no_duplicate_panel_ids(self) -> None:
        dashboard = generate_security_heatmap_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        _assert_no_duplicate_panel_ids(panels)

    def test_references_approval_required_attribute(self) -> None:
        dashboard_str = json.dumps(generate_security_heatmap_dashboard())
        assert AumOSAttributes.APPROVAL_REQUIRED in dashboard_str

    def test_references_approval_granted_attribute(self) -> None:
        dashboard_str = json.dumps(generate_security_heatmap_dashboard())
        assert AumOSAttributes.APPROVAL_GRANTED in dashboard_str

    def test_references_error_category_attribute(self) -> None:
        dashboard_str = json.dumps(generate_security_heatmap_dashboard())
        assert AumOSAttributes.ERROR_CATEGORY in dashboard_str

    def test_has_heatmap_panel(self) -> None:
        """Security dashboard should include at least one heatmap panel."""
        dashboard = generate_security_heatmap_dashboard()
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        heatmap_panels = [p for p in panels if p.get("type") == "heatmap"]
        assert len(heatmap_panels) >= 1

    def test_references_human_approval_span(self) -> None:
        dashboard_str = json.dumps(generate_security_heatmap_dashboard())
        assert "gen_ai.agent.human_approval" in dashboard_str

    def test_tags_include_security(self) -> None:
        dashboard = generate_security_heatmap_dashboard()
        assert "security" in dashboard.get("tags", [])


# ── export_dashboard_json tests ───────────────────────────────────────────────


class TestExportDashboardJson:
    def test_creates_file(self, tmp_path: pathlib.Path) -> None:
        dashboard = generate_agent_waterfall_dashboard()
        output_path = str(tmp_path / "test-dashboard.json")
        result = export_dashboard_json(dashboard, output_path)
        assert result.exists()

    def test_returns_path_object(self, tmp_path: pathlib.Path) -> None:
        dashboard = generate_cost_attribution_dashboard()
        output_path = str(tmp_path / "cost.json")
        result = export_dashboard_json(dashboard, output_path)
        assert isinstance(result, pathlib.Path)

    def test_written_json_is_valid(self, tmp_path: pathlib.Path) -> None:
        dashboard = generate_drift_timeline_dashboard()
        output_path = str(tmp_path / "drift.json")
        export_dashboard_json(dashboard, output_path)
        with open(output_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["title"] == dashboard["title"]

    def test_creates_parent_directories(self, tmp_path: pathlib.Path) -> None:
        """export_dashboard_json should create missing parent directories."""
        deep_path = str(tmp_path / "nested" / "dir" / "dashboard.json")
        export_dashboard_json(generate_security_heatmap_dashboard(), deep_path)
        assert pathlib.Path(deep_path).exists()

    def test_written_json_is_pretty_printed(self, tmp_path: pathlib.Path) -> None:
        """Output JSON should be indented (pretty-printed)."""
        output_path = str(tmp_path / "pretty.json")
        export_dashboard_json(generate_agent_waterfall_dashboard(), output_path)
        content = pathlib.Path(output_path).read_text(encoding="utf-8")
        # Indented JSON has newlines; compact JSON does not
        assert "\n" in content

    def test_output_path_is_absolute(self, tmp_path: pathlib.Path) -> None:
        output_path = str(tmp_path / "abs.json")
        result = export_dashboard_json(generate_cost_attribution_dashboard(), output_path)
        assert result.is_absolute()


# ── Dashboards __init__ re-export tests ──────────────────────────────────────


class TestDashboardsPackageExports:
    """Verify that dashboards/__init__.py exports everything."""

    def test_generator_functions_importable_from_package(self) -> None:
        from agent_observability.dashboards import (  # noqa: F401
            export_dashboard_json,
            generate_agent_waterfall_dashboard,
            generate_cost_attribution_dashboard,
            generate_drift_timeline_dashboard,
            generate_security_heatmap_dashboard,
        )

    def test_legacy_generator_still_importable(self) -> None:
        from agent_observability.dashboards import GrafanaDashboardGenerator  # noqa: F401

        assert callable(GrafanaDashboardGenerator)

    def test_all_exports_in_dunder_all(self) -> None:
        import agent_observability.dashboards as dashboards_pkg

        expected = {
            "GrafanaDashboardGenerator",
            "generate_agent_waterfall_dashboard",
            "generate_cost_attribution_dashboard",
            "generate_drift_timeline_dashboard",
            "generate_security_heatmap_dashboard",
            "export_dashboard_json",
        }
        assert expected.issubset(set(dashboards_pkg.__all__))
