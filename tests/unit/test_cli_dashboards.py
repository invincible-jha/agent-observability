"""Tests for the CLI dashboards command."""
from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from agent_observability.cli.main import cli


class TestDashboardsCommand:
    """Test the `agent-observability dashboards` CLI command."""

    def test_exports_all_dashboards_by_default(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboards", "--output", str(tmp_path)])
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        expected_files = [
            "agent-waterfall-dashboard.json",
            "agent-cost-dashboard.json",
            "agent-drift-dashboard.json",
            "agent-security-dashboard.json",
        ]
        for filename in expected_files:
            assert (tmp_path / filename).exists(), (
                f"Expected {filename} to be created. Output:\n{result.output}"
            )

    def test_exports_single_waterfall_dashboard(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "--type", "waterfall", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (tmp_path / "agent-waterfall-dashboard.json").exists()
        # Other dashboards should NOT be created
        assert not (tmp_path / "agent-cost-dashboard.json").exists()

    def test_exports_single_cost_dashboard(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "--type", "cost", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "agent-cost-dashboard.json").exists()

    def test_exports_single_drift_dashboard(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "--type", "drift", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "agent-drift-dashboard.json").exists()

    def test_exports_single_security_dashboard(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "--type", "security", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "agent-security-dashboard.json").exists()

    def test_output_files_contain_valid_json(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        runner.invoke(cli, ["dashboards", "--output", str(tmp_path)])
        for json_file in tmp_path.glob("*.json"):
            content = json_file.read_text(encoding="utf-8")
            loaded = json.loads(content)
            assert "panels" in loaded, f"{json_file.name} missing 'panels' key"
            assert "title" in loaded, f"{json_file.name} missing 'title' key"

    def test_output_is_echoed_to_stdout(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboards", "--output", str(tmp_path)])
        assert "Exported:" in result.output

    def test_short_output_flag_works(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "-o", str(tmp_path), "--type", "cost"],
        )
        assert result.exit_code == 0
        assert (tmp_path / "agent-cost-dashboard.json").exists()

    def test_creates_output_directory_if_missing(self, tmp_path: pathlib.Path) -> None:
        nested = tmp_path / "new" / "nested" / "dir"
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboards", "--output", str(nested)])
        assert result.exit_code == 0
        assert nested.exists()
        assert any(nested.glob("*.json"))

    def test_all_type_choice_exports_four_files(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "--type", "all", "--output", str(tmp_path)],
        )
        assert result.exit_code == 0
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 4, f"Expected 4 JSON files, got {len(json_files)}"

    def test_invalid_type_rejected(self, tmp_path: pathlib.Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboards", "--type", "invalid_type", "--output", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_help_text_is_present(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboards", "--help"])
        assert result.exit_code == 0
        assert "Export Grafana" in result.output

    def test_command_is_registered_on_cli(self) -> None:
        """dashboards should appear in the top-level CLI help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "dashboards" in result.output
