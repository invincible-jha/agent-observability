"""CLI entry point for agent-observability.

Invoked as::

    agent-observability [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_observability.cli.main
"""
from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """OpenTelemetry-native agent tracing, cost attribution, and drift detection"""


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_observability import __version__

    console.print(f"[bold]agent-observability[/bold] v{__version__}")


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print("  (No plugins registered. Install a plugin package to see entries here.)")


@cli.command(name="dashboards")
@click.option(
    "--type",
    "dashboard_type",
    type=click.Choice(["waterfall", "cost", "drift", "security", "all"]),
    default="all",
    help="Dashboard type to export",
)
@click.option("--output", "-o", default="dashboards/", help="Output directory")
def dashboards_command(dashboard_type: str, output: str) -> None:
    """Export Grafana dashboard templates (OTel GenAI convention aligned)."""
    import pathlib

    from agent_observability.dashboards.generator import (
        export_dashboard_json,
        generate_agent_waterfall_dashboard,
        generate_cost_attribution_dashboard,
        generate_drift_timeline_dashboard,
        generate_security_heatmap_dashboard,
    )

    out_dir = pathlib.Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "waterfall": generate_agent_waterfall_dashboard,
        "cost": generate_cost_attribution_dashboard,
        "drift": generate_drift_timeline_dashboard,
        "security": generate_security_heatmap_dashboard,
    }

    if dashboard_type == "all":
        targets = generators
    else:
        targets = {dashboard_type: generators[dashboard_type]}

    for name, gen_fn in targets.items():
        path = str(out_dir / f"agent-{name}-dashboard.json")
        export_dashboard_json(gen_fn(), path)
        click.echo(f"Exported: {path}")


# ── Replay command group ───────────────────────────────────────────────────────


@cli.group(name="replay")
def replay_group() -> None:
    """Record, replay, and diff agent execution traces (JSONL format)."""


@replay_group.command(name="record")
@click.option(
    "--output",
    "-o",
    required=True,
    help="Path to output JSONL trace file.",
    metavar="FILE",
)
@click.option(
    "--session-id",
    "session_id",
    default=None,
    help="Session ID to embed in the trace header (auto-generated if omitted).",
    metavar="ID",
)
@click.argument("span_json", required=False, default=None)
def replay_record_command(output: str, session_id: str | None, span_json: str | None) -> None:
    """Record a single span (or start a named session) to a JSONL trace file.

    When SPAN_JSON is supplied it is parsed as a JSON object and appended
    as one span.  When omitted the command starts a session, writes the
    header, and immediately ends it (useful for scripting or testing).

    Examples::

        agent-obs replay record -o trace.jsonl '{"span_type":"llm_call","duration_ms":120}'

        agent-obs replay record -o trace.jsonl --session-id my-session
    """
    import json
    import pathlib
    import uuid

    from agent_observability.replay.recorder import TraceRecorder, TraceRecorderError

    resolved_session_id: str = session_id if session_id is not None else str(uuid.uuid4())
    out_path = pathlib.Path(output)

    try:
        recorder = TraceRecorder(out_path)
        recorder.start_session(resolved_session_id)

        if span_json is not None:
            try:
                span_data: dict[str, object] = json.loads(span_json)
            except json.JSONDecodeError as exc:
                click.echo(f"Error: SPAN_JSON is not valid JSON: {exc}", err=True)
                recorder.end_session()
                raise SystemExit(1) from exc

            recorder.record_span(span_data)
            click.echo(f"Recorded 1 span to {out_path} (session: {resolved_session_id})")
        else:
            click.echo(
                f"Session '{resolved_session_id}' started (no span data supplied — writing empty session)."
            )

        recorder.end_session()
        click.echo(f"Session '{resolved_session_id}' ended. Trace written to: {out_path}")
    except TraceRecorderError as exc:
        click.echo(f"Recorder error: {exc}", err=True)
        raise SystemExit(1) from exc


@replay_group.command(name="play")
@click.argument("trace_file", type=click.Path(exists=True, dir_okay=False), metavar="FILE")
@click.option(
    "--filter-type",
    "filter_type",
    default=None,
    help="Only show spans of this span_type.",
    metavar="TYPE",
)
@click.option(
    "--step",
    "step_index",
    default=None,
    type=int,
    help="Print only the span at this zero-based index.",
    metavar="N",
)
@click.option(
    "--summary",
    "show_summary",
    is_flag=True,
    default=False,
    help="Print a summary of the trace instead of individual spans.",
)
def replay_play_command(
    trace_file: str,
    filter_type: str | None,
    step_index: int | None,
    show_summary: bool,
) -> None:
    """Replay (print) spans from a JSONL trace file.

    By default every span is printed as pretty-printed JSON.  Use
    ``--filter-type`` to narrow the output, ``--step N`` to inspect a single
    span, or ``--summary`` for aggregate statistics.

    Examples::

        agent-obs replay play trace.jsonl
        agent-obs replay play trace.jsonl --filter-type llm_call
        agent-obs replay play trace.jsonl --step 0
        agent-obs replay play trace.jsonl --summary
    """
    import json
    import pathlib

    from agent_observability.replay.player import TracePlayer, TracePlayerError

    path = pathlib.Path(trace_file)

    try:
        player = TracePlayer(path)
        player.load()
    except TracePlayerError as exc:
        click.echo(f"Error loading trace: {exc}", err=True)
        raise SystemExit(1) from exc

    if show_summary:
        info = player.summary()
        click.echo(json.dumps(info, indent=2))
        return

    if step_index is not None:
        try:
            span = player.step(step_index)
        except IndexError:
            click.echo(
                f"Error: index {step_index} is out of range "
                f"(trace has {len(player.spans)} span(s)).",
                err=True,
            )
            raise SystemExit(1)
        click.echo(json.dumps(span, indent=2, default=str))
        return

    spans_to_print = (
        player.filter_by_type(filter_type) if filter_type is not None else list(player.play_all())
    )

    if not spans_to_print:
        click.echo("No spans matched the given filter.")
        return

    for span in spans_to_print:
        click.echo(json.dumps(span, indent=2, default=str))


@replay_group.command(name="diff")
@click.argument(
    "trace_a",
    type=click.Path(exists=True, dir_okay=False),
    metavar="TRACE_A",
)
@click.argument(
    "trace_b",
    type=click.Path(exists=True, dir_okay=False),
    metavar="TRACE_B",
)
@click.option(
    "--json-output",
    "json_output",
    is_flag=True,
    default=False,
    help="Emit the full diff result as JSON instead of human-readable text.",
)
def replay_diff_command(trace_a: str, trace_b: str, json_output: bool) -> None:
    """Compare two JSONL trace files and report differences.

    Prints added/removed/modified spans and timing deltas.  Exit code is
    ``0`` when the traces are identical, ``1`` when differences are found.

    Examples::

        agent-obs replay diff baseline.jsonl candidate.jsonl
        agent-obs replay diff baseline.jsonl candidate.jsonl --json-output
    """
    import json
    import pathlib

    from agent_observability.replay.diff import TraceDiff, TraceDiffError

    path_a = pathlib.Path(trace_a)
    path_b = pathlib.Path(trace_b)

    try:
        differ = TraceDiff(path_a, path_b)
        result = differ.compare()
    except TraceDiffError as exc:
        click.echo(f"Diff error: {exc}", err=True)
        raise SystemExit(2) from exc

    if json_output:
        output_data: dict[str, object] = {
            "is_identical": result.is_identical,
            "total_changes": result.total_changes,
            "added_spans": result.added_spans,
            "removed_spans": result.removed_spans,
            "modified_spans": result.modified_spans,
            "timing_deltas": {
                key: {"trace_a_ms": a, "trace_b_ms": b, "delta_ms": delta}
                for key, (a, b, delta) in result.timing_deltas.items()
            },
            "structural_changes": result.structural_changes,
        }
        click.echo(json.dumps(output_data, indent=2, default=str))
    else:
        for note in result.structural_changes:
            click.echo(note)

        if result.added_spans:
            click.echo(f"\nAdded spans ({len(result.added_spans)}):")
            for span in result.added_spans:
                label = span.get("span_type") or span.get("name") or "<unknown>"
                click.echo(f"  + {label}")

        if result.removed_spans:
            click.echo(f"\nRemoved spans ({len(result.removed_spans)}):")
            for span in result.removed_spans:
                label = span.get("span_type") or span.get("name") or "<unknown>"
                click.echo(f"  - {label}")

        if result.modified_spans:
            click.echo(f"\nModified spans ({len(result.modified_spans)}):")
            for entry in result.modified_spans:
                fields = entry.get("changed_fields", [])
                click.echo(f"  ~ {entry['span_id']} (fields: {', '.join(fields)})")  # type: ignore[arg-type]

        if result.timing_deltas:
            click.echo("\nTiming deltas:")
            for key, (dur_a, dur_b, delta) in result.timing_deltas.items():
                click.echo(f"  {key}: {dur_a:.1f}ms → {dur_b:.1f}ms ({delta:+.1f}ms)")

    raise SystemExit(0 if result.is_identical else 1)


if __name__ == "__main__":
    cli()
