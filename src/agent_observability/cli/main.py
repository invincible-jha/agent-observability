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


if __name__ == "__main__":
    cli()
