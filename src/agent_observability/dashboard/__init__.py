"""agent-observability web dashboard subpackage.

Provides a self-contained HTTP dashboard for viewing traces, spans,
costs, and latency histograms.  Requires no external dependencies
beyond the Python standard library.

Usage
-----
::

    from agent_observability.dashboard import DashboardServer
    from agent_observability.dashboard.server import DashboardDataSource

    source = DashboardDataSource()
    server = DashboardServer(data_source=source, host="127.0.0.1", port=8081)
    server.start()
"""
from __future__ import annotations

from agent_observability.dashboard.server import DashboardServer, DashboardDataSource

__all__ = [
    "DashboardServer",
    "DashboardDataSource",
]
