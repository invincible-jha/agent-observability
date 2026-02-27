"""HTTP server mode for agent-observability.

Provides a lightweight stdlib-based HTTP API for trace collection and
cost reporting without requiring any additional web framework dependencies.
"""
from __future__ import annotations

from agent_observability.server.app import AgentObservabilityHandler, create_server, run_server

__all__ = ["AgentObservabilityHandler", "create_server", "run_server"]
