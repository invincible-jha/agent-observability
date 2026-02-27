"""CrewAI adapter for agent_observability."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrewAITracer:
    """OTel tracing adapter for CrewAI.

    Maps CrewAI task and crew lifecycle events to OpenTelemetry span data
    dictionaries suitable for recording in a tracer provider.

    Usage::

        from agent_observability.adapters.crewai import CrewAITracer
        adapter = CrewAITracer()
    """

    def __init__(self, tracer_provider: Any = None) -> None:
        self.tracer_provider = tracer_provider
        logger.info("CrewAITracer initialized.")

    def on_task_start(self, task_name: str) -> dict[str, Any]:
        """Record the start of a CrewAI task.

        Returns span data with task identification.
        """
        return {
            "name": "crewai.task.start",
            "attributes": {
                "task.name": task_name,
                "framework": "crewai",
            },
        }

    def on_task_end(self, task_name: str, result: Any) -> dict[str, Any]:
        """Record the end of a CrewAI task.

        Returns span data with task result metadata.
        """
        return {
            "name": "crewai.task.end",
            "attributes": {
                "task.name": task_name,
                "task.result_type": type(result).__name__,
                "framework": "crewai",
            },
        }

    def on_agent_action(self, agent_name: str, action: str) -> dict[str, Any]:
        """Record an action taken by a CrewAI agent.

        Returns span data capturing agent identity and action taken.
        """
        return {
            "name": "crewai.agent.action",
            "attributes": {
                "agent.name": agent_name,
                "agent.action": action,
                "framework": "crewai",
            },
        }

    def on_crew_start(self, crew_name: str) -> dict[str, Any]:
        """Record the start of a CrewAI crew run.

        Returns span data with crew identification.
        """
        return {
            "name": "crewai.crew.start",
            "attributes": {
                "crew.name": crew_name,
                "framework": "crewai",
            },
        }

    def on_crew_end(self, crew_name: str) -> dict[str, Any]:
        """Record the end of a CrewAI crew run.

        Returns span data marking crew completion.
        """
        return {
            "name": "crewai.crew.end",
            "attributes": {
                "crew.name": crew_name,
                "framework": "crewai",
            },
        }
