"""Microsoft Agents adapter for agent_observability."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MicrosoftAgentTracer:
    """OTel tracing adapter for Microsoft Agents.

    Maps Microsoft Bot Framework / Agents SDK turn and activity lifecycle
    events to OpenTelemetry span data dictionaries.

    Usage::

        from agent_observability.adapters.microsoft_agents import MicrosoftAgentTracer
        adapter = MicrosoftAgentTracer()
    """

    def __init__(self, tracer_provider: Any = None) -> None:
        self.tracer_provider = tracer_provider
        logger.info("MicrosoftAgentTracer initialized.")

    def on_turn_start(self, turn_id: str) -> dict[str, Any]:
        """Record the start of a conversation turn.

        Returns span data with turn identification.
        """
        return {
            "name": "microsoft_agents.turn.start",
            "attributes": {
                "turn.id": turn_id,
                "framework": "microsoft_agents",
            },
        }

    def on_turn_end(self, turn_id: str) -> dict[str, Any]:
        """Record the end of a conversation turn.

        Returns span data marking turn completion.
        """
        return {
            "name": "microsoft_agents.turn.end",
            "attributes": {
                "turn.id": turn_id,
                "framework": "microsoft_agents",
            },
        }

    def on_activity(self, activity_type: str, data: Any) -> dict[str, Any]:
        """Record a Bot Framework activity event.

        Returns span data with activity type and data type.
        """
        return {
            "name": "microsoft_agents.activity",
            "attributes": {
                "activity.type": activity_type,
                "activity.data_type": type(data).__name__,
                "framework": "microsoft_agents",
            },
        }

    def on_dialog_step(self, dialog_id: str, step: str) -> dict[str, Any]:
        """Record a dialog waterfall step.

        Returns span data with dialog identity and step name.
        """
        return {
            "name": "microsoft_agents.dialog.step",
            "attributes": {
                "dialog.id": dialog_id,
                "dialog.step": step,
                "framework": "microsoft_agents",
            },
        }
