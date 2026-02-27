"""OpenAI Agents SDK adapter for agent_observability."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OpenAIAgentsTracer:
    """OTel tracing adapter for the OpenAI Agents SDK.

    Maps OpenAI agent lifecycle events—start, end, tool calls, and handoffs—to
    OpenTelemetry span data dictionaries ready for recording.

    Usage::

        from agent_observability.adapters.openai_agents import OpenAIAgentsTracer
        adapter = OpenAIAgentsTracer()
    """

    def __init__(self, tracer_provider: Any = None) -> None:
        self.tracer_provider = tracer_provider
        logger.info("OpenAIAgentsTracer initialized.")

    def on_agent_start(self, agent_name: str, task: str) -> dict[str, Any]:
        """Record the start of an OpenAI agent run.

        Returns span data with agent name and task description length.
        """
        return {
            "name": "openai_agents.agent.start",
            "attributes": {
                "agent.name": agent_name,
                "agent.task_length": len(task),
                "framework": "openai_agents",
            },
        }

    def on_agent_end(self, agent_name: str, result: Any) -> dict[str, Any]:
        """Record the end of an OpenAI agent run.

        Returns span data with agent name and result type.
        """
        return {
            "name": "openai_agents.agent.end",
            "attributes": {
                "agent.name": agent_name,
                "agent.result_type": type(result).__name__,
                "framework": "openai_agents",
            },
        }

    def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Record a tool call made by an OpenAI agent.

        Returns span data with tool name and argument key count.
        """
        return {
            "name": "openai_agents.tool.call",
            "attributes": {
                "tool.name": tool_name,
                "tool.arg_count": len(args),
                "framework": "openai_agents",
            },
        }

    def on_handoff(self, from_agent: str, to_agent: str) -> dict[str, Any]:
        """Record a handoff between OpenAI agents.

        Returns span data capturing source and destination agent names.
        """
        return {
            "name": "openai_agents.handoff",
            "attributes": {
                "handoff.from_agent": from_agent,
                "handoff.to_agent": to_agent,
                "framework": "openai_agents",
            },
        }
