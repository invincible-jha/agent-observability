"""Framework adapters for agent_observability.

Each adapter maps framework-specific lifecycle events to OpenTelemetry span
data dictionaries without requiring the framework package to be installed.
"""
from __future__ import annotations

from agent_observability.adapters.anthropic_sdk import AnthropicTracer
from agent_observability.adapters.crewai import CrewAITracer
from agent_observability.adapters.langchain import LangChainTracer
from agent_observability.adapters.microsoft_agents import MicrosoftAgentTracer
from agent_observability.adapters.openai_agents import OpenAIAgentsTracer

__all__ = [
    "AnthropicTracer",
    "CrewAITracer",
    "LangChainTracer",
    "MicrosoftAgentTracer",
    "OpenAIAgentsTracer",
]
