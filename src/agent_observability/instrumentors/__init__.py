"""Framework instrumentors — monkey-patch popular agent frameworks with tracing."""
from __future__ import annotations

from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor
from agent_observability.instrumentors.autogen import AutoGenInstrumentor
from agent_observability.instrumentors.crewai import CrewAIInstrumentor
from agent_observability.instrumentors.langchain import LangChainInstrumentor
from agent_observability.instrumentors.mcp import MCPInstrumentor
from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

__all__ = [
    "LangChainInstrumentor",
    "CrewAIInstrumentor",
    "AutoGenInstrumentor",
    "AnthropicInstrumentor",
    "OpenAIInstrumentor",
    "MCPInstrumentor",
]
