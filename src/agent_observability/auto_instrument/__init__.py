"""Auto-instrumentation — framework adapters and generic decorator.

All instrumentors follow the same protocol:

.. code-block:: python

    from agent_observability.auto_instrument import LangChainInstrumentor
    from agent_observability.tracer import AgentTracer

    tracer = AgentTracer(agent_id="my-agent")
    instr = LangChainInstrumentor(tracer)
    instr.instrument()
    # ... your LangChain code ...
    instr.uninstrument()
"""
from __future__ import annotations

from agent_observability.auto_instrument.anthropic_sdk import AnthropicInstrumentor
from agent_observability.auto_instrument.autogen import AutoGenInstrumentor
from agent_observability.auto_instrument.crewai import CrewAIInstrumentor
from agent_observability.auto_instrument.generic import GenericInstrumentor
from agent_observability.auto_instrument.langchain import LangChainInstrumentor
from agent_observability.auto_instrument.openai_sdk import OpenAIInstrumentor

__all__ = [
    "LangChainInstrumentor",
    "CrewAIInstrumentor",
    "AutoGenInstrumentor",
    "AnthropicInstrumentor",
    "OpenAIInstrumentor",
    "GenericInstrumentor",
]
