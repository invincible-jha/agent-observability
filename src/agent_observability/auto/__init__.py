"""Auto-instrumentation for popular agent frameworks.

Usage::

    from agent_observability.auto import instrument_langchain, instrument_crewai
    instrument_langchain()  # Automatic OTel span creation for all LangChain operations
"""
from __future__ import annotations

from agent_observability.auto.crewai_instrumentor import instrument_crewai
from agent_observability.auto.langchain_instrumentor import instrument_langchain

__all__ = ["instrument_crewai", "instrument_langchain"]
