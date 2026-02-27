"""LangChain adapter for agent_observability."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LangChainTracer:
    """OTel tracing adapter for LangChain.

    Wraps LangChain lifecycle callbacks and maps them to OpenTelemetry span
    data dictionaries that can be passed directly to a tracer provider.

    Usage::

        from agent_observability.adapters.langchain import LangChainTracer
        adapter = LangChainTracer()
    """

    def __init__(self, tracer_provider: Any = None) -> None:
        self.tracer_provider = tracer_provider
        logger.info("LangChainTracer initialized.")

    def on_llm_start(self, model: str, prompt: str) -> dict[str, Any]:
        """Record the start of an LLM invocation.

        Returns span data describing the model and prompt.
        """
        return {
            "name": "langchain.llm.start",
            "attributes": {
                "llm.model": model,
                "llm.prompt_length": len(prompt),
                "framework": "langchain",
            },
        }

    def on_llm_end(self, response: Any) -> dict[str, Any]:
        """Record the end of an LLM invocation.

        Returns span data with response metadata.
        """
        response_text = str(response) if response is not None else ""
        return {
            "name": "langchain.llm.end",
            "attributes": {
                "llm.response_length": len(response_text),
                "framework": "langchain",
            },
        }

    def on_tool_start(self, name: str, input: Any) -> dict[str, Any]:
        """Record the start of a tool invocation.

        Returns span data with tool name and input metadata.
        """
        return {
            "name": "langchain.tool.start",
            "attributes": {
                "tool.name": name,
                "tool.input_type": type(input).__name__,
                "framework": "langchain",
            },
        }

    def on_tool_end(self, output: Any) -> dict[str, Any]:
        """Record the end of a tool invocation.

        Returns span data with tool output metadata.
        """
        return {
            "name": "langchain.tool.end",
            "attributes": {
                "tool.output_type": type(output).__name__,
                "framework": "langchain",
            },
        }

    def on_chain_start(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Record the start of a chain run.

        Returns span data with chain name and input key count.
        """
        return {
            "name": "langchain.chain.start",
            "attributes": {
                "chain.name": name,
                "chain.input_keys": list(inputs.keys()),
                "framework": "langchain",
            },
        }

    def on_chain_end(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Record the end of a chain run.

        Returns span data with output key count.
        """
        return {
            "name": "langchain.chain.end",
            "attributes": {
                "chain.output_keys": list(outputs.keys()),
                "framework": "langchain",
            },
        }
