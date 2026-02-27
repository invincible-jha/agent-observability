"""Anthropic SDK adapter for agent_observability."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicTracer:
    """OTel tracing adapter for the Anthropic SDK.

    Maps Anthropic message lifecycle events and tool use to OpenTelemetry span
    data dictionaries for recording via a tracer provider.

    Usage::

        from agent_observability.adapters.anthropic_sdk import AnthropicTracer
        adapter = AnthropicTracer()
    """

    def __init__(self, tracer_provider: Any = None) -> None:
        self.tracer_provider = tracer_provider
        logger.info("AnthropicTracer initialized.")

    def on_message_start(self, model: str, messages: list[Any]) -> dict[str, Any]:
        """Record the start of an Anthropic messages API call.

        Returns span data with model name and message count.
        """
        return {
            "name": "anthropic.message.start",
            "attributes": {
                "llm.model": model,
                "llm.message_count": len(messages),
                "framework": "anthropic",
            },
        }

    def on_message_end(self, response: Any) -> dict[str, Any]:
        """Record the end of an Anthropic messages API call.

        Returns span data with response type metadata.
        """
        return {
            "name": "anthropic.message.end",
            "attributes": {
                "llm.response_type": type(response).__name__,
                "framework": "anthropic",
            },
        }

    def on_tool_use(self, tool_name: str, input: Any) -> dict[str, Any]:
        """Record a tool_use content block from an Anthropic response.

        Returns span data with tool name and input type.
        """
        return {
            "name": "anthropic.tool.use",
            "attributes": {
                "tool.name": tool_name,
                "tool.input_type": type(input).__name__,
                "framework": "anthropic",
            },
        }

    def on_content_block(self, block_type: str, content: Any) -> dict[str, Any]:
        """Record a content block emitted during an Anthropic response.

        Returns span data with block type and content length estimate.
        """
        content_str = str(content) if content is not None else ""
        return {
            "name": "anthropic.content_block",
            "attributes": {
                "content.block_type": block_type,
                "content.length": len(content_str),
                "framework": "anthropic",
            },
        }
