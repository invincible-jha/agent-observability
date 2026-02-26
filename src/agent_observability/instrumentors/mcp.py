"""MCPInstrumentor — auto-trace MCP (Model Context Protocol) tool calls.

Wraps MCP client tool execution methods with agent-semantic ``tool_invoke``
spans.  The MCP import is guarded; if the MCP SDK is not installed the
instrumentor is a no-op.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.tracer.agent_tracer import AgentTracer

logger = logging.getLogger(__name__)

try:
    import mcp  # noqa: F401

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


class MCPInstrumentor:
    """Instrument MCP tool calls with agent-semantic tracing.

    Parameters
    ----------
    tracer:
        The :class:`~agent_observability.tracer.AgentTracer` instance to use.
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer
        self._original_call_tool: Optional[object] = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch the MCP SDK to add tracing.

        Patches ``mcp.client.session.ClientSession.call_tool``.
        Safe to call multiple times.
        """
        if self._instrumented:
            return
        if not _MCP_AVAILABLE:
            logger.debug("MCPInstrumentor: mcp not installed, skipping")
            return

        tracer = self._tracer

        try:
            from mcp.client.session import ClientSession

            original_call_tool = ClientSession.call_tool

            async def _traced_call_tool(
                self_session: object,
                name: str,
                arguments: object = None,
                *args: object,
                **kwargs: object,
            ) -> object:
                with tracer.tool_invoke(tool_name=name) as span:
                    span.set_attribute("mcp.tool_name", name)
                    result = await original_call_tool(  # type: ignore[operator,misc]
                        self_session, name, arguments, *args, **kwargs
                    )
                    span.set_attribute("tool.success", True)
                    return result

            ClientSession.call_tool = _traced_call_tool  # type: ignore[method-assign]
            self._original_call_tool = original_call_tool
            logger.debug("MCPInstrumentor: patched ClientSession.call_tool")
        except (ImportError, AttributeError) as exc:
            logger.debug("MCPInstrumentor: could not patch ClientSession.call_tool: %s", exc)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove all patches installed by :meth:`instrument`."""
        if not self._instrumented:
            return

        try:
            from mcp.client.session import ClientSession

            if self._original_call_tool is not None:
                ClientSession.call_tool = self._original_call_tool  # type: ignore[method-assign]
                self._original_call_tool = None
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        logger.debug("MCPInstrumentor: uninstrumented")
