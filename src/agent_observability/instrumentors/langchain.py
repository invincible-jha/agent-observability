"""LangChainInstrumentor — auto-trace LangChain calls via monkey-patching.

Wraps ``BaseLLM.__call__`` and ``BaseTool.run`` with agent-semantic spans.
The LangChain import is guarded; if LangChain is not installed the
instrumentor silently becomes a no-op.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.tracer.agent_tracer import AgentTracer

logger = logging.getLogger(__name__)

try:
    import langchain  # noqa: F401

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


class LangChainInstrumentor:
    """Instrument LangChain with agent-semantic tracing.

    Parameters
    ----------
    tracer:
        The :class:`~agent_observability.tracer.AgentTracer` instance to use
        for creating spans.

    Example
    -------
    >>> instrumentor = LangChainInstrumentor(tracer)
    >>> instrumentor.instrument()
    >>> # ... your LangChain code ...
    >>> instrumentor.uninstrument()
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer
        self._original_llm_call: Optional[object] = None
        self._original_tool_run: Optional[object] = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch LangChain to add tracing.

        Patches:
        - ``langchain.llms.base.BaseLLM.__call__``
        - ``langchain.tools.base.BaseTool.run``

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._instrumented:
            return
        if not _LANGCHAIN_AVAILABLE:
            logger.debug("LangChainInstrumentor: langchain not installed, skipping")
            return

        tracer = self._tracer

        try:
            from langchain.llms.base import BaseLLM

            original_llm_call = BaseLLM.__call__

            def _traced_llm_call(
                self_llm: object,
                prompt: str,
                *args: object,
                **kwargs: object,
            ) -> object:
                model_name = getattr(self_llm, "model_name", "unknown")
                with tracer.llm_call(model=model_name, provider="langchain") as span:
                    result = original_llm_call(self_llm, prompt, *args, **kwargs)  # type: ignore[operator]
                    span.set_attribute("llm.prompt_length", len(prompt) if isinstance(prompt, str) else 0)
                    return result

            BaseLLM.__call__ = _traced_llm_call  # type: ignore[method-assign]
            self._original_llm_call = original_llm_call
            logger.debug("LangChainInstrumentor: patched BaseLLM.__call__")
        except (ImportError, AttributeError) as exc:
            logger.debug("LangChainInstrumentor: could not patch BaseLLM: %s", exc)

        try:
            from langchain.tools.base import BaseTool

            original_tool_run = BaseTool.run

            def _traced_tool_run(
                self_tool: object,
                tool_input: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                tool_name = getattr(self_tool, "name", "unknown_tool")
                with tracer.tool_invoke(tool_name=tool_name) as span:
                    result = original_tool_run(self_tool, tool_input, *args, **kwargs)  # type: ignore[operator]
                    span.set_attribute("tool.success", True)
                    return result

            BaseTool.run = _traced_tool_run  # type: ignore[method-assign]
            self._original_tool_run = original_tool_run
            logger.debug("LangChainInstrumentor: patched BaseTool.run")
        except (ImportError, AttributeError) as exc:
            logger.debug("LangChainInstrumentor: could not patch BaseTool: %s", exc)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove all patches installed by :meth:`instrument`."""
        if not self._instrumented:
            return

        try:
            from langchain.llms.base import BaseLLM

            if self._original_llm_call is not None:
                BaseLLM.__call__ = self._original_llm_call  # type: ignore[method-assign]
                self._original_llm_call = None
        except (ImportError, AttributeError):
            pass

        try:
            from langchain.tools.base import BaseTool

            if self._original_tool_run is not None:
                BaseTool.run = self._original_tool_run  # type: ignore[method-assign]
                self._original_tool_run = None
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        logger.debug("LangChainInstrumentor: uninstrumented")
