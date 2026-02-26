"""OpenAIInstrumentor — auto-trace the OpenAI Python SDK via monkey-patching.

Wraps ``openai.resources.chat.completions.Completions.create`` so that every
ChatCompletion call is captured as an ``llm_call`` span.  The OpenAI import
is guarded; if the SDK is not installed the instrumentor is a no-op.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.tracer.agent_tracer import AgentTracer

logger = logging.getLogger(__name__)

try:
    import openai  # noqa: F401

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenAIInstrumentor:
    """Instrument the OpenAI SDK with agent-semantic tracing.

    Parameters
    ----------
    tracer:
        The :class:`~agent_observability.tracer.AgentTracer` instance to use.
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer
        self._original_chat_create: Optional[object] = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch the OpenAI SDK to add tracing.

        Patches ``openai.resources.chat.completions.Completions.create``.
        Safe to call multiple times.
        """
        if self._instrumented:
            return
        if not _OPENAI_AVAILABLE:
            logger.debug("OpenAIInstrumentor: openai not installed, skipping")
            return

        tracer = self._tracer

        try:
            from openai.resources.chat.completions import Completions

            original_create = Completions.create

            def _traced_create(
                self_completions: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                model = str(kwargs.get("model", "gpt-unknown"))
                messages_list = kwargs.get("messages", [])
                prompt_token_estimate = sum(
                    len(str(m.get("content", ""))) // 4
                    for m in messages_list
                    if isinstance(m, dict)
                )

                with tracer.llm_call(
                    model=model,
                    provider="openai",
                    prompt_tokens=prompt_token_estimate,
                ) as span:
                    response = original_create(self_completions, *args, **kwargs)  # type: ignore[operator]
                    # Extract usage from response if available
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        input_tok = getattr(usage, "prompt_tokens", 0)
                        output_tok = getattr(usage, "completion_tokens", 0)
                        span.set_tokens(input_tok, output_tok)
                    return response

            Completions.create = _traced_create  # type: ignore[method-assign]
            self._original_chat_create = original_create
            logger.debug("OpenAIInstrumentor: patched Completions.create")
        except (ImportError, AttributeError) as exc:
            logger.debug("OpenAIInstrumentor: could not patch Completions.create: %s", exc)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove all patches installed by :meth:`instrument`."""
        if not self._instrumented:
            return

        try:
            from openai.resources.chat.completions import Completions

            if self._original_chat_create is not None:
                Completions.create = self._original_chat_create  # type: ignore[method-assign]
                self._original_chat_create = None
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        logger.debug("OpenAIInstrumentor: uninstrumented")
