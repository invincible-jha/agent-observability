"""AnthropicInstrumentor — auto-trace the Anthropic Python SDK via monkey-patching.

Wraps ``anthropic.Anthropic.messages.create`` (and the async variant) so that
every Claude API call is captured as an ``llm_call`` span.  The Anthropic
import is guarded; if the SDK is not installed the instrumentor is a no-op.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.tracer.agent_tracer import AgentTracer

logger = logging.getLogger(__name__)

try:
    import anthropic  # noqa: F401

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


class AnthropicInstrumentor:
    """Instrument the Anthropic SDK with agent-semantic tracing.

    Parameters
    ----------
    tracer:
        The :class:`~agent_observability.tracer.AgentTracer` instance to use.
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer
        self._original_messages_create: Optional[object] = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch the Anthropic SDK to add tracing.

        Patches ``anthropic.resources.messages.Messages.create``.
        Safe to call multiple times.
        """
        if self._instrumented:
            return
        if not _ANTHROPIC_AVAILABLE:
            logger.debug("AnthropicInstrumentor: anthropic not installed, skipping")
            return

        tracer = self._tracer

        try:
            from anthropic.resources.messages import Messages

            original_create = Messages.create

            def _traced_create(
                self_messages: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                model = str(kwargs.get("model", "claude-unknown"))
                max_tokens = int(kwargs.get("max_tokens", 0))
                messages_list = kwargs.get("messages", [])
                prompt_token_estimate = sum(
                    len(str(m.get("content", ""))) // 4
                    for m in messages_list
                    if isinstance(m, dict)
                )
                with tracer.llm_call(
                    model=model,
                    provider="anthropic",
                    prompt_tokens=prompt_token_estimate,
                ) as span:
                    response = original_create(self_messages, *args, **kwargs)  # type: ignore[operator]
                    # Extract usage from response if available
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        input_tok = getattr(usage, "input_tokens", 0)
                        output_tok = getattr(usage, "output_tokens", 0)
                        span.set_tokens(input_tok, output_tok)
                    if max_tokens:
                        span.set_attribute("llm.max_tokens", max_tokens)
                    return response

            Messages.create = _traced_create  # type: ignore[method-assign]
            self._original_messages_create = original_create
            logger.debug("AnthropicInstrumentor: patched Messages.create")
        except (ImportError, AttributeError) as exc:
            logger.debug("AnthropicInstrumentor: could not patch Messages.create: %s", exc)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove all patches installed by :meth:`instrument`."""
        if not self._instrumented:
            return

        try:
            from anthropic.resources.messages import Messages

            if self._original_messages_create is not None:
                Messages.create = self._original_messages_create  # type: ignore[method-assign]
                self._original_messages_create = None
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        logger.debug("AnthropicInstrumentor: uninstrumented")
