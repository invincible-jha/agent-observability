"""AutoGenInstrumentor — auto-trace AutoGen agent conversations via monkey-patching.

Wraps ``ConversableAgent.initiate_chat`` and ``ConversableAgent.generate_reply``
with agent-semantic spans.  The AutoGen import is guarded so that if AutoGen
is not installed the instrumentor silently becomes a no-op.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.tracer.agent_tracer import AgentTracer

logger = logging.getLogger(__name__)

try:
    import autogen  # noqa: F401

    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


class AutoGenInstrumentor:
    """Instrument AutoGen with agent-semantic tracing.

    Parameters
    ----------
    tracer:
        The :class:`~agent_observability.tracer.AgentTracer` instance to use.
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer
        self._original_initiate_chat: Optional[object] = None
        self._original_generate_reply: Optional[object] = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch AutoGen to add tracing.

        Patches:
        - ``autogen.ConversableAgent.initiate_chat``
        - ``autogen.ConversableAgent.generate_reply``

        Safe to call multiple times.
        """
        if self._instrumented:
            return
        if not _AUTOGEN_AVAILABLE:
            logger.debug("AutoGenInstrumentor: autogen not installed, skipping")
            return

        tracer = self._tracer

        try:
            from autogen import ConversableAgent

            original_initiate_chat = ConversableAgent.initiate_chat

            def _traced_initiate_chat(
                self_agent: object,
                recipient: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                self_name = getattr(self_agent, "name", "agent")
                target_name = getattr(recipient, "name", "recipient")
                with tracer.agent_delegate(target_agent=target_name, task="initiate_chat") as span:
                    span.set_attribute("autogen.initiator", self_name)
                    span.set_attribute("autogen.recipient", target_name)
                    return original_initiate_chat(self_agent, recipient, *args, **kwargs)  # type: ignore[operator]

            ConversableAgent.initiate_chat = _traced_initiate_chat  # type: ignore[method-assign]
            self._original_initiate_chat = original_initiate_chat
            logger.debug("AutoGenInstrumentor: patched ConversableAgent.initiate_chat")
        except (ImportError, AttributeError) as exc:
            logger.debug("AutoGenInstrumentor: could not patch initiate_chat: %s", exc)

        try:
            from autogen import ConversableAgent

            original_generate_reply = ConversableAgent.generate_reply

            def _traced_generate_reply(
                self_agent: object,
                messages: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                agent_name = getattr(self_agent, "name", "agent")
                llm_config = getattr(self_agent, "llm_config", {}) or {}
                model = llm_config.get("model", "unknown") if isinstance(llm_config, dict) else "unknown"
                with tracer.llm_call(model=str(model), provider="autogen") as span:
                    span.set_attribute("autogen.agent", agent_name)
                    return original_generate_reply(self_agent, messages, *args, **kwargs)  # type: ignore[operator]

            ConversableAgent.generate_reply = _traced_generate_reply  # type: ignore[method-assign]
            self._original_generate_reply = original_generate_reply
            logger.debug("AutoGenInstrumentor: patched ConversableAgent.generate_reply")
        except (ImportError, AttributeError) as exc:
            logger.debug("AutoGenInstrumentor: could not patch generate_reply: %s", exc)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove all patches installed by :meth:`instrument`."""
        if not self._instrumented:
            return

        try:
            from autogen import ConversableAgent

            if self._original_initiate_chat is not None:
                ConversableAgent.initiate_chat = self._original_initiate_chat  # type: ignore[method-assign]
                self._original_initiate_chat = None
            if self._original_generate_reply is not None:
                ConversableAgent.generate_reply = self._original_generate_reply  # type: ignore[method-assign]
                self._original_generate_reply = None
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        logger.debug("AutoGenInstrumentor: uninstrumented")
