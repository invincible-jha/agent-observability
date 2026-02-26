"""CrewAIInstrumentor — auto-trace CrewAI task execution via monkey-patching.

Wraps ``Task.execute`` and ``Agent.execute_task`` with agent-semantic spans.
The CrewAI import is guarded; if CrewAI is not installed the instrumentor
silently becomes a no-op.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.tracer.agent_tracer import AgentTracer

logger = logging.getLogger(__name__)

try:
    import crewai  # noqa: F401

    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False


class CrewAIInstrumentor:
    """Instrument CrewAI with agent-semantic tracing.

    Parameters
    ----------
    tracer:
        The :class:`~agent_observability.tracer.AgentTracer` instance to use.
    """

    def __init__(self, tracer: AgentTracer) -> None:
        self._tracer = tracer
        self._original_task_execute: Optional[object] = None
        self._original_agent_execute: Optional[object] = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch CrewAI to add tracing.

        Patches:
        - ``crewai.Task.execute``
        - ``crewai.Agent.execute_task``

        Safe to call multiple times.
        """
        if self._instrumented:
            return
        if not _CREWAI_AVAILABLE:
            logger.debug("CrewAIInstrumentor: crewai not installed, skipping")
            return

        tracer = self._tracer

        try:
            from crewai import Task

            original_task_execute = Task.execute

            def _traced_task_execute(
                self_task: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                task_description = getattr(self_task, "description", "crewai.task")
                agent_attr = getattr(self_task, "agent", None)
                target = getattr(agent_attr, "role", "unknown") if agent_attr else "unknown"
                with tracer.agent_delegate(target_agent=target, task=task_description[:100]) as span:
                    span.set_attribute("crewai.task.description", str(task_description)[:256])
                    return original_task_execute(self_task, *args, **kwargs)  # type: ignore[operator]

            Task.execute = _traced_task_execute  # type: ignore[method-assign]
            self._original_task_execute = original_task_execute
            logger.debug("CrewAIInstrumentor: patched Task.execute")
        except (ImportError, AttributeError) as exc:
            logger.debug("CrewAIInstrumentor: could not patch Task: %s", exc)

        try:
            from crewai import Agent

            original_agent_execute = Agent.execute_task

            def _traced_agent_execute(
                self_agent: object,
                task: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                agent_role = getattr(self_agent, "role", "unknown_agent")
                with tracer.reasoning_step(step_name="execute_task", description=agent_role) as span:
                    span.set_attribute("crewai.agent.role", agent_role)
                    return original_agent_execute(self_agent, task, *args, **kwargs)  # type: ignore[operator]

            Agent.execute_task = _traced_agent_execute  # type: ignore[method-assign]
            self._original_agent_execute = original_agent_execute
            logger.debug("CrewAIInstrumentor: patched Agent.execute_task")
        except (ImportError, AttributeError) as exc:
            logger.debug("CrewAIInstrumentor: could not patch Agent: %s", exc)

        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove all patches installed by :meth:`instrument`."""
        if not self._instrumented:
            return

        try:
            from crewai import Task

            if self._original_task_execute is not None:
                Task.execute = self._original_task_execute  # type: ignore[method-assign]
                self._original_task_execute = None
        except (ImportError, AttributeError):
            pass

        try:
            from crewai import Agent

            if self._original_agent_execute is not None:
                Agent.execute_task = self._original_agent_execute  # type: ignore[method-assign]
                self._original_agent_execute = None
        except (ImportError, AttributeError):
            pass

        self._instrumented = False
        logger.debug("CrewAIInstrumentor: uninstrumented")
