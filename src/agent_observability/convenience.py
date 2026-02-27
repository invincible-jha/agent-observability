"""Convenience API for agent-observability — 3-line quickstart.

Example
-------
::

    from agent_observability import Tracer
    tracer = Tracer()
    with tracer.trace("my-operation"):
        pass  # your code here

"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator


class Tracer:
    """Zero-config agent tracer for the 80% use case.

    Wraps AgentTracer with sensible defaults so you can start tracing
    immediately without configuring an OTLP endpoint.

    Parameters
    ----------
    service_name:
        Logical service name included in every exported span.
    agent_id:
        Agent identifier forwarded to the underlying tracer.

    Example
    -------
    ::

        from agent_observability import Tracer
        tracer = Tracer()
        with tracer.trace("my-llm-call"):
            result = call_llm(prompt)
    """

    def __init__(
        self,
        service_name: str = "agent",
        agent_id: str = "default-agent",
    ) -> None:
        from agent_observability.tracer.agent_tracer import AgentTracer

        self._tracer = AgentTracer(
            service_name=service_name,
            agent_id=agent_id,
        )
        self.service_name = service_name
        self.agent_id = agent_id

    @contextmanager
    def trace(self, name: str) -> Generator[Any, None, None]:
        """Generic trace context manager.

        Parameters
        ----------
        name:
            Human-readable name for the operation being traced.

        Yields
        ------
        AgentSpan
            The active span; call ``.set_attribute(key, value)`` on it
            to attach metadata.

        Example
        -------
        ::

            with tracer.trace("data-retrieval") as span:
                span.set_attribute("source", "database")
                data = fetch_data()
        """
        with self._tracer.reasoning_step(name, description=name) as span:
            yield span

    @contextmanager
    def trace_llm(self, model: str, provider: str = "unknown") -> Generator[Any, None, None]:
        """Trace an LLM call with model and provider metadata.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"claude-sonnet-4"``).
        provider:
            Provider name (e.g. ``"anthropic"``).

        Yields
        ------
        AgentSpan
        """
        with self._tracer.llm_call(model=model, provider=provider) as span:
            yield span

    def export(self) -> list[dict[str, Any]]:
        """Return all collected spans as serialisable dicts.

        Returns
        -------
        list[dict[str, Any]]
            List of span metadata dicts.
        """
        return self._tracer.export()

    def flush(self) -> None:
        """Clear all stored spans from local collection."""
        self._tracer.flush()

    @property
    def underlying(self) -> Any:
        """The underlying AgentTracer instance."""
        return self._tracer

    def __repr__(self) -> str:
        return f"Tracer(service_name={self.service_name!r}, agent_id={self.agent_id!r})"
