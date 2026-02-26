"""AgentTracerProvider — configure OTel with agent-specific span processors.

``setup_tracing()`` is the single entry point that most users will call.
It wires up the provider, sampler, enricher, and exporter in one shot.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.spans.enricher import SpanEnricher
from agent_observability.tracing.exporter import (
    AgentSpanExporter,
    ConsoleSpanExporter,
    JsonLinesExporter,
    build_otlp_exporter,
)
from agent_observability.tracing.sampler import CostAwareSampler

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False
    otel_trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment,misc]
    TracerProvider = None  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment,misc]
    SimpleSpanProcessor = None  # type: ignore[assignment,misc]


class AgentTracerProvider:
    """Thin wrapper around the OTel ``TracerProvider`` adding agent conveniences.

    Parameters
    ----------
    service_name:
        Reported in OTel resource attributes.
    service_version:
        Reported in OTel resource attributes.
    agent_id:
        Injected into every span via :class:`SpanEnricher`.
    session_id:
        Injected into every span via :class:`SpanEnricher`.
    framework:
        Injected into every span (e.g. ``"langchain"``).
    environment:
        Deployment environment (``"production"``, ``"staging"``, …).
    """

    def __init__(
        self,
        service_name: str = "agent",
        service_version: str = "0.0.0",
        agent_id: str = "",
        session_id: str = "",
        framework: str = "",
        environment: str = "",
    ) -> None:
        self._service_name = service_name
        self._service_version = service_version
        self._agent_id = agent_id
        self._session_id = session_id
        self._framework = framework
        self._environment = environment
        self._provider: Optional[object] = None
        self._enricher: Optional[SpanEnricher] = None

    def build(
        self,
        exporter: Optional[object] = None,
        sampler: Optional[object] = None,
        batch: bool = True,
    ) -> Optional[object]:
        """Create and register the OTel TracerProvider.

        Parameters
        ----------
        exporter:
            A SpanExporter instance.  Defaults to :class:`ConsoleSpanExporter`.
        sampler:
            A Sampler instance.  Defaults to :class:`CostAwareSampler`.
        batch:
            Use ``BatchSpanProcessor`` when ``True``, ``SimpleSpanProcessor``
            when ``False``.

        Returns
        -------
        The configured ``TracerProvider``, or ``None`` if OTel SDK is absent.
        """
        if not _OTEL_SDK_AVAILABLE:
            logger.warning(
                "opentelemetry-sdk not installed; AgentTracerProvider.build() is a no-op"
            )
            return None

        assert Resource is not None
        assert TracerProvider is not None
        assert BatchSpanProcessor is not None
        assert SimpleSpanProcessor is not None

        resource = Resource.create(
            {
                "service.name": self._service_name,
                "service.version": self._service_version,
                "agent.id": self._agent_id,
                "deployment.environment": self._environment,
            }
        )

        chosen_sampler = sampler or CostAwareSampler()
        provider = TracerProvider(resource=resource, sampler=chosen_sampler)

        # Span enricher processor
        self._enricher = SpanEnricher(
            agent_id=self._agent_id,
            session_id=self._session_id,
            framework=self._framework,
            environment=self._environment,
        )
        provider.add_span_processor(self._enricher)

        # Export processor
        chosen_exporter = exporter or ConsoleSpanExporter()
        processor_cls = BatchSpanProcessor if batch else SimpleSpanProcessor
        provider.add_span_processor(processor_cls(chosen_exporter))

        otel_trace.set_tracer_provider(provider)  # type: ignore[union-attr]
        self._provider = provider
        logger.info(
            "AgentTracerProvider configured for service '%s'", self._service_name
        )
        return provider

    def get_tracer(self, name: str = "agent-observability") -> object:
        """Return a tracer scoped to *name*."""
        if not _OTEL_SDK_AVAILABLE or otel_trace is None:
            return _NoOpTracer()
        return otel_trace.get_tracer(name)

    def shutdown(self) -> None:
        """Flush and shut down all span processors."""
        if self._provider is not None and hasattr(self._provider, "shutdown"):
            self._provider.shutdown()  # type: ignore[union-attr]


class _NoOpTracer:
    """Fallback tracer used when OTel SDK is not installed."""

    def start_span(self, name: str, **kwargs: object) -> object:
        return _NoOpSpanHandle(name)

    def start_as_current_span(self, name: str, **kwargs: object) -> object:
        from contextlib import contextmanager

        @contextmanager  # type: ignore[misc]
        def _ctx() -> object:  # type: ignore[misc]
            yield _NoOpSpanHandle(name)

        return _ctx()


class _NoOpSpanHandle:
    def __init__(self, name: str) -> None:
        self.name = name

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def set_status(self, status: object, description: str = "") -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpanHandle":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def setup_tracing(
    service_name: str = "agent",
    service_version: str = "0.0.0",
    agent_id: str = "",
    session_id: str = "",
    framework: str = "",
    environment: str = "development",
    otlp_endpoint: str = "",
    otlp_use_http: bool = False,
    jsonl_path: str = "",
    console: bool = False,
    batch: bool = True,
    high_cost_threshold_usd: float = 0.01,
    low_cost_sample_rate: float = 0.1,
) -> AgentTracerProvider:
    """One-shot helper to configure agent observability tracing.

    Parameters
    ----------
    service_name:
        OTel resource ``service.name``.
    service_version:
        OTel resource ``service.version``.
    agent_id:
        Agent identifier stamped on every span.
    session_id:
        Session identifier stamped on every span.
    framework:
        Framework name stamped on every span.
    environment:
        Deployment environment.
    otlp_endpoint:
        If non-empty, send spans to this OTLP collector endpoint.
    otlp_use_http:
        Use HTTP transport for OTLP (default: gRPC).
    jsonl_path:
        If non-empty, also write spans to this JSON Lines file.
    console:
        If ``True``, also emit spans to the console logger.
    batch:
        Use ``BatchSpanProcessor`` (recommended for production).
    high_cost_threshold_usd:
        Traces above this cost are always sampled.
    low_cost_sample_rate:
        Fraction of cheap traces to keep.

    Returns
    -------
    The configured :class:`AgentTracerProvider`.
    """
    provider = AgentTracerProvider(
        service_name=service_name,
        service_version=service_version,
        agent_id=agent_id,
        session_id=session_id,
        framework=framework,
        environment=environment,
    )

    # Build the fan-out exporter
    downstream_exporters: list[object] = []
    if otlp_endpoint:
        downstream_exporters.append(
            build_otlp_exporter(
                endpoint=otlp_endpoint,
                use_http=otlp_use_http,
            )
        )
    if jsonl_path:
        downstream_exporters.append(JsonLinesExporter(path=jsonl_path))
    if console or not downstream_exporters:
        downstream_exporters.append(ConsoleSpanExporter())

    fan_out = AgentSpanExporter(
        exporters=downstream_exporters,  # type: ignore[arg-type]
        agent_id=agent_id,
        session_id=session_id,
    )

    sampler = CostAwareSampler(
        high_cost_threshold_usd=high_cost_threshold_usd,
        low_cost_sample_rate=low_cost_sample_rate,
    )

    provider.build(exporter=fan_out, sampler=sampler, batch=batch)
    return provider
