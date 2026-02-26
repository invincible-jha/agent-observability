"""SpanEnricher — middleware that auto-enriches spans with agent metadata.

When installed as an OTel SpanProcessor, it stamps every span with the
agent_id, session_id, and framework that were configured at startup.
Falls back gracefully when OTel is not installed.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_observability.spans.conventions import (
    AGENT_ENVIRONMENT,
    AGENT_FRAMEWORK,
    AGENT_ID,
    AGENT_NAME,
    AGENT_SESSION_ID,
    AGENT_VERSION,
)

logger = logging.getLogger(__name__)

try:
    from opentelemetry.sdk.trace import ReadableSpan, Span
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False
    ReadableSpan = None  # type: ignore[assignment,misc]
    Span = None  # type: ignore[assignment,misc]
    SpanExporter = None  # type: ignore[assignment]
    SpanExportResult = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.trace import SpanProcessor

    _PROCESSOR_AVAILABLE = True
except ImportError:
    _PROCESSOR_AVAILABLE = False
    SpanProcessor = object  # type: ignore[assignment,misc]


class SpanEnricher(SpanProcessor):  # type: ignore[misc]
    """OTel SpanProcessor that stamps agent metadata on every span.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent instance.
    session_id:
        Session or conversation identifier.
    framework:
        Framework name (e.g. ``"langchain"``, ``"crewai"``).
    agent_name:
        Human-readable agent name.
    agent_version:
        Semantic version of the agent.
    environment:
        Deployment environment (``"production"``, ``"staging"``, ``"development"``).
    """

    def __init__(
        self,
        agent_id: str = "",
        session_id: str = "",
        framework: str = "",
        agent_name: str = "",
        agent_version: str = "",
        environment: str = "",
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._framework = framework
        self._agent_name = agent_name
        self._agent_version = agent_version
        self._environment = environment

        self._static_attributes: dict[str, str] = {}
        if agent_id:
            self._static_attributes[AGENT_ID] = agent_id
        if session_id:
            self._static_attributes[AGENT_SESSION_ID] = session_id
        if framework:
            self._static_attributes[AGENT_FRAMEWORK] = framework
        if agent_name:
            self._static_attributes[AGENT_NAME] = agent_name
        if agent_version:
            self._static_attributes[AGENT_VERSION] = agent_version
        if environment:
            self._static_attributes[AGENT_ENVIRONMENT] = environment

    def on_start(
        self,
        span: "Span",  # type: ignore[override]
        parent_context: Optional[object] = None,
    ) -> None:
        """Stamp all static attributes when a span starts."""
        if not _OTEL_SDK_AVAILABLE:
            return
        for key, value in self._static_attributes.items():
            try:
                span.set_attribute(key, value)
            except Exception:
                logger.debug("SpanEnricher: could not set attribute %s", key)

    def on_end(self, span: "ReadableSpan") -> None:  # type: ignore[override]
        """No post-processing needed; enrichment happens at start."""

    def shutdown(self) -> None:
        """Clean shutdown — nothing to flush."""

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # ── Convenience ───────────────────────────────────────────────────────────

    def update_session(self, session_id: str) -> None:
        """Replace the session ID (e.g. after a conversation reset)."""
        self._session_id = session_id
        self._static_attributes[AGENT_SESSION_ID] = session_id

    def update_agent_id(self, agent_id: str) -> None:
        """Replace the agent ID (e.g. after a hot-reload)."""
        self._agent_id = agent_id
        self._static_attributes[AGENT_ID] = agent_id
