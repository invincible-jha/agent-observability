"""CrossAgentPropagator — propagate trace headers in agent-to-agent messages.

Provides a thin façade over OTel's W3C propagator so that agent message
buses (queues, HTTP, gRPC, in-process) can carry trace context without
coupling to OTel internals.
"""
from __future__ import annotations

import logging
from typing import Callable, MutableMapping, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import propagate
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.propagators.textmap import (
        CarrierT,
        DefaultGetter,
        DefaultSetter,
        TextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    propagate = None  # type: ignore[assignment]
    CompositePropagator = None  # type: ignore[assignment,misc]
    TextMapPropagator = None  # type: ignore[assignment]
    DefaultGetter = None  # type: ignore[assignment]
    DefaultSetter = None  # type: ignore[assignment]
    CarrierT = None  # type: ignore[assignment]

# Type alias for carrier dicts
Carrier = MutableMapping[str, str]

_AGENT_CTX_PREFIX = "x-agent-ctx-"


class CrossAgentPropagator:
    """Inject and extract trace context from arbitrary message carriers.

    In addition to standard W3C ``traceparent``/``tracestate`` headers, this
    propagator also injects optional *agent-level* metadata (agent_id,
    session_id) under the ``x-agent-ctx-*`` prefix.

    Parameters
    ----------
    agent_id:
        Value to inject under ``x-agent-ctx-agent-id``.
    session_id:
        Value to inject under ``x-agent-ctx-session-id``.
    extra_fields:
        Additional ``{header_name: value}`` pairs to inject.
    """

    def __init__(
        self,
        agent_id: str = "",
        session_id: str = "",
        extra_fields: Optional[dict[str, str]] = None,
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._extra_fields: dict[str, str] = extra_fields or {}

    # ── Inject ────────────────────────────────────────────────────────────────

    def inject(self, carrier: Carrier) -> None:
        """Inject the current active trace context into *carrier*.

        Also injects agent-level metadata fields.
        """
        if _OTEL_AVAILABLE and propagate is not None:
            propagate.inject(carrier)

        if self._agent_id:
            carrier[f"{_AGENT_CTX_PREFIX}agent-id"] = self._agent_id
        if self._session_id:
            carrier[f"{_AGENT_CTX_PREFIX}session-id"] = self._session_id
        for key, value in self._extra_fields.items():
            carrier[key] = value

    # ── Extract ───────────────────────────────────────────────────────────────

    def extract(self, carrier: Carrier) -> dict[str, str]:
        """Extract agent-level metadata from *carrier*.

        The OTel context is restored as a side-effect (via ``propagate.extract``).
        Returns a dict of the *agent* metadata fields found.
        """
        agent_meta: dict[str, str] = {}

        if _OTEL_AVAILABLE and propagate is not None:
            try:
                propagate.extract(carrier)
            except Exception:
                logger.debug("CrossAgentPropagator.extract: OTel extract failed")

        for key, value in carrier.items():
            if key.startswith(_AGENT_CTX_PREFIX):
                short_key = key[len(_AGENT_CTX_PREFIX):]
                agent_meta[short_key] = value

        return agent_meta

    # ── Convenience ───────────────────────────────────────────────────────────

    def make_carrier(self) -> dict[str, str]:
        """Return a fresh dict with the current context already injected."""
        carrier: dict[str, str] = {}
        self.inject(carrier)
        return carrier

    @staticmethod
    def get_trace_id(carrier: Carrier) -> str:
        """Parse the trace-id hex string from a ``traceparent`` header if present."""
        traceparent = carrier.get("traceparent", "")
        # W3C format: 00-<trace-id>-<parent-id>-<flags>
        parts = traceparent.split("-")
        if len(parts) == 4:
            return parts[1]
        return ""

    @staticmethod
    def get_span_id(carrier: Carrier) -> str:
        """Parse the span-id hex string from a ``traceparent`` header if present."""
        traceparent = carrier.get("traceparent", "")
        parts = traceparent.split("-")
        if len(parts) == 4:
            return parts[2]
        return ""


def inject_into_dict(
    carrier: Carrier,
    agent_id: str = "",
    session_id: str = "",
) -> None:
    """Module-level helper: inject trace context into an existing dict."""
    prop = CrossAgentPropagator(agent_id=agent_id, session_id=session_id)
    prop.inject(carrier)


def extract_from_dict(carrier: Carrier) -> dict[str, str]:
    """Module-level helper: extract trace context from a dict."""
    prop = CrossAgentPropagator()
    return prop.extract(carrier)
