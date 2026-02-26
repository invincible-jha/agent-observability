"""AgentTraceContext — propagate trace context across agent boundaries.

Allows a parent agent to hand off its trace context to a child agent
via a serialisable dict, which the child can then attach before starting
its own spans.  Works with or without OTel installed.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import context as otel_context
    from opentelemetry import propagate
    from opentelemetry.context import Context

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    otel_context = None  # type: ignore[assignment]
    propagate = None  # type: ignore[assignment]
    Context = None  # type: ignore[assignment,misc]


class AgentTraceContext:
    """Serialisable trace context for cross-agent propagation.

    Example
    -------
    Parent agent::

        ctx = AgentTraceContext.extract_current()
        message = {"payload": ..., "trace_ctx": ctx.to_headers()}

    Child agent::

        ctx = AgentTraceContext.from_headers(message["trace_ctx"])
        token = ctx.attach()
        try:
            # child spans are now children of the parent trace
            ...
        finally:
            ctx.detach(token)
    """

    _W3C_KEYS: tuple[str, ...] = ("traceparent", "tracestate", "baggage")

    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = headers

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def extract_current(cls) -> "AgentTraceContext":
        """Build a context object from the currently-active OTel context."""
        if not _OTEL_AVAILABLE or propagate is None:
            return cls({})
        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        return cls(carrier)

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "AgentTraceContext":
        """Build a context from a dict of W3C trace headers."""
        return cls({k: v for k, v in headers.items() if k in cls._W3C_KEYS or k.startswith("x-")})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def attach(self) -> Optional[object]:
        """Attach this context as the active OTel context.

        Returns the token needed to detach later (or ``None`` if OTel is absent).
        """
        if not _OTEL_AVAILABLE or propagate is None or otel_context is None:
            return None
        ctx = propagate.extract(self._headers)
        token = otel_context.attach(ctx)
        return token

    def detach(self, token: Optional[object]) -> None:
        """Detach a previously-attached context."""
        if not _OTEL_AVAILABLE or token is None or otel_context is None:
            return
        try:
            otel_context.detach(token)  # type: ignore[arg-type]
        except Exception:
            logger.debug("AgentTraceContext.detach: failed to detach token")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_headers(self) -> dict[str, str]:
        """Return the W3C headers suitable for injection into any transport."""
        return dict(self._headers)

    def is_valid(self) -> bool:
        """Return ``True`` if at least a ``traceparent`` header is present."""
        return bool(self._headers.get("traceparent"))

    def __repr__(self) -> str:
        return f"AgentTraceContext(valid={self.is_valid()}, headers={list(self._headers.keys())})"
