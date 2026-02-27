"""CorrelationContext — trace context propagated across agent boundaries.

Carries the trace_id, parent_span_id, and baggage items that allow spans
from different agents to be linked into a coherent distributed trace.

Example
-------
::

    from agent_observability.correlation.correlation_context import CorrelationContext

    ctx = CorrelationContext.new_root()
    child_ctx = ctx.new_child_span("child-span-id")
    print(child_ctx.parent_span_id == ctx.span_id)  # True
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BaggageItem:
    """A single baggage key-value pair propagated through a trace.

    Baggage items carry cross-cutting concerns like tenant ID, user ID,
    or feature flags across all spans in a trace.

    Attributes
    ----------
    key:
        Baggage item key. Must be non-empty.
    value:
        Baggage item value.
    """

    key: str
    value: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("BaggageItem key must be non-empty.")

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dictionary."""
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True)
class CorrelationContext:
    """Trace context that propagates across agent boundaries.

    Carries a trace_id (shared across all spans in a trace), a span_id
    (unique to the current operation), and optional parent_span_id for
    linking parent and child spans.

    Attributes
    ----------
    trace_id:
        Globally unique trace identifier. Shared across all spans in one trace.
    span_id:
        Unique identifier for the current span.
    parent_span_id:
        ID of the parent span. None for root spans.
    baggage:
        Tuple of BaggageItem instances propagated through the trace.
    sampled:
        Whether this trace is being sampled for collection.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: tuple[BaggageItem, ...] = field(default_factory=tuple)
    sampled: bool = True

    @classmethod
    def new_root(
        cls,
        *,
        trace_id: Optional[str] = None,
        sampled: bool = True,
    ) -> "CorrelationContext":
        """Create a new root context (no parent).

        Parameters
        ----------
        trace_id:
            Optional trace ID. Auto-generated if not provided.
        sampled:
            Whether this trace is sampled. Defaults to True.

        Returns
        -------
        CorrelationContext
            A new root context with no parent_span_id.
        """
        return cls(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            sampled=sampled,
        )

    def new_child_span(
        self,
        span_id: Optional[str] = None,
    ) -> "CorrelationContext":
        """Create a child context inheriting trace_id and setting parent_span_id.

        Parameters
        ----------
        span_id:
            Optional span ID for the child. Auto-generated if not provided.

        Returns
        -------
        CorrelationContext
            A new child context with this span's span_id as parent_span_id.
        """
        return CorrelationContext(
            trace_id=self.trace_id,
            span_id=span_id or str(uuid.uuid4()),
            parent_span_id=self.span_id,
            baggage=self.baggage,
            sampled=self.sampled,
        )

    @property
    def is_root(self) -> bool:
        """True if this context has no parent span."""
        return self.parent_span_id is None

    def get_baggage(self, key: str) -> Optional[str]:
        """Return the value of a baggage item by key, or None if not found."""
        for item in self.baggage:
            if item.key == key:
                return item.value
        return None

    def with_baggage(self, key: str, value: str) -> "CorrelationContext":
        """Return a new context with an additional baggage item.

        If a baggage item with the same key already exists, it is replaced.

        Parameters
        ----------
        key:
            Baggage item key.
        value:
            Baggage item value.

        Returns
        -------
        CorrelationContext
            New context with the added/updated baggage item.
        """
        existing = tuple(item for item in self.baggage if item.key != key)
        new_baggage = existing + (BaggageItem(key=key, value=value),)
        return CorrelationContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            baggage=new_baggage,
            sampled=self.sampled,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary for header injection."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sampled": self.sampled,
            "baggage": [item.to_dict() for item in self.baggage],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CorrelationContext":
        """Reconstruct a CorrelationContext from a header dictionary.

        Parameters
        ----------
        data:
            Dictionary as produced by :meth:`to_dict` or extracted from
            HTTP headers.

        Returns
        -------
        CorrelationContext
            Reconstructed context.

        Raises
        ------
        ValueError
            If required fields are missing.
        """
        trace_id = data.get("trace_id")
        span_id = data.get("span_id")
        if not trace_id or not span_id:
            raise ValueError("CorrelationContext requires 'trace_id' and 'span_id'.")

        raw_baggage = data.get("baggage", [])
        baggage_items: list[BaggageItem] = []
        if isinstance(raw_baggage, list):
            for item_dict in raw_baggage:
                if isinstance(item_dict, dict) and "key" in item_dict and "value" in item_dict:
                    baggage_items.append(
                        BaggageItem(key=str(item_dict["key"]), value=str(item_dict["value"]))
                    )

        return cls(
            trace_id=str(trace_id),
            span_id=str(span_id),
            parent_span_id=data.get("parent_span_id"),  # type: ignore[arg-type]
            baggage=tuple(baggage_items),
            sampled=bool(data.get("sampled", True)),
        )

    def to_headers(self) -> dict[str, str]:
        """Render as W3C traceparent-compatible headers.

        Returns
        -------
        dict[str, str]
            HTTP headers for trace context propagation.
        """
        headers: dict[str, str] = {}
        # W3C traceparent format: version-traceid-spanid-flags
        flags = "01" if self.sampled else "00"
        # Use first 32 hex chars of trace_id (strip dashes, pad if needed)
        trace_hex = self.trace_id.replace("-", "")[:32].ljust(32, "0")
        span_hex = self.span_id.replace("-", "")[:16].ljust(16, "0")
        headers["traceparent"] = f"00-{trace_hex}-{span_hex}-{flags}"

        if self.parent_span_id:
            headers["x-parent-span-id"] = self.parent_span_id

        if self.baggage:
            baggage_str = ",".join(f"{item.key}={item.value}" for item in self.baggage)
            headers["baggage"] = baggage_str

        return headers
