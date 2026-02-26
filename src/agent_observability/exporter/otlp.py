"""OTLPExporter — export AgentSpan collections to an OTLP/HTTP endpoint.

Converts each :class:`~agent_observability.spans.types.AgentSpan` to the
OTLP JSON wire format and POSTs to the configured collector endpoint using
only the stdlib ``urllib`` — no third-party HTTP library required.

The OTLP JSON format is a minimal representation compatible with OpenTelemetry
Collector's HTTP receiver.  Full protobuf OTLP encoding would require
``opentelemetry-exporter-otlp-proto-http``; callers that need that should use
the OTel SDK exporter directly.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

from agent_observability.spans.types import AgentSpan

logger = logging.getLogger(__name__)

# OTLP HTTP Protobuf Content-Type (we send JSON)
_CONTENT_TYPE = "application/json"
_DEFAULT_TIMEOUT_SECONDS = 10


def _agent_span_to_otlp_dict(span: AgentSpan) -> dict[str, object]:
    """Convert an :class:`AgentSpan` to an OTLP-compatible span dict.

    This produces a structure matching the OTLP JSON encoding for a single
    span as described in the OpenTelemetry proto specification.
    """
    # Pull attributes from the underlying _NoOpSpan if available
    underlying = span._span
    attributes_raw: dict[str, object] = {}
    if hasattr(underlying, "_attributes"):
        attributes_raw = dict(underlying._attributes)

    def _attr_value(v: object) -> dict[str, object]:
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        return {"stringValue": str(v)}

    otel_attributes = [
        {"key": k, "value": _attr_value(v)}
        for k, v in attributes_raw.items()
    ]

    now_ns = int(time.time() * 1e9)
    elapsed_ns = int(span.elapsed_seconds * 1e9)

    return {
        "traceId": "00000000000000000000000000000000",
        "spanId": "0000000000000000",
        "name": attributes_raw.get("agent.span.kind", "agent.span"),
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(now_ns - elapsed_ns),
        "endTimeUnixNano": str(now_ns),
        "attributes": otel_attributes,
        "status": {},
    }


class OTLPExporter:
    """Export agent spans to an OTLP/HTTP JSON endpoint.

    Parameters
    ----------
    endpoint:
        Full OTLP HTTP traces endpoint URL, e.g.
        ``"http://localhost:4318/v1/traces"``.
    headers:
        Additional HTTP headers (e.g. authentication tokens).
    timeout_seconds:
        HTTP request timeout in seconds.
    service_name:
        ``service.name`` resource attribute to include in every export batch.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        headers: Optional[dict[str, str]] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        service_name: str = "agent",
    ) -> None:
        self._endpoint = endpoint
        self._headers: dict[str, str] = headers or {}
        self._timeout = timeout_seconds
        self._service_name = service_name

    def export(self, spans: list[AgentSpan]) -> None:
        """Convert *spans* to OTLP JSON and POST to the configured endpoint.

        Failures are logged at WARNING level; exceptions are not re-raised so
        that export failures do not disrupt agent execution.

        Parameters
        ----------
        spans:
            List of :class:`~agent_observability.spans.types.AgentSpan` to export.
        """
        if not spans:
            return

        payload = self._build_payload(spans)
        body = json.dumps(payload, default=str).encode("utf-8")

        request = urllib.request.Request(
            url=self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": _CONTENT_TYPE,
                "Content-Length": str(len(body)),
                **self._headers,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                status = resp.status
                if status not in (200, 204):
                    logger.warning(
                        "OTLPExporter: unexpected status %d from %s",
                        status,
                        self._endpoint,
                    )
                else:
                    logger.debug(
                        "OTLPExporter: exported %d spans to %s (status=%d)",
                        len(spans),
                        self._endpoint,
                        status,
                    )
        except urllib.error.URLError as exc:
            logger.warning(
                "OTLPExporter: failed to export to %s — %s",
                self._endpoint,
                exc,
            )
        except Exception as exc:
            logger.warning("OTLPExporter: unexpected error — %s", exc)

    def _build_payload(self, spans: list[AgentSpan]) -> dict[str, object]:
        """Build the OTLP resource spans payload."""
        span_dicts = [_agent_span_to_otlp_dict(s) for s in spans]

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._service_name},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "agent-observability",
                                "version": "0.1.0",
                            },
                            "spans": span_dicts,
                        }
                    ],
                }
            ]
        }
