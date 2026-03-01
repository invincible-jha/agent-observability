"""ConsoleExporter — write AgentSpan data to stdout for development/debugging."""
from __future__ import annotations

import json
import logging
import time

from agent_observability.spans.types import AgentSpan

logger = logging.getLogger(__name__)


class ConsoleExporter:
    """Write agent spans to stdout as JSON lines.

    Intended for local development and debugging.  Each call to
    :meth:`export` emits one JSON object per span to stdout.

    Parameters
    ----------
    pretty:
        When *True* each span is printed with indentation.  Defaults to
        *False* for machine-readable single-line output.
    """

    def __init__(self, pretty: bool = False) -> None:
        self._pretty = pretty

    def export(self, spans: list[AgentSpan]) -> None:
        """Print each span in *spans* to stdout as a JSON object.

        Parameters
        ----------
        spans:
            List of :class:`~agent_observability.spans.types.AgentSpan` to
            print.
        """
        for span in spans:
            underlying = span._span
            attributes_raw: dict[str, object] = {}
            if hasattr(underlying, "_attributes"):
                attributes_raw = dict(underlying._attributes)

            now_ns = int(time.time() * 1e9)
            elapsed_ns = int(span.elapsed_seconds * 1e9)

            record: dict[str, object] = {
                "exporter": "console",
                "name": attributes_raw.get("agent.span.kind", "agent.span"),
                "startTimeUnixNano": str(now_ns - elapsed_ns),
                "endTimeUnixNano": str(now_ns),
                "attributes": attributes_raw,
            }

            indent = 2 if self._pretty else None
            print(json.dumps(record, default=str, indent=indent))
