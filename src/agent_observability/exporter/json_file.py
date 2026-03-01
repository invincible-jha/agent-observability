"""JSONFileExporter — write AgentSpan data to a JSON Lines file."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent_observability.spans.types import AgentSpan

logger = logging.getLogger(__name__)


class JSONFileExporter:
    """Append agent spans to a JSON Lines file.

    Each call to :meth:`export` appends one JSON object per span to *path*.
    The file is opened in append mode so existing content is preserved.

    Parameters
    ----------
    path:
        File system path to write JSON Lines output.  The parent directory
        must already exist.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def export(self, spans: list[AgentSpan]) -> None:
        """Append each span in *spans* to the configured JSON Lines file.

        Parameters
        ----------
        spans:
            List of :class:`~agent_observability.spans.types.AgentSpan` to
            write.
        """
        if not spans:
            return

        with self._path.open("a", encoding="utf-8") as file_handle:
            for span in spans:
                underlying = span._span
                attributes_raw: dict[str, object] = {}
                if hasattr(underlying, "_attributes"):
                    attributes_raw = dict(underlying._attributes)

                now_ns = int(time.time() * 1e9)
                elapsed_ns = int(span.elapsed_seconds * 1e9)

                record: dict[str, object] = {
                    "exporter": "json_file",
                    "name": attributes_raw.get("agent.span.kind", "agent.span"),
                    "startTimeUnixNano": str(now_ns - elapsed_ns),
                    "endTimeUnixNano": str(now_ns),
                    "attributes": attributes_raw,
                }

                file_handle.write(json.dumps(record, default=str) + "\n")
