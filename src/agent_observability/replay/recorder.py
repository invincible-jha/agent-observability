"""Trace recorder — writes agent spans to a JSONL file.

One JSON object per line (JSONL / newline-delimited JSON).  The file starts
with a *session header* record and ends with a *session footer* record that
summarises the captured spans.

Usage::

    from pathlib import Path
    from agent_observability.replay.recorder import TraceRecorder

    recorder = TraceRecorder(Path("/tmp/my-trace.jsonl"))
    recorder.start_session("session-001")
    recorder.record_span({"span_type": "llm_call", "duration_ms": 200})
    recorder.record_span({"span_type": "tool_invoke", "duration_ms": 50})
    recorder.end_session()
"""
from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Record kinds written by the recorder itself (not user spans).
_RECORD_KIND_HEADER = "session_header"
_RECORD_KIND_FOOTER = "session_footer"


class TraceRecorderError(Exception):
    """Raised when the recorder is used incorrectly."""


class TraceRecorder:
    """Appends agent spans to a JSONL file.

    Parameters
    ----------
    output_path:
        Destination file.  Parents are created if they do not exist.
        If the file already exists its contents are preserved; each new
        session is appended.

    Notes
    -----
    The recorder is **not** thread-safe.  Use one recorder per thread or
    protect access externally.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path: Path = output_path
        self._session_id: Optional[str] = None
        self._session_start_time: Optional[float] = None
        self._span_count: int = 0
        self._type_counts: dict[str, int] = {}
        self._file_handle: Optional[io.TextIOWrapper] = None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_session(self, session_id: str) -> None:
        """Open a new recording session.

        Writes a ``session_header`` record to the file.  Must be called before
        :meth:`record_span`.  A session that is already active must be ended
        with :meth:`end_session` before a new one can be started.

        Parameters
        ----------
        session_id:
            Unique identifier for this recording session.

        Raises
        ------
        TraceRecorderError
            If a session is already active.
        """
        if self._session_id is not None:
            raise TraceRecorderError(
                f"Session '{self._session_id}' is already active. "
                "Call end_session() before starting a new one."
            )
        self._session_id = session_id
        self._session_start_time = time.time()
        self._span_count = 0
        self._type_counts = {}
        self._open_file()
        header: dict[str, object] = {
            "record_kind": _RECORD_KIND_HEADER,
            "session_id": session_id,
            "started_at": self._session_start_time,
        }
        self._write_record(header)
        logger.debug("TraceRecorder: started session %s", session_id)

    def record_span(self, span: dict[str, object]) -> None:
        """Append a single span to the JSONL file.

        Parameters
        ----------
        span:
            Arbitrary span dictionary.  A ``record_kind`` key with value
            ``"span"`` is injected automatically.

        Raises
        ------
        TraceRecorderError
            If no session is active (call :meth:`start_session` first).
        """
        if self._session_id is None:
            raise TraceRecorderError(
                "No active session. Call start_session() before record_span()."
            )
        enriched: dict[str, object] = {
            "record_kind": "span",
            "session_id": self._session_id,
            **span,
        }
        self._write_record(enriched)
        self._span_count += 1
        span_type = str(span.get("span_type", "unknown"))
        self._type_counts[span_type] = self._type_counts.get(span_type, 0) + 1

    def end_session(self) -> None:
        """Close the active session and write a ``session_footer`` summary.

        Flushes and closes the underlying file handle.

        Raises
        ------
        TraceRecorderError
            If no session is active.
        """
        if self._session_id is None:
            raise TraceRecorderError("No active session to end.")

        ended_at = time.time()
        duration_seconds: float = (
            ended_at - self._session_start_time
            if self._session_start_time is not None
            else 0.0
        )
        footer: dict[str, object] = {
            "record_kind": _RECORD_KIND_FOOTER,
            "session_id": self._session_id,
            "ended_at": ended_at,
            "total_spans": self._span_count,
            "duration_seconds": duration_seconds,
            "span_type_counts": dict(self._type_counts),
        }
        self._write_record(footer)
        self.flush()
        self._close_file()
        logger.debug(
            "TraceRecorder: ended session %s (%d spans)", self._session_id, self._span_count
        )
        self._session_id = None
        self._session_start_time = None
        self._span_count = 0
        self._type_counts = {}

    def flush(self) -> None:
        """Flush the internal write buffer to disk.

        Safe to call even when no file handle is open.
        """
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.flush()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def output_path(self) -> Path:
        """The destination file path."""
        return self._output_path

    @property
    def active_session_id(self) -> Optional[str]:
        """The current session ID, or ``None`` if no session is active."""
        return self._session_id

    @property
    def span_count(self) -> int:
        """Number of spans recorded in the current session."""
        return self._span_count

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _open_file(self) -> None:
        """Open the output file in append mode."""
        self._file_handle = self._output_path.open("a", encoding="utf-8")

    def _close_file(self) -> None:
        """Close the output file if it is open."""
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
        self._file_handle = None

    def _write_record(self, record: dict[str, object]) -> None:
        """Serialise *record* as a JSONL line and write it."""
        if self._file_handle is None or self._file_handle.closed:
            raise TraceRecorderError("File handle is not open. Start a session first.")
        line = json.dumps(record, default=str) + "\n"
        self._file_handle.write(line)
