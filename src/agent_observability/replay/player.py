"""Trace player — reads and replays agent spans from a JSONL file.

Usage::

    from pathlib import Path
    from agent_observability.replay.player import TracePlayer

    player = TracePlayer(Path("/tmp/my-trace.jsonl"))
    spans = player.load()

    # Step through individual spans
    first = player.step(0)

    # Stream all spans lazily
    for span in player.play_all():
        print(span)

    # Filter by span type
    llm_spans = player.filter_by_type("llm_call")

    # Get a summary
    info = player.summary()
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Record kinds injected by TraceRecorder — excluded from user-visible spans.
_META_RECORD_KINDS = {"session_header", "session_footer"}


class TracePlayerError(Exception):
    """Raised when the player encounters an invalid state or bad input."""


class TracePlayer:
    """Reads and replays agent execution traces from a JSONL file.

    Parameters
    ----------
    trace_path:
        Path to the JSONL trace file written by :class:`TraceRecorder`.

    Notes
    -----
    Call :meth:`load` before using any other method.  All methods that
    operate on spans will raise :class:`TracePlayerError` if the file has
    not been loaded yet.
    """

    def __init__(self, trace_path: Path) -> None:
        self._trace_path: Path = trace_path
        self._spans: list[dict[str, object]] = []
        self._loaded: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> list[dict[str, object]]:
        """Parse the JSONL file and cache spans in memory.

        Only records with ``record_kind == "span"`` (or records that carry no
        ``record_kind`` key at all) are returned and cached.  Session header
        and footer records are silently skipped.

        Returns
        -------
        list[dict]
            The ordered list of span dictionaries.

        Raises
        ------
        TracePlayerError
            If the file does not exist or contains invalid JSONL.
        """
        if not self._trace_path.exists():
            raise TracePlayerError(f"Trace file not found: {self._trace_path}")

        spans: list[dict[str, object]] = []
        raw_text = self._trace_path.read_text(encoding="utf-8")
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record: dict[str, object] = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TracePlayerError(
                    f"Invalid JSON on line {line_number} of {self._trace_path}: {exc}"
                ) from exc
            record_kind = record.get("record_kind")
            if record_kind in _META_RECORD_KINDS:
                continue
            spans.append(record)

        self._spans = spans
        self._loaded = True
        logger.debug("TracePlayer: loaded %d spans from %s", len(spans), self._trace_path)
        return spans

    def step(self, index: int) -> dict[str, object]:
        """Return the span at position *index*.

        Parameters
        ----------
        index:
            Zero-based index into the loaded span list.

        Returns
        -------
        dict
            The span at the given index.

        Raises
        ------
        TracePlayerError
            If :meth:`load` has not been called.
        IndexError
            If *index* is out of range.
        """
        self._require_loaded()
        return self._spans[index]

    def play_all(self) -> Iterator[dict[str, object]]:
        """Yield spans in recorded order.

        Yields
        ------
        dict
            Each span dictionary in sequence.

        Raises
        ------
        TracePlayerError
            If :meth:`load` has not been called.
        """
        self._require_loaded()
        yield from self._spans

    def filter_by_type(self, span_type: str) -> list[dict[str, object]]:
        """Return all spans whose ``span_type`` field matches *span_type*.

        Parameters
        ----------
        span_type:
            Value to match against the ``span_type`` key in each span.

        Returns
        -------
        list[dict]
            Filtered list (may be empty if no spans match).

        Raises
        ------
        TracePlayerError
            If :meth:`load` has not been called.
        """
        self._require_loaded()
        return [span for span in self._spans if span.get("span_type") == span_type]

    def summary(self) -> dict[str, object]:
        """Return aggregated statistics about the loaded trace.

        The returned dictionary contains:

        - ``total_spans`` — total number of user spans
        - ``duration_seconds`` — wall-clock duration derived from ``started_at``
          / ``ended_at`` timestamps embedded in spans, if available; otherwise
          the sum of ``duration_ms`` values converted to seconds
        - ``span_type_counts`` — mapping of span type to occurrence count
        - ``unique_span_types`` — sorted list of distinct span types

        Returns
        -------
        dict
            Summary statistics.

        Raises
        ------
        TracePlayerError
            If :meth:`load` has not been called.
        """
        self._require_loaded()

        type_counter: Counter[str] = Counter(
            str(span.get("span_type", "unknown")) for span in self._spans
        )

        # Derive duration from timestamps when available.
        duration_seconds = self._compute_duration()

        return {
            "total_spans": len(self._spans),
            "duration_seconds": duration_seconds,
            "span_type_counts": dict(type_counter),
            "unique_span_types": sorted(type_counter.keys()),
        }

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def trace_path(self) -> Path:
        """The source file path."""
        return self._trace_path

    @property
    def spans(self) -> list[dict[str, object]]:
        """The loaded spans (empty until :meth:`load` is called)."""
        return list(self._spans)

    @property
    def is_loaded(self) -> bool:
        """``True`` after :meth:`load` has been called successfully."""
        return self._loaded

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise TracePlayerError(
                "Trace has not been loaded. Call load() first."
            )

    def _compute_duration(self) -> float:
        """Estimate wall-clock duration from span data."""
        # Strategy 1: explicit timestamp fields on spans
        timestamps: list[float] = []
        for span in self._spans:
            for key in ("started_at", "timestamp", "start_time"):
                value = span.get(key)
                if isinstance(value, (int, float)):
                    timestamps.append(float(value))
            for key in ("ended_at", "end_time"):
                value = span.get(key)
                if isinstance(value, (int, float)):
                    timestamps.append(float(value))

        if len(timestamps) >= 2:
            return max(timestamps) - min(timestamps)

        # Strategy 2: sum of duration_ms fields
        total_ms: float = sum(
            float(span["duration_ms"])
            for span in self._spans
            if isinstance(span.get("duration_ms"), (int, float))
        )
        return total_ms / 1000.0
