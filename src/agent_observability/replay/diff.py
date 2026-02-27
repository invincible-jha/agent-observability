"""Trace diff — compare two agent execution traces side-by-side.

Identifies spans that were added, removed, or modified between two JSONL
traces, and computes timing deltas for spans that appear in both.

Usage::

    from pathlib import Path
    from agent_observability.replay.diff import TraceDiff, DiffResult

    diff = TraceDiff(Path("/tmp/trace_a.jsonl"), Path("/tmp/trace_b.jsonl"))
    result = diff.compare()

    print(result.added_spans)    # spans only in trace_b
    print(result.removed_spans)  # spans only in trace_a
    print(result.modified_spans) # spans in both but with changed fields
    print(result.timing_deltas)  # duration_ms differences keyed by span id
    print(result.structural_changes) # high-level summary strings
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agent_observability.replay.player import TracePlayer, TracePlayerError

logger = logging.getLogger(__name__)

# Keys used to identify a span across traces.
_IDENTITY_KEYS: tuple[str, ...] = ("span_id", "name", "operation_id")

# Keys considered timing-related.
_TIMING_KEYS: tuple[str, ...] = ("duration_ms", "duration_seconds", "elapsed_ms")

# Keys excluded from content comparison (they change between runs by design).
_EXCLUDE_FROM_DIFF: frozenset[str] = frozenset(
    {"started_at", "ended_at", "timestamp", "start_time", "end_time", "record_kind", "session_id"}
)


@dataclass
class DiffResult:
    """The outcome of comparing two agent traces.

    Attributes
    ----------
    added_spans:
        Spans present in *trace_b* but absent from *trace_a*.
    removed_spans:
        Spans present in *trace_a* but absent from *trace_b*.
    modified_spans:
        Spans present in both traces whose non-timing fields differ.
        Each entry is a dict with keys ``"span_id"``, ``"trace_a"``,
        ``"trace_b"``, and ``"changed_fields"``.
    timing_deltas:
        Mapping of span identity key to ``(trace_a_ms, trace_b_ms, delta_ms)``
        tuples for spans that exist in both traces and carry timing data.
    structural_changes:
        Human-readable summary strings describing high-level differences
        (e.g. span count changes, new span types).
    """

    added_spans: list[dict[str, object]] = field(default_factory=list)
    removed_spans: list[dict[str, object]] = field(default_factory=list)
    modified_spans: list[dict[str, object]] = field(default_factory=list)
    timing_deltas: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    structural_changes: list[str] = field(default_factory=list)

    @property
    def is_identical(self) -> bool:
        """``True`` when both traces are semantically equivalent."""
        return (
            not self.added_spans
            and not self.removed_spans
            and not self.modified_spans
        )

    @property
    def total_changes(self) -> int:
        """Total number of span-level differences."""
        return len(self.added_spans) + len(self.removed_spans) + len(self.modified_spans)


class TraceDiffError(Exception):
    """Raised when the diff operation cannot be completed."""


class TraceDiff:
    """Compares two JSONL agent traces and reports differences.

    Parameters
    ----------
    trace_a:
        Path to the *baseline* JSONL trace file.
    trace_b:
        Path to the *comparison* JSONL trace file.

    Notes
    -----
    Span identity is determined by, in priority order:

    1. A ``span_id`` field
    2. A ``name`` field combined with span index
    3. A ``operation_id`` field
    4. Fall back to positional index

    When neither trace has explicit IDs the diff is purely positional.
    """

    def __init__(self, trace_a: Path, trace_b: Path) -> None:
        self._path_a: Path = trace_a
        self._path_b: Path = trace_b

    # ── Public API ─────────────────────────────────────────────────────────────

    def compare(self) -> DiffResult:
        """Load both traces and compute the full diff.

        Returns
        -------
        DiffResult
            Populated result dataclass.

        Raises
        ------
        TraceDiffError
            If either trace file is missing or contains invalid JSONL.
        """
        spans_a = self._load_trace(self._path_a, label="trace_a")
        spans_b = self._load_trace(self._path_b, label="trace_b")

        index_a = self._index_spans(spans_a)
        index_b = self._index_spans(spans_b)

        keys_a = set(index_a.keys())
        keys_b = set(index_b.keys())

        added_keys = keys_b - keys_a
        removed_keys = keys_a - keys_b
        common_keys = keys_a & keys_b

        added_spans = [index_b[k] for k in sorted(added_keys)]
        removed_spans = [index_a[k] for k in sorted(removed_keys)]

        modified_spans: list[dict[str, object]] = []
        timing_deltas: dict[str, tuple[float, float, float]] = {}

        for key in sorted(common_keys):
            span_a = index_a[key]
            span_b = index_b[key]

            changed_fields = self._diff_fields(span_a, span_b)
            if changed_fields:
                modified_spans.append(
                    {
                        "span_id": key,
                        "trace_a": span_a,
                        "trace_b": span_b,
                        "changed_fields": changed_fields,
                    }
                )

            timing_key = self._timing_key_for(span_a, span_b)
            if timing_key is not None:
                dur_a, dur_b = timing_key
                timing_deltas[key] = (dur_a, dur_b, dur_b - dur_a)

        structural_changes = self._build_structural_summary(
            spans_a=spans_a,
            spans_b=spans_b,
            added_count=len(added_spans),
            removed_count=len(removed_spans),
            modified_count=len(modified_spans),
        )

        return DiffResult(
            added_spans=added_spans,
            removed_spans=removed_spans,
            modified_spans=modified_spans,
            timing_deltas=timing_deltas,
            structural_changes=structural_changes,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def path_a(self) -> Path:
        """Path to the baseline trace."""
        return self._path_a

    @property
    def path_b(self) -> Path:
        """Path to the comparison trace."""
        return self._path_b

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_trace(self, path: Path, label: str) -> list[dict[str, object]]:
        """Load and return spans from *path*."""
        try:
            player = TracePlayer(path)
            return player.load()
        except TracePlayerError as exc:
            raise TraceDiffError(f"Cannot load {label} from {path}: {exc}") from exc

    @staticmethod
    def _span_identity(span: dict[str, object], index: int) -> str:
        """Derive a stable identity string for *span*."""
        for key in _IDENTITY_KEYS:
            value = span.get(key)
            if value is not None:
                return f"{key}:{value}"
        # Fall back to positional identity including type for a better key.
        span_type = span.get("span_type", "span")
        return f"pos:{index}:{span_type}"

    @classmethod
    def _index_spans(
        cls, spans: list[dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        """Build an identity-keyed mapping from *spans*."""
        index: dict[str, dict[str, object]] = {}
        for position, span in enumerate(spans):
            key = cls._span_identity(span, position)
            # Deduplicate: if a key already exists append a counter suffix.
            original_key = key
            counter = 1
            while key in index:
                key = f"{original_key}#{counter}"
                counter += 1
            index[key] = span
        return index

    @staticmethod
    def _diff_fields(
        span_a: dict[str, object],
        span_b: dict[str, object],
    ) -> list[str]:
        """Return field names that differ between *span_a* and *span_b*.

        Timing-only fields and excluded keys are not reported here (they are
        captured in ``timing_deltas`` instead).
        """
        all_keys = (set(span_a.keys()) | set(span_b.keys())) - _EXCLUDE_FROM_DIFF
        timing_keys = set(_TIMING_KEYS)
        comparison_keys = all_keys - timing_keys

        changed: list[str] = []
        for key in sorted(comparison_keys):
            val_a = span_a.get(key)
            val_b = span_b.get(key)
            if val_a != val_b:
                changed.append(key)
        return changed

    @staticmethod
    def _timing_key_for(
        span_a: dict[str, object],
        span_b: dict[str, object],
    ) -> Optional[tuple[float, float]]:
        """Extract ``(duration_a_ms, duration_b_ms)`` if both spans have timing data."""
        for key in _TIMING_KEYS:
            val_a = span_a.get(key)
            val_b = span_b.get(key)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                return float(val_a), float(val_b)
        return None

    @staticmethod
    def _build_structural_summary(
        spans_a: list[dict[str, object]],
        spans_b: list[dict[str, object]],
        added_count: int,
        removed_count: int,
        modified_count: int,
    ) -> list[str]:
        """Generate human-readable structural change notes."""
        notes: list[str] = []

        count_a = len(spans_a)
        count_b = len(spans_b)
        if count_a != count_b:
            notes.append(
                f"Span count changed: {count_a} → {count_b} "
                f"({count_b - count_a:+d})"
            )

        if added_count:
            notes.append(f"{added_count} span(s) added in trace_b")
        if removed_count:
            notes.append(f"{removed_count} span(s) removed from trace_a")
        if modified_count:
            notes.append(f"{modified_count} span(s) have modified fields")

        # Detect new span types introduced in trace_b.
        types_a = {str(s.get("span_type", "")) for s in spans_a}
        types_b = {str(s.get("span_type", "")) for s in spans_b}
        new_types = (types_b - types_a) - {""}
        dropped_types = (types_a - types_b) - {""}
        if new_types:
            notes.append(f"New span type(s) in trace_b: {', '.join(sorted(new_types))}")
        if dropped_types:
            notes.append(f"Span type(s) dropped from trace_a: {', '.join(sorted(dropped_types))}")

        if not notes:
            notes.append("Traces are structurally identical.")

        return notes
