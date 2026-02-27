"""Comprehensive tests for the Live Agent Replay Debugger.

Covers:
- TraceRecorder  — write, flush, session lifecycle, error conditions
- TracePlayer    — load, step, filter, play_all, summary, error conditions
- TraceDiff      — identical traces, added/removed/modified spans, timing changes
- DiffResult     — dataclass properties, is_identical, total_changes
- CLI commands   — replay record, replay play, replay diff
"""
from __future__ import annotations

import json
import pathlib
import uuid
from typing import Iterator

import pytest
from click.testing import CliRunner

from agent_observability.cli.main import cli
from agent_observability.replay.diff import DiffResult, TraceDiff, TraceDiffError
from agent_observability.replay.player import TracePlayer, TracePlayerError
from agent_observability.replay.recorder import TraceRecorder, TraceRecorderError


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_span(
    span_type: str = "llm_call",
    duration_ms: float = 100.0,
    **extra: object,
) -> dict[str, object]:
    """Build a minimal span dictionary."""
    return {"span_type": span_type, "duration_ms": duration_ms, **extra}


def _write_jsonl(path: pathlib.Path, spans: list[dict[str, object]]) -> None:
    """Write raw JSONL without a recorder (for player/diff tests)."""
    path.write_text(
        "\n".join(json.dumps(s) for s in spans) + "\n",
        encoding="utf-8",
    )


def _full_trace(
    path: pathlib.Path,
    spans: list[dict[str, object]],
    session_id: str = "test-session",
) -> None:
    """Record a complete trace (header + spans + footer) via TraceRecorder."""
    recorder = TraceRecorder(path)
    recorder.start_session(session_id)
    for span in spans:
        recorder.record_span(span)
    recorder.end_session()


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def trace_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "trace.jsonl"


@pytest.fixture()
def trace_path_a(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "trace_a.jsonl"


@pytest.fixture()
def trace_path_b(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "trace_b.jsonl"


@pytest.fixture()
def recorder(trace_path: pathlib.Path) -> TraceRecorder:
    return TraceRecorder(trace_path)


@pytest.fixture()
def started_recorder(recorder: TraceRecorder) -> TraceRecorder:
    recorder.start_session("fixture-session")
    return recorder


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ═══════════════════════════════════════════════════════════════════════════════
# TraceRecorder tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceRecorderInit:
    def test_output_path_stored(self, trace_path: pathlib.Path) -> None:
        recorder = TraceRecorder(trace_path)
        assert recorder.output_path == trace_path

    def test_no_active_session_on_init(self, trace_path: pathlib.Path) -> None:
        recorder = TraceRecorder(trace_path)
        assert recorder.active_session_id is None

    def test_span_count_zero_on_init(self, trace_path: pathlib.Path) -> None:
        recorder = TraceRecorder(trace_path)
        assert recorder.span_count == 0

    def test_creates_parent_directories(self, tmp_path: pathlib.Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "trace.jsonl"
        recorder = TraceRecorder(nested)
        assert nested.parent.exists()
        # File itself not created until session starts
        assert not nested.exists()
        recorder.start_session("s1")
        recorder.end_session()
        assert nested.exists()


class TestTraceRecorderStartSession:
    def test_sets_active_session_id(self, recorder: TraceRecorder) -> None:
        recorder.start_session("my-session")
        assert recorder.active_session_id == "my-session"
        recorder.end_session()

    def test_creates_file(self, recorder: TraceRecorder, trace_path: pathlib.Path) -> None:
        recorder.start_session("s1")
        assert trace_path.exists()
        recorder.end_session()

    def test_writes_header_line(self, recorder: TraceRecorder, trace_path: pathlib.Path) -> None:
        recorder.start_session("s1")
        recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        header = lines[0]
        assert header["record_kind"] == "session_header"
        assert header["session_id"] == "s1"

    def test_header_contains_started_at(
        self, recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        recorder.start_session("s1")
        recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        assert "started_at" in lines[0]

    def test_raises_if_session_already_active(self, started_recorder: TraceRecorder) -> None:
        with pytest.raises(TraceRecorderError, match="already active"):
            started_recorder.start_session("another")
        started_recorder.end_session()

    def test_allows_second_session_after_end(self, recorder: TraceRecorder) -> None:
        recorder.start_session("first")
        recorder.end_session()
        recorder.start_session("second")
        assert recorder.active_session_id == "second"
        recorder.end_session()

    def test_resets_span_count_between_sessions(self, recorder: TraceRecorder) -> None:
        recorder.start_session("first")
        recorder.record_span(_make_span())
        recorder.record_span(_make_span())
        recorder.end_session()
        recorder.start_session("second")
        assert recorder.span_count == 0
        recorder.end_session()


class TestTraceRecorderRecordSpan:
    def test_increments_span_count(self, started_recorder: TraceRecorder) -> None:
        started_recorder.record_span(_make_span())
        assert started_recorder.span_count == 1
        started_recorder.record_span(_make_span())
        assert started_recorder.span_count == 2
        started_recorder.end_session()

    def test_span_written_to_file(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span({"span_type": "tool_invoke", "duration_ms": 50})
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        span_records = [l for l in lines if l.get("record_kind") == "span"]
        assert len(span_records) == 1
        assert span_records[0]["span_type"] == "tool_invoke"

    def test_session_id_injected_into_span(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span({"span_type": "llm_call"})
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        span_record = next(l for l in lines if l.get("record_kind") == "span")
        assert span_record["session_id"] == "fixture-session"

    def test_raises_without_active_session(self, recorder: TraceRecorder) -> None:
        with pytest.raises(TraceRecorderError, match="No active session"):
            recorder.record_span(_make_span())

    def test_multiple_spans_preserved_in_order(
        self, recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        recorder.start_session("order-test")
        for i in range(5):
            recorder.record_span({"span_type": "step", "index": i})
        recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        span_records = [l for l in lines if l.get("record_kind") == "span"]
        indices = [l["index"] for l in span_records]
        assert indices == list(range(5))

    def test_span_with_extra_fields_preserved(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span(
            {"span_type": "llm_call", "model": "gpt-4", "tokens": 500, "cost_usd": 0.01}
        )
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        span_record = next(l for l in lines if l.get("record_kind") == "span")
        assert span_record["model"] == "gpt-4"
        assert span_record["tokens"] == 500

    def test_non_serialisable_values_coerced_to_string(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        import datetime

        started_recorder.record_span({"ts": datetime.datetime(2024, 1, 1)})
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        span_record = next(l for l in lines if l.get("record_kind") == "span")
        assert isinstance(span_record["ts"], str)

    def test_empty_span_dict_recorded(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span({})
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        span_records = [l for l in lines if l.get("record_kind") == "span"]
        assert len(span_records) == 1


class TestTraceRecorderEndSession:
    def test_clears_active_session(self, started_recorder: TraceRecorder) -> None:
        started_recorder.end_session()
        assert started_recorder.active_session_id is None

    def test_resets_span_count(self, started_recorder: TraceRecorder) -> None:
        started_recorder.record_span(_make_span())
        started_recorder.end_session()
        assert started_recorder.span_count == 0

    def test_writes_footer_record(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span(_make_span())
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        footer = lines[-1]
        assert footer["record_kind"] == "session_footer"

    def test_footer_contains_total_spans(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span(_make_span())
        started_recorder.record_span(_make_span())
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        footer = lines[-1]
        assert footer["total_spans"] == 2

    def test_footer_contains_duration_seconds(
        self, recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        recorder.start_session("dur-test")
        recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        footer = lines[-1]
        assert "duration_seconds" in footer
        assert isinstance(footer["duration_seconds"], float)

    def test_footer_contains_span_type_counts(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span({"span_type": "llm_call"})
        started_recorder.record_span({"span_type": "llm_call"})
        started_recorder.record_span({"span_type": "tool_invoke"})
        started_recorder.end_session()
        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        footer = lines[-1]
        counts = footer["span_type_counts"]
        assert counts["llm_call"] == 2
        assert counts["tool_invoke"] == 1

    def test_raises_without_active_session(self, recorder: TraceRecorder) -> None:
        with pytest.raises(TraceRecorderError, match="No active session to end"):
            recorder.end_session()

    def test_file_closed_after_end(self, started_recorder: TraceRecorder) -> None:
        started_recorder.end_session()
        # File handle should be closed; recorder should allow new session
        started_recorder.start_session("new")
        assert started_recorder.active_session_id == "new"
        started_recorder.end_session()


class TestTraceRecorderFlush:
    def test_flush_with_open_handle(self, started_recorder: TraceRecorder) -> None:
        started_recorder.record_span(_make_span())
        started_recorder.flush()  # Should not raise
        started_recorder.end_session()

    def test_flush_without_open_handle_is_noop(self, recorder: TraceRecorder) -> None:
        recorder.flush()  # No session open — should not raise

    def test_data_readable_after_flush(
        self, started_recorder: TraceRecorder, trace_path: pathlib.Path
    ) -> None:
        started_recorder.record_span({"span_type": "test_span"})
        started_recorder.flush()
        # File should be readable mid-session after flush
        content = trace_path.read_text()
        assert "test_span" in content
        started_recorder.end_session()


class TestTraceRecorderAppendBehaviour:
    def test_multiple_sessions_appended(
        self, trace_path: pathlib.Path
    ) -> None:
        recorder = TraceRecorder(trace_path)
        recorder.start_session("s1")
        recorder.record_span({"span_type": "a"})
        recorder.end_session()

        recorder.start_session("s2")
        recorder.record_span({"span_type": "b"})
        recorder.end_session()

        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        headers = [l for l in lines if l.get("record_kind") == "session_header"]
        assert len(headers) == 2

    def test_span_types_tracked_correctly_across_types(
        self, trace_path: pathlib.Path
    ) -> None:
        recorder = TraceRecorder(trace_path)
        recorder.start_session("type-track")
        for span_type in ["a", "b", "a", "c", "b", "b"]:
            recorder.record_span({"span_type": span_type})
        recorder.end_session()

        lines = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        footer = next(l for l in lines if l.get("record_kind") == "session_footer")
        counts = footer["span_type_counts"]
        assert counts["a"] == 2
        assert counts["b"] == 3
        assert counts["c"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TracePlayer tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTracePlayerInit:
    def test_trace_path_stored(self, trace_path: pathlib.Path) -> None:
        player = TracePlayer(trace_path)
        assert player.trace_path == trace_path

    def test_not_loaded_on_init(self, trace_path: pathlib.Path) -> None:
        player = TracePlayer(trace_path)
        assert not player.is_loaded

    def test_spans_empty_before_load(self, trace_path: pathlib.Path) -> None:
        player = TracePlayer(trace_path)
        assert player.spans == []


class TestTracePlayerLoad:
    def test_load_returns_list(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        result = player.load()
        assert isinstance(result, list)

    def test_load_returns_correct_span_count(self, trace_path: pathlib.Path) -> None:
        spans = [_make_span("llm_call"), _make_span("tool_invoke"), _make_span("memory_read")]
        _full_trace(trace_path, spans)
        player = TracePlayer(trace_path)
        result = player.load()
        assert len(result) == 3

    def test_load_excludes_session_header(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        result = player.load()
        for record in result:
            assert record.get("record_kind") != "session_header"

    def test_load_excludes_session_footer(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        result = player.load()
        for record in result:
            assert record.get("record_kind") != "session_footer"

    def test_load_includes_user_span_data(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [{"span_type": "llm_call", "model": "gpt-4", "duration_ms": 200}])
        player = TracePlayer(trace_path)
        result = player.load()
        assert result[0]["span_type"] == "llm_call"
        assert result[0]["model"] == "gpt-4"

    def test_load_sets_is_loaded(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        player.load()
        assert player.is_loaded

    def test_load_raises_for_missing_file(self, tmp_path: pathlib.Path) -> None:
        player = TracePlayer(tmp_path / "nonexistent.jsonl")
        with pytest.raises(TracePlayerError, match="not found"):
            player.load()

    def test_load_raises_for_invalid_json(self, trace_path: pathlib.Path) -> None:
        trace_path.write_text("not valid json\n", encoding="utf-8")
        player = TracePlayer(trace_path)
        with pytest.raises(TracePlayerError, match="Invalid JSON"):
            player.load()

    def test_load_handles_blank_lines(self, trace_path: pathlib.Path) -> None:
        trace_path.write_text(
            '\n{"span_type":"llm_call"}\n\n{"span_type":"tool_invoke"}\n',
            encoding="utf-8",
        )
        player = TracePlayer(trace_path)
        result = player.load()
        assert len(result) == 2

    def test_load_raw_jsonl_without_recorder(self, trace_path: pathlib.Path) -> None:
        """Player should handle JSONL files that lack recorder metadata."""
        _write_jsonl(trace_path, [_make_span("llm_call"), _make_span("tool_invoke")])
        player = TracePlayer(trace_path)
        result = player.load()
        assert len(result) == 2

    def test_load_idempotent(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        result1 = player.load()
        result2 = player.load()
        assert result1 == result2

    def test_load_empty_trace(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [])
        player = TracePlayer(trace_path)
        result = player.load()
        assert result == []


class TestTracePlayerStep:
    @pytest.fixture(autouse=True)
    def _setup(self, trace_path: pathlib.Path) -> None:
        spans = [_make_span("llm_call", index=i) for i in range(3)]
        _full_trace(trace_path, spans)
        self._player = TracePlayer(trace_path)
        self._player.load()

    def test_step_returns_first_span(self) -> None:
        span = self._player.step(0)
        assert span["span_type"] == "llm_call"
        assert span["index"] == 0

    def test_step_returns_last_span(self) -> None:
        span = self._player.step(2)
        assert span["index"] == 2

    def test_step_raises_on_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            self._player.step(99)

    def test_step_raises_negative_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            self._player.step(-99)

    def test_step_negative_index_works(self) -> None:
        # Python supports negative indexing
        span = self._player.step(-1)
        assert span["index"] == 2

    def test_step_raises_if_not_loaded(self, trace_path: pathlib.Path) -> None:
        fresh = TracePlayer(trace_path)
        with pytest.raises(TracePlayerError, match="load"):
            fresh.step(0)


class TestTracePlayerPlayAll:
    def test_play_all_returns_iterator(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        player.load()
        assert hasattr(player.play_all(), "__iter__")

    def test_play_all_yields_all_spans(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span() for _ in range(5)])
        player = TracePlayer(trace_path)
        player.load()
        count = sum(1 for _ in player.play_all())
        assert count == 5

    def test_play_all_preserves_order(self, trace_path: pathlib.Path) -> None:
        spans = [{"span_type": "step", "order": i} for i in range(10)]
        _full_trace(trace_path, spans)
        player = TracePlayer(trace_path)
        player.load()
        orders = [span["order"] for span in player.play_all()]
        assert orders == list(range(10))

    def test_play_all_raises_if_not_loaded(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        with pytest.raises(TracePlayerError, match="load"):
            list(player.play_all())

    def test_play_all_iterable_twice(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span() for _ in range(3)])
        player = TracePlayer(trace_path)
        player.load()
        first = list(player.play_all())
        second = list(player.play_all())
        assert first == second


class TestTracePlayerFilterByType:
    @pytest.fixture(autouse=True)
    def _setup(self, trace_path: pathlib.Path) -> None:
        spans = [
            _make_span("llm_call"),
            _make_span("tool_invoke"),
            _make_span("llm_call"),
            _make_span("memory_read"),
        ]
        _full_trace(trace_path, spans)
        self._player = TracePlayer(trace_path)
        self._player.load()

    def test_filter_returns_matching_spans(self) -> None:
        result = self._player.filter_by_type("llm_call")
        assert len(result) == 2
        assert all(s["span_type"] == "llm_call" for s in result)

    def test_filter_returns_single_match(self) -> None:
        result = self._player.filter_by_type("tool_invoke")
        assert len(result) == 1

    def test_filter_returns_empty_for_no_match(self) -> None:
        result = self._player.filter_by_type("nonexistent_type")
        assert result == []

    def test_filter_does_not_mutate_internal_state(self) -> None:
        before = len(self._player.spans)
        self._player.filter_by_type("llm_call")
        assert len(self._player.spans) == before

    def test_filter_raises_if_not_loaded(self, trace_path: pathlib.Path) -> None:
        fresh = TracePlayer(trace_path)
        with pytest.raises(TracePlayerError, match="load"):
            fresh.filter_by_type("llm_call")


class TestTracePlayerSummary:
    def test_summary_contains_total_spans(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span() for _ in range(7)])
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        assert summary["total_spans"] == 7

    def test_summary_contains_span_type_counts(self, trace_path: pathlib.Path) -> None:
        spans = [_make_span("llm_call"), _make_span("llm_call"), _make_span("tool_invoke")]
        _full_trace(trace_path, spans)
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        counts = summary["span_type_counts"]
        assert counts["llm_call"] == 2
        assert counts["tool_invoke"] == 1

    def test_summary_contains_unique_span_types(self, trace_path: pathlib.Path) -> None:
        spans = [_make_span("llm_call"), _make_span("tool_invoke"), _make_span("llm_call")]
        _full_trace(trace_path, spans)
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        assert sorted(summary["unique_span_types"]) == ["llm_call", "tool_invoke"]

    def test_summary_duration_from_duration_ms(self, trace_path: pathlib.Path) -> None:
        spans = [_make_span(duration_ms=200), _make_span(duration_ms=300)]
        _full_trace(trace_path, spans)
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        # Duration derived from sum of duration_ms / 1000 when no timestamps
        assert summary["duration_seconds"] == pytest.approx(0.5, abs=0.001)

    def test_summary_duration_from_timestamps(self, trace_path: pathlib.Path) -> None:
        spans = [
            {"span_type": "step", "started_at": 1000.0},
            {"span_type": "step", "ended_at": 1005.5},
        ]
        _write_jsonl(trace_path, spans)
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        assert summary["duration_seconds"] == pytest.approx(5.5, abs=0.001)

    def test_summary_empty_trace(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [])
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        assert summary["total_spans"] == 0
        assert summary["span_type_counts"] == {}
        assert summary["unique_span_types"] == []

    def test_summary_raises_if_not_loaded(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        with pytest.raises(TracePlayerError, match="load"):
            player.summary()

    def test_summary_contains_duration_seconds_key(self, trace_path: pathlib.Path) -> None:
        _full_trace(trace_path, [_make_span()])
        player = TracePlayer(trace_path)
        player.load()
        summary = player.summary()
        assert "duration_seconds" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# DiffResult tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiffResult:
    def test_is_identical_true_when_no_changes(self) -> None:
        result = DiffResult()
        assert result.is_identical is True

    def test_is_identical_false_when_added(self) -> None:
        result = DiffResult(added_spans=[_make_span()])
        assert result.is_identical is False

    def test_is_identical_false_when_removed(self) -> None:
        result = DiffResult(removed_spans=[_make_span()])
        assert result.is_identical is False

    def test_is_identical_false_when_modified(self) -> None:
        result = DiffResult(
            modified_spans=[{"span_id": "s1", "changed_fields": ["model"]}]
        )
        assert result.is_identical is False

    def test_total_changes_zero_when_identical(self) -> None:
        result = DiffResult()
        assert result.total_changes == 0

    def test_total_changes_counts_all_categories(self) -> None:
        result = DiffResult(
            added_spans=[_make_span(), _make_span()],
            removed_spans=[_make_span()],
            modified_spans=[{"span_id": "x"}],
        )
        assert result.total_changes == 4

    def test_timing_deltas_ignored_in_is_identical(self) -> None:
        result = DiffResult(timing_deltas={"span1": (100.0, 200.0, 100.0)})
        assert result.is_identical is True

    def test_structural_changes_ignored_in_is_identical(self) -> None:
        result = DiffResult(structural_changes=["Span count changed: 3 → 4 (+1)"])
        assert result.is_identical is True

    def test_default_fields_are_empty(self) -> None:
        result = DiffResult()
        assert result.added_spans == []
        assert result.removed_spans == []
        assert result.modified_spans == []
        assert result.timing_deltas == {}
        assert result.structural_changes == []


# ═══════════════════════════════════════════════════════════════════════════════
# TraceDiff tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceDiffIdenticalTraces:
    def test_identical_traces_is_identical(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        spans = [_make_span("llm_call", span_id="s1"), _make_span("tool_invoke", span_id="s2")]
        _full_trace(trace_path_a, spans)
        _full_trace(trace_path_b, spans)
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.is_identical

    def test_identical_traces_no_added_spans(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        spans = [_make_span(span_id="s1")]
        _full_trace(trace_path_a, spans)
        _full_trace(trace_path_b, spans)
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.added_spans == []

    def test_identical_traces_no_removed_spans(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        spans = [_make_span(span_id="s1")]
        _full_trace(trace_path_a, spans)
        _full_trace(trace_path_b, spans)
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.removed_spans == []

    def test_identical_traces_structural_says_identical(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        spans = [_make_span(span_id="s1")]
        _full_trace(trace_path_a, spans)
        _full_trace(trace_path_b, spans)
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert any("identical" in note.lower() for note in result.structural_changes)

    def test_empty_traces_are_identical(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [])
        _full_trace(trace_path_b, [])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.is_identical


class TestTraceDiffAddedSpans:
    def test_detects_added_span(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [_make_span("llm_call", span_id="s1")])
        _full_trace(
            trace_path_b,
            [_make_span("llm_call", span_id="s1"), _make_span("tool_invoke", span_id="s2")],
        )
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.added_spans) == 1
        assert result.added_spans[0]["span_type"] == "tool_invoke"

    def test_added_span_not_in_removed(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [])
        _full_trace(trace_path_b, [_make_span(span_id="new")])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.added_spans) == 1
        assert result.removed_spans == []

    def test_structural_note_for_added(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [])
        _full_trace(trace_path_b, [_make_span(span_id="x")])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert any("added" in note.lower() for note in result.structural_changes)

    def test_multiple_added_spans(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [])
        new_spans = [_make_span(span_id=f"s{i}") for i in range(4)]
        _full_trace(trace_path_b, new_spans)
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.added_spans) == 4


class TestTraceDiffRemovedSpans:
    def test_detects_removed_span(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(
            trace_path_a,
            [_make_span("llm_call", span_id="s1"), _make_span("tool_invoke", span_id="s2")],
        )
        _full_trace(trace_path_b, [_make_span("llm_call", span_id="s1")])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.removed_spans) == 1
        assert result.removed_spans[0]["span_type"] == "tool_invoke"

    def test_structural_note_for_removed(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [_make_span(span_id="x")])
        _full_trace(trace_path_b, [])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert any("removed" in note.lower() for note in result.structural_changes)

    def test_multiple_removed_spans(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        old_spans = [_make_span(span_id=f"s{i}") for i in range(5)]
        _full_trace(trace_path_a, old_spans)
        _full_trace(trace_path_b, [])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.removed_spans) == 5


class TestTraceDiffModifiedSpans:
    def test_detects_modified_field(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"span_type": "llm_call", "span_id": "s1", "model": "gpt-3"}])
        _full_trace(trace_path_b, [{"span_type": "llm_call", "span_id": "s1", "model": "gpt-4"}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.modified_spans) == 1
        assert "model" in result.modified_spans[0]["changed_fields"]

    def test_unchanged_span_not_in_modified(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        spans = [{"span_type": "llm_call", "span_id": "s1", "model": "gpt-4"}]
        _full_trace(trace_path_a, spans)
        _full_trace(trace_path_b, spans)
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.modified_spans == []

    def test_modified_span_contains_both_versions(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"span_type": "llm_call", "span_id": "s1", "model": "v1"}])
        _full_trace(trace_path_b, [{"span_type": "llm_call", "span_id": "s1", "model": "v2"}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        entry = result.modified_spans[0]
        assert entry["trace_a"]["model"] == "v1"
        assert entry["trace_b"]["model"] == "v2"

    def test_modified_entry_has_span_id(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"span_id": "unique-id", "status": "ok"}])
        _full_trace(trace_path_b, [{"span_id": "unique-id", "status": "error"}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.modified_spans[0]["span_id"] == "span_id:unique-id"


class TestTraceDiffTimingDeltas:
    def test_timing_delta_computed(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"span_id": "s1", "duration_ms": 100.0}])
        _full_trace(trace_path_b, [{"span_id": "s1", "duration_ms": 150.0}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert "span_id:s1" in result.timing_deltas
        dur_a, dur_b, delta = result.timing_deltas["span_id:s1"]
        assert dur_a == pytest.approx(100.0)
        assert dur_b == pytest.approx(150.0)
        assert delta == pytest.approx(50.0)

    def test_timing_delta_negative_when_faster(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"span_id": "s1", "duration_ms": 200.0}])
        _full_trace(trace_path_b, [{"span_id": "s1", "duration_ms": 100.0}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        _, _, delta = result.timing_deltas["span_id:s1"]
        assert delta == pytest.approx(-100.0)

    def test_no_timing_delta_when_no_timing_data(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"span_id": "s1", "span_type": "step"}])
        _full_trace(trace_path_b, [{"span_id": "s1", "span_type": "step"}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.timing_deltas == {}

    def test_timing_not_counted_in_modified_spans(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        """Timing-only changes should NOT appear as modified spans."""
        _full_trace(trace_path_a, [{"span_id": "s1", "span_type": "step", "duration_ms": 100.0}])
        _full_trace(trace_path_b, [{"span_id": "s1", "span_type": "step", "duration_ms": 200.0}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert result.modified_spans == []


class TestTraceDiffStructuralChanges:
    def test_new_span_type_reported(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [_make_span("llm_call", span_id="s1")])
        _full_trace(
            trace_path_b,
            [_make_span("llm_call", span_id="s1"), _make_span("reasoning_step", span_id="s2")],
        )
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert any("reasoning_step" in note for note in result.structural_changes)

    def test_dropped_span_type_reported(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(
            trace_path_a,
            [_make_span("llm_call", span_id="s1"), _make_span("tool_invoke", span_id="s2")],
        )
        _full_trace(trace_path_b, [_make_span("llm_call", span_id="s1")])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert any("tool_invoke" in note for note in result.structural_changes)

    def test_span_count_change_reported(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [_make_span(span_id=f"s{i}") for i in range(3)])
        _full_trace(trace_path_b, [_make_span(span_id=f"s{i}") for i in range(5)])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert any("3" in note and "5" in note for note in result.structural_changes)


class TestTraceDiffErrors:
    def test_raises_for_missing_trace_a(
        self, tmp_path: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_b, [_make_span()])
        differ = TraceDiff(tmp_path / "missing.jsonl", trace_path_b)
        with pytest.raises(TraceDiffError):
            differ.compare()

    def test_raises_for_missing_trace_b(
        self, trace_path_a: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [_make_span()])
        differ = TraceDiff(trace_path_a, tmp_path / "missing.jsonl")
        with pytest.raises(TraceDiffError):
            differ.compare()

    def test_path_properties_accessible(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        differ = TraceDiff(trace_path_a, trace_path_b)
        assert differ.path_a == trace_path_a
        assert differ.path_b == trace_path_b


class TestTraceDiffPositionalFallback:
    def test_diff_by_position_when_no_ids(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        """Without span_id/name the diff should fall back to positional matching."""
        _full_trace(trace_path_a, [{"span_type": "step", "value": 1}])
        _full_trace(trace_path_b, [{"span_type": "step", "value": 2}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.modified_spans) == 1
        assert "value" in result.modified_spans[0]["changed_fields"]

    def test_diff_by_name_field(
        self, trace_path_a: pathlib.Path, trace_path_b: pathlib.Path
    ) -> None:
        _full_trace(trace_path_a, [{"name": "fetch-data", "status": "ok"}])
        _full_trace(trace_path_b, [{"name": "fetch-data", "status": "error"}])
        result = TraceDiff(trace_path_a, trace_path_b).compare()
        assert len(result.modified_spans) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — replay record command
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIReplayRecord:
    def test_record_writes_trace_file(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "trace.jsonl"
        result = runner.invoke(
            cli,
            ["replay", "record", "--output", str(out), '{"span_type":"llm_call"}'],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_record_file_contains_span(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "trace.jsonl"
        runner.invoke(
            cli, ["replay", "record", "--output", str(out), '{"span_type":"tool_invoke"}']
        )
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        span_lines = [l for l in lines if l.get("record_kind") == "span"]
        assert span_lines[0]["span_type"] == "tool_invoke"

    def test_record_session_id_option(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "trace.jsonl"
        runner.invoke(
            cli,
            [
                "replay",
                "record",
                "--output",
                str(out),
                "--session-id",
                "my-session",
                '{"span_type":"step"}',
            ],
        )
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        header = next(l for l in lines if l.get("record_kind") == "session_header")
        assert header["session_id"] == "my-session"

    def test_record_without_span_writes_empty_session(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        out = tmp_path / "trace.jsonl"
        result = runner.invoke(cli, ["replay", "record", "--output", str(out)])
        assert result.exit_code == 0, result.output
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        headers = [l for l in lines if l.get("record_kind") == "session_header"]
        assert len(headers) == 1

    def test_record_output_flag_required(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "record", '{"span_type":"step"}'])
        assert result.exit_code != 0

    def test_record_invalid_json_exits_nonzero(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        out = tmp_path / "trace.jsonl"
        result = runner.invoke(cli, ["replay", "record", "--output", str(out), "not_json"])
        assert result.exit_code != 0

    def test_record_short_flag(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "trace.jsonl"
        result = runner.invoke(cli, ["replay", "record", "-o", str(out), '{"span_type":"step"}'])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_record_output_mentions_trace_path(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        out = tmp_path / "trace.jsonl"
        result = runner.invoke(cli, ["replay", "record", "-o", str(out), '{"span_type":"step"}'])
        assert str(out) in result.output

    def test_record_help_text_present(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "record", "--help"])
        assert result.exit_code == 0
        assert "JSONL" in result.output or "trace" in result.output.lower()

    def test_record_registered_in_replay_group(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "--help"])
        assert "record" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — replay play command
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIReplayPlay:
    @pytest.fixture()
    def trace_file(self, tmp_path: pathlib.Path) -> pathlib.Path:
        path = tmp_path / "trace.jsonl"
        spans = [
            {"span_type": "llm_call", "span_id": "s1", "duration_ms": 100},
            {"span_type": "tool_invoke", "span_id": "s2", "duration_ms": 50},
            {"span_type": "llm_call", "span_id": "s3", "duration_ms": 200},
        ]
        _full_trace(path, spans)
        return path

    def test_play_prints_all_spans(
        self, runner: CliRunner, trace_file: pathlib.Path
    ) -> None:
        result = runner.invoke(cli, ["replay", "play", str(trace_file)])
        assert result.exit_code == 0, result.output
        assert "llm_call" in result.output
        assert "tool_invoke" in result.output

    def test_play_filter_type(self, runner: CliRunner, trace_file: pathlib.Path) -> None:
        result = runner.invoke(
            cli, ["replay", "play", str(trace_file), "--filter-type", "llm_call"]
        )
        assert result.exit_code == 0, result.output
        assert "llm_call" in result.output
        assert "tool_invoke" not in result.output

    def test_play_filter_type_no_match(
        self, runner: CliRunner, trace_file: pathlib.Path
    ) -> None:
        result = runner.invoke(
            cli, ["replay", "play", str(trace_file), "--filter-type", "nonexistent"]
        )
        assert result.exit_code == 0
        assert "No spans matched" in result.output

    def test_play_step_option(self, runner: CliRunner, trace_file: pathlib.Path) -> None:
        result = runner.invoke(cli, ["replay", "play", str(trace_file), "--step", "0"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output.strip())
        assert data["span_type"] == "llm_call"

    def test_play_step_out_of_range(
        self, runner: CliRunner, trace_file: pathlib.Path
    ) -> None:
        result = runner.invoke(cli, ["replay", "play", str(trace_file), "--step", "999"])
        assert result.exit_code != 0

    def test_play_summary_flag(self, runner: CliRunner, trace_file: pathlib.Path) -> None:
        result = runner.invoke(cli, ["replay", "play", str(trace_file), "--summary"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output.strip())
        assert data["total_spans"] == 3

    def test_play_missing_file_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "play", "/nonexistent/trace.jsonl"])
        assert result.exit_code != 0

    def test_play_output_is_valid_json_per_span(
        self, runner: CliRunner, trace_file: pathlib.Path
    ) -> None:
        result = runner.invoke(cli, ["replay", "play", str(trace_file)])
        # Each span block should be parseable JSON.
        # Split on empty line boundaries for blocks
        output_lines = result.output.strip().split("\n")
        buffer: list[str] = []
        parsed_count = 0
        for line in output_lines:
            buffer.append(line)
            combined = "\n".join(buffer)
            try:
                json.loads(combined)
                parsed_count += 1
                buffer = []
            except json.JSONDecodeError:
                pass
        assert parsed_count >= 3

    def test_play_help_text_present(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "play", "--help"])
        assert result.exit_code == 0

    def test_play_registered_in_replay_group(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "--help"])
        assert "play" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — replay diff command
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIReplayDiff:
    @pytest.fixture()
    def trace_a(self, tmp_path: pathlib.Path) -> pathlib.Path:
        path = tmp_path / "a.jsonl"
        _full_trace(path, [_make_span("llm_call", span_id="s1")])
        return path

    @pytest.fixture()
    def trace_b_same(self, tmp_path: pathlib.Path) -> pathlib.Path:
        path = tmp_path / "b_same.jsonl"
        _full_trace(path, [_make_span("llm_call", span_id="s1")])
        return path

    @pytest.fixture()
    def trace_b_diff(self, tmp_path: pathlib.Path) -> pathlib.Path:
        path = tmp_path / "b_diff.jsonl"
        _full_trace(
            path,
            [
                _make_span("llm_call", span_id="s1"),
                _make_span("tool_invoke", span_id="s2"),
            ],
        )
        return path

    def test_diff_identical_exits_zero(
        self, runner: CliRunner, trace_a: pathlib.Path, trace_b_same: pathlib.Path
    ) -> None:
        result = runner.invoke(cli, ["replay", "diff", str(trace_a), str(trace_b_same)])
        assert result.exit_code == 0, result.output

    def test_diff_changed_exits_nonzero(
        self, runner: CliRunner, trace_a: pathlib.Path, trace_b_diff: pathlib.Path
    ) -> None:
        result = runner.invoke(cli, ["replay", "diff", str(trace_a), str(trace_b_diff)])
        assert result.exit_code == 1

    def test_diff_prints_structural_changes(
        self, runner: CliRunner, trace_a: pathlib.Path, trace_b_diff: pathlib.Path
    ) -> None:
        result = runner.invoke(cli, ["replay", "diff", str(trace_a), str(trace_b_diff)])
        assert "added" in result.output.lower() or "span" in result.output.lower()

    def test_diff_json_output_flag(
        self, runner: CliRunner, trace_a: pathlib.Path, trace_b_diff: pathlib.Path
    ) -> None:
        result = runner.invoke(
            cli, ["replay", "diff", str(trace_a), str(trace_b_diff), "--json-output"]
        )
        data = json.loads(result.output.strip())
        assert "is_identical" in data
        assert "added_spans" in data

    def test_diff_json_output_identical_traces(
        self, runner: CliRunner, trace_a: pathlib.Path, trace_b_same: pathlib.Path
    ) -> None:
        result = runner.invoke(
            cli, ["replay", "diff", str(trace_a), str(trace_b_same), "--json-output"]
        )
        data = json.loads(result.output.strip())
        assert data["is_identical"] is True

    def test_diff_missing_trace_a_exits_2(
        self, runner: CliRunner, tmp_path: pathlib.Path, trace_b_same: pathlib.Path
    ) -> None:
        result = runner.invoke(
            cli, ["replay", "diff", str(tmp_path / "nope.jsonl"), str(trace_b_same)]
        )
        assert result.exit_code != 0

    def test_diff_help_text_present(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "diff", "--help"])
        assert result.exit_code == 0

    def test_diff_registered_in_replay_group(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "--help"])
        assert "diff" in result.output

    def test_diff_timing_deltas_in_json_output(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        path_a = tmp_path / "ta.jsonl"
        path_b = tmp_path / "tb.jsonl"
        _full_trace(path_a, [{"span_id": "s1", "duration_ms": 100.0}])
        _full_trace(path_b, [{"span_id": "s1", "duration_ms": 200.0}])
        result = runner.invoke(
            cli, ["replay", "diff", str(path_a), str(path_b), "--json-output"]
        )
        data = json.loads(result.output.strip())
        assert data["timing_deltas"]  # at least one entry


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — replay group registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIReplayGroup:
    def test_replay_group_in_top_level_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert "replay" in result.output

    def test_replay_group_help_lists_commands(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay", "--help"])
        assert result.exit_code == 0
        assert "record" in result.output
        assert "play" in result.output
        assert "diff" in result.output

    def test_replay_group_no_args_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["replay"])
        # Click group with no subcommand shows help and exits 0
        assert "record" in result.output or result.exit_code == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration — round-trip record → play
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundTrip:
    def test_recorded_spans_playable(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "round_trip.jsonl"
        recorder = TraceRecorder(path)
        recorder.start_session("rt-session")
        for i in range(10):
            recorder.record_span({"span_type": "step", "index": i, "duration_ms": i * 10.0})
        recorder.end_session()

        player = TracePlayer(path)
        spans = player.load()
        assert len(spans) == 10
        for i, span in enumerate(spans):
            assert span["index"] == i

    def test_player_summary_matches_recorder(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "rt2.jsonl"
        recorder = TraceRecorder(path)
        recorder.start_session("rt2")
        for span_type in ["llm_call", "tool_invoke", "llm_call"]:
            recorder.record_span({"span_type": span_type, "duration_ms": 100.0})
        recorder.end_session()

        player = TracePlayer(path)
        player.load()
        summary = player.summary()
        assert summary["total_spans"] == 3
        assert summary["span_type_counts"]["llm_call"] == 2
        assert summary["span_type_counts"]["tool_invoke"] == 1

    def test_diff_of_same_recorded_trace_is_identical(self, tmp_path: pathlib.Path) -> None:
        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"
        spans = [_make_span("llm_call", span_id=f"s{i}") for i in range(5)]
        _full_trace(path_a, spans)
        _full_trace(path_b, spans)
        result = TraceDiff(path_a, path_b).compare()
        assert result.is_identical

    def test_public_api_imports(self) -> None:
        from agent_observability.replay import (
            DiffResult,
            TraceDiff,
            TracePlayer,
            TraceRecorder,
        )

        assert TraceRecorder is not None
        assert TracePlayer is not None
        assert TraceDiff is not None
        assert DiffResult is not None
