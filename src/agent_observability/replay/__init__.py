"""Live Agent Replay Debugger.

Provides tools for recording, replaying, and diffing agent execution traces
stored in JSONL format (one JSON object per line).

Public API::

    from agent_observability.replay import TraceRecorder, TracePlayer, TraceDiff, DiffResult

Example::

    from pathlib import Path
    from agent_observability.replay import TraceRecorder, TracePlayer, TraceDiff

    # Record
    recorder = TraceRecorder(Path("/tmp/trace.jsonl"))
    recorder.start_session("session-abc")
    recorder.record_span({"span_type": "llm_call", "duration_ms": 120})
    recorder.end_session()

    # Play back
    player = TracePlayer(Path("/tmp/trace.jsonl"))
    player.load()
    for span in player.play_all():
        print(span)

    # Diff two traces
    diff = TraceDiff(Path("/tmp/trace_a.jsonl"), Path("/tmp/trace_b.jsonl"))
    result = diff.compare()
"""
from __future__ import annotations

from agent_observability.replay.diff import DiffResult, TraceDiff
from agent_observability.replay.player import TracePlayer
from agent_observability.replay.recorder import TraceRecorder

__all__ = [
    "DiffResult",
    "TraceDiff",
    "TracePlayer",
    "TraceRecorder",
]
