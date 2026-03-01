#!/usr/bin/env python3
"""Example: OpenTelemetry Export

Demonstrates how to export agent traces to an OpenTelemetry-compatible
backend using the replay and trace recording subsystems.

Usage:
    python examples/07_otel_export.py

Requirements:
    pip install agent-observability
"""
from __future__ import annotations

import agent_observability
from agent_observability import (
    Tracer,
    TraceRecorder,
    TracePlayer,
    TraceDiff,
    DiffResult,
)


def build_trace(tracer: Tracer, recorder: TraceRecorder) -> str:
    """Build and record a sample trace."""
    recorder.start()
    with tracer.span("pipeline.run", attributes={"pipeline": "data-ingestion"}):
        with tracer.span("pipeline.fetch", attributes={"source": "postgres"}):
            pass  # simulate fetch
        with tracer.span("pipeline.transform", attributes={"rows": 1000}):
            pass  # simulate transform
        with tracer.span("pipeline.load", attributes={"target": "data-warehouse"}):
            pass  # simulate load
    trace_id = recorder.stop()
    return trace_id


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    # Step 1: Create tracer and recorder
    tracer = Tracer(service_name="data-pipeline")
    recorder = TraceRecorder(tracer=tracer)

    # Step 2: Record a baseline trace
    print("Recording baseline trace...")
    baseline_trace_id = build_trace(tracer, recorder)
    print(f"  Baseline trace: {baseline_trace_id}")

    # Step 3: Replay the trace (useful for debugging)
    player = TracePlayer(recorder=recorder)
    try:
        replay_result = player.replay(trace_id=baseline_trace_id)
        print(f"\nReplay complete:")
        print(f"  Spans replayed: {replay_result.span_count}")
        print(f"  Duration: {replay_result.duration_ms:.1f}ms")
    except Exception as error:
        print(f"Replay error: {error}")

    # Step 4: Record a second trace and diff them
    print("\nRecording second trace for diff...")
    second_trace_id = build_trace(tracer, recorder)
    print(f"  Second trace: {second_trace_id}")

    try:
        differ = TraceDiff(recorder=recorder)
        diff_result: DiffResult = differ.diff(
            trace_id_a=baseline_trace_id,
            trace_id_b=second_trace_id,
        )
        print(f"\nTrace diff result:")
        print(f"  Spans added: {diff_result.spans_added}")
        print(f"  Spans removed: {diff_result.spans_removed}")
        print(f"  Spans changed: {diff_result.spans_changed}")
        print(f"  Structurally identical: {diff_result.is_identical}")
    except Exception as error:
        print(f"Diff error: {error}")

    # Step 5: Print trace summary
    summary = tracer.summary()
    print(f"\nFinal trace summary:")
    print(f"  Total spans recorded: {summary.total_spans}")
    print(f"  Unique trace IDs: {recorder.trace_count()}")


if __name__ == "__main__":
    main()
