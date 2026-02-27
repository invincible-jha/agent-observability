"""Benchmark: Memory overhead of the AgentTracer span infrastructure.

Measures the incremental memory used per span and per 1K spans created
using the tracemalloc stdlib module (no external dependencies).

Competitor context
------------------
Langfuse SDK's in-memory queue can grow unbounded if the export thread
stalls. This was reported as a concern in GitHub #9766 (2024-10).
agent-observability spans are lightweight Python objects — this benchmark
quantifies that cost.
"""
from __future__ import annotations

import gc
import json
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_observability.spans.types import AgentTracer


def _measure_span_memory(n_spans: int) -> int:
    """Return net bytes allocated by creating and finishing n_spans.

    Uses the public llm_call() context manager so the measurement is
    representative of real usage and avoids passing invalid sentinel
    values to _start_span.

    Parameters
    ----------
    n_spans:
        Number of spans to create in a batch.

    Returns
    -------
    int
        Net bytes allocated during the creation of n_spans.
    """
    tracer = AgentTracer(
        tracer_name="bench-mem-tracer",
        agent_id="bench-agent",
        session_id="bench-session",
    )

    gc.collect()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for index in range(n_spans):
        # Use the public context manager — representative of real usage.
        with tracer.llm_call(model="gpt-4o", name=f"bench.llm.{index}") as span:
            span.set_tokens(100, 200)
            span.set_cost(0.00003)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    net_bytes = sum(stat.size_diff for stat in top_stats)
    return net_bytes


def run_benchmark(
    span_counts: list[int] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Measure memory overhead for different numbers of spans.

    Parameters
    ----------
    span_counts:
        List of span counts to measure. Defaults to [100, 1000, 10000].
    seed:
        Unused (reserved for consistency).

    Returns
    -------
    dict with bytes_per_span and total_bytes per span count.
    """
    if span_counts is None:
        span_counts = [100, 1_000, 10_000]

    size_results: dict[str, dict[str, object]] = {}
    for n_spans in span_counts:
        net_bytes = _measure_span_memory(n_spans)
        bytes_per_span = round(net_bytes / n_spans, 2) if n_spans > 0 else 0.0
        size_results[str(n_spans)] = {
            "net_bytes_total": net_bytes,
            "bytes_per_span": bytes_per_span,
            "kb_total": round(net_bytes / 1024, 2),
        }

    return {
        "benchmark": "memory_overhead",
        "span_counts": span_counts,
        "seed": seed,
        "results_by_span_count": size_results,
        "note": (
            "Measured via tracemalloc (stdlib). Net bytes may be negative when GC "
            "collects other objects during the measurement window. "
            "Langfuse #9766 noted unbounded queue growth under stalled exporters; "
            "agent-observability uses a no-op span lifecycle in the NoOp path."
        ),
    }


if __name__ == "__main__":
    print("Running memory overhead benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "memory_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
