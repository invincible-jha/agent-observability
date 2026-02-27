"""Benchmark: Span creation throughput (spans/second) at different batch sizes.

Measures how many AgentTracer spans can be created and finished per second
under different workload batch sizes.

Competitor context
------------------
OpenLLMetry community reports suggest ~10K-50K spans/sec for no-op exporters.
Langfuse's Python SDK with batch export can handle ~5K trace events/sec.
This benchmark measures agent-observability's span lifecycle rate only
(no exporter is configured in benchmarks — pure in-process cost).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_observability.spans.types import AgentTracer, AgentSpanKind
from datasets.workload_generator import ActionKind, generate_workload, workload_batches


def _run_batch(
    tracer: AgentTracer,
    batch_size: int,
    n_batches: int,
) -> list[float]:
    """Run n_batches of batch_size spans, return per-batch wall-clock times.

    Parameters
    ----------
    tracer:
        Configured AgentTracer instance.
    batch_size:
        Number of spans per batch.
    n_batches:
        Number of batches to run.

    Returns
    -------
    list of elapsed seconds per batch.
    """
    elapsed_times: list[float] = []
    for _ in range(n_batches):
        start = time.perf_counter()
        for span_index in range(batch_size):
            with tracer.llm_call(model="gpt-4o-mini") as span:
                span.set_tokens(64, 128)
                span.set_cost(0.00003)
        elapsed_times.append(time.perf_counter() - start)
    return elapsed_times


def run_benchmark(
    batch_sizes: list[int] | None = None,
    n_batches: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    """Measure span throughput across different batch sizes.

    Parameters
    ----------
    batch_sizes:
        List of batch sizes to test. Defaults to [1, 10, 100, 1000].
    n_batches:
        Number of timing runs per batch size.
    seed:
        Unused (reserved for future parametric workloads).

    Returns
    -------
    dict with spans/sec per batch size.
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 1000]

    tracer = AgentTracer(
        tracer_name="bench-throughput-tracer",
        agent_id="bench-agent",
        session_id="bench-session",
    )

    # Warmup
    for _ in range(50):
        with tracer.llm_call(model="gpt-4o-mini") as span:
            span.set_tokens(10, 20)

    size_results: dict[str, dict[str, object]] = {}
    for batch_size in batch_sizes:
        elapsed_times = _run_batch(tracer, batch_size, n_batches)
        spans_per_second_per_run = [
            batch_size / elapsed for elapsed in elapsed_times if elapsed > 0
        ]
        total_spans = batch_size * n_batches
        total_elapsed = sum(elapsed_times)
        overall_throughput = total_spans / total_elapsed if total_elapsed > 0 else 0.0

        size_results[str(batch_size)] = {
            "spans_per_second": round(overall_throughput, 1),
            "mean_batch_elapsed_ms": round(statistics.mean(elapsed_times) * 1000, 3),
            "p50_batch_elapsed_ms": round(
                sorted(elapsed_times)[int(n_batches * 0.50)] * 1000, 3
            ),
            "p95_batch_elapsed_ms": round(
                sorted(elapsed_times)[min(int(n_batches * 0.95), n_batches - 1)] * 1000, 3
            ),
        }

    return {
        "benchmark": "span_throughput",
        "batch_sizes": batch_sizes,
        "n_batches_per_size": n_batches,
        "results_by_batch_size": size_results,
        "note": (
            "No exporter configured — this measures pure in-process span lifecycle cost. "
            "OTel SDK BatchSpanProcessor adds queue overhead separately."
        ),
    }


if __name__ == "__main__":
    print("Running span throughput benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "throughput_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
