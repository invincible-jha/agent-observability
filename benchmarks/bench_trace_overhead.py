"""Benchmark: Latency overhead of AgentTracer span creation vs no tracing.

Measures the wall-clock overhead introduced by creating and finishing an
AgentSpan relative to the equivalent no-op baseline. This is the primary
concern raised against tracing libraries.

Competitor context
------------------
- Langfuse issue #9429 (2024-09): users reported 2-4x latency overhead for
  synchronous SDK calls in tight loops.
  Source: https://github.com/langfuse/langfuse/issues/9429
- Langfuse issue #9766 (2024-10): follow-up confirming batch export reduces
  overhead to <5% in async usage.
  Source: https://github.com/langfuse/langfuse/issues/9766
- OpenLLMetry: no published overhead numbers; community reports <1ms per span.

agent-observability uses a NoOp fallback when OTel SDK is absent, making
the span lifecycle overhead pure Python object creation overhead.
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

from agent_observability.spans.types import AgentTracer
from datasets.workload_generator import ActionKind, generate_workload


def _no_op_operation(index: int) -> str:
    """Simulate a trivially fast operation without any tracing overhead.

    Returns a string to prevent dead-code elimination by the interpreter.
    """
    return f"result_{index}"


def _traced_operation(tracer: AgentTracer, index: int) -> str:
    """The same trivial operation wrapped in an AgentTracer span."""
    with tracer.llm_call(model="gpt-4o-mini") as span:
        result = f"result_{index}"
        span.set_tokens(50, 100)
        span.set_cost(0.000015)
    return result


def run_benchmark(
    n_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Measure the latency overhead ratio of traced vs untraced operations.

    Parameters
    ----------
    n_iterations:
        Number of individual operations per condition.
    seed:
        Unused here (reserved for future parametric workloads).

    Returns
    -------
    dict with overhead metrics.
    """
    tracer = AgentTracer(
        tracer_name="bench-tracer",
        agent_id="bench-agent",
        session_id="bench-session",
    )

    # Warmup — 100 iterations of each to prime interpreter paths
    for w in range(100):
        _no_op_operation(w)
    for w in range(100):
        _traced_operation(tracer, w)

    # Baseline: untraced
    baseline_lats: list[float] = []
    for index in range(n_iterations):
        start = time.perf_counter()
        _no_op_operation(index)
        baseline_lats.append((time.perf_counter() - start) * 1_000_000)  # microseconds

    # Traced: same operation inside an AgentTracer span
    traced_lats: list[float] = []
    for index in range(n_iterations):
        start = time.perf_counter()
        _traced_operation(tracer, index)
        traced_lats.append((time.perf_counter() - start) * 1_000_000)

    def _percentile(data: list[float], pct: float) -> float:
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct)
        idx = min(idx, len(sorted_data) - 1)
        return round(sorted_data[idx], 3)

    baseline_mean = statistics.mean(baseline_lats)
    traced_mean = statistics.mean(traced_lats)
    overhead_pct = (
        round((traced_mean - baseline_mean) / baseline_mean * 100, 2)
        if baseline_mean > 0
        else 0.0
    )

    return {
        "benchmark": "trace_overhead",
        "n_iterations": n_iterations,
        "baseline_us": {
            "p50": _percentile(baseline_lats, 0.50),
            "p95": _percentile(baseline_lats, 0.95),
            "p99": _percentile(baseline_lats, 0.99),
            "mean": round(baseline_mean, 3),
        },
        "traced_us": {
            "p50": _percentile(traced_lats, 0.50),
            "p95": _percentile(traced_lats, 0.95),
            "p99": _percentile(traced_lats, 0.99),
            "mean": round(traced_mean, 3),
        },
        "overhead_percent": overhead_pct,
        "note": (
            "Langfuse sync SDK showed 2-4x overhead (issues #9429, #9766). "
            "agent-observability uses NoOp spans when OTel SDK absent, "
            "so overhead is pure Python object lifecycle cost."
        ),
    }


if __name__ == "__main__":
    print("Running trace overhead benchmark (n=2000 iterations each)...")
    result = run_benchmark(n_iterations=2000)
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
