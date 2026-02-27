"""Comparison visualiser for agent-observability benchmark results."""
from __future__ import annotations

import json
from pathlib import Path


COMPETITOR_NOTES = {
    "trace_overhead": (
        "Langfuse sync SDK: 2-4x overhead reported in issues #9429 and #9766 (2024-09/10). "
        "agent-observability NoOp spans: overhead is pure Python object lifecycle."
    ),
    "throughput": (
        "Langfuse batch export: ~5K events/sec (community reports). "
        "OpenLLMetry no-op exporter: ~10K-50K spans/sec (community reports). "
        "This benchmark measures in-process span lifecycle only."
    ),
    "memory": (
        "Langfuse #9766: unbounded queue growth reported under stalled exporters. "
        "agent-observability: NoOp path has no export queue."
    ),
}


def _fmt_table(rows: list[tuple[str, str]], title: str) -> None:
    col1_width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    for key, value in rows:
        print(f"  {key:<{col1_width}} {value}")


def _load(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[return-value]


def display_overhead(data: dict[str, object]) -> None:
    baseline = data.get("baseline_us", {})
    traced = data.get("traced_us", {})
    rows: list[tuple[str, str]] = [
        ("Baseline p50 (us)", str(baseline.get("p50"))),
        ("Traced p50 (us)", str(traced.get("p50"))),
        ("Baseline p99 (us)", str(baseline.get("p99"))),
        ("Traced p99 (us)", str(traced.get("p99"))),
        ("Overhead %", str(data.get("overhead_percent")) + "%"),
    ]
    _fmt_table(rows, "Trace Overhead Results")
    print(f"\n  Competitor: {COMPETITOR_NOTES['trace_overhead']}")


def display_throughput(data: dict[str, object]) -> None:
    size_results = data.get("results_by_batch_size", {})
    rows: list[tuple[str, str]] = []
    for batch_size, stats in size_results.items():
        rows.append(
            (f"Batch {batch_size} — spans/sec", str(stats.get("spans_per_second")))
        )
        rows.append(
            (f"Batch {batch_size} — mean elapsed (ms)", str(stats.get("mean_batch_elapsed_ms")))
        )
    _fmt_table(rows, "Span Throughput Results")
    print(f"\n  Competitor: {COMPETITOR_NOTES['throughput']}")


def display_memory(data: dict[str, object]) -> None:
    size_results = data.get("results_by_span_count", {})
    rows: list[tuple[str, str]] = []
    for n_spans, stats in size_results.items():
        rows.append(
            (f"{n_spans} spans — bytes/span", str(stats.get("bytes_per_span")))
        )
        rows.append(
            (f"{n_spans} spans — KB total", str(stats.get("kb_total")))
        )
    _fmt_table(rows, "Memory Overhead Results")
    print(f"\n  Competitor: {COMPETITOR_NOTES['memory']}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    for fname, display_fn, label in [
        ("baseline.json", display_overhead, "trace overhead"),
        ("throughput_baseline.json", display_throughput, "throughput"),
        ("memory_baseline.json", display_memory, "memory"),
    ]:
        data = _load(results_dir / fname)
        if data:
            display_fn(data)  # type: ignore[arg-type]
        else:
            print(f"No {fname} found. Run bench_{label.replace(' ', '_')}.py first.")

    print("\n" + "=" * 65)
    print("  Run all benchmarks:")
    print("    python benchmarks/bench_trace_overhead.py")
    print("    python benchmarks/bench_throughput.py")
    print("    python benchmarks/bench_memory_usage.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
