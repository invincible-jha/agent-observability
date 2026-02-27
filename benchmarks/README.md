# agent-observability Benchmarks

Reproducible benchmark suite measuring tracing overhead, span throughput, and memory usage.

## Quick Start

```bash
cd repos/agent-observability
python benchmarks/bench_trace_overhead.py
python benchmarks/bench_throughput.py
python benchmarks/bench_memory_usage.py
python benchmarks/compare.py
```

## Benchmarks

### bench_trace_overhead.py

**What it measures:** Wall-clock overhead percentage of traced operations vs untraced baseline.

**Method:**
2,000 iterations of a trivial operation are run without tracing (baseline) and then with
an `AgentTracer.llm_call()` span wrapping the same operation. Latency is measured in
microseconds. The overhead percentage is `(traced_mean - baseline_mean) / baseline_mean * 100`.

**Key metrics:**
- `overhead_percent` — the key competitive metric
- `baseline_us.p50`, `traced_us.p50` — median latency for each condition

**Competitor reference:**
- Langfuse sync SDK: 2-4x overhead (GitHub #9429, #9766, Sep/Oct 2024)
- Langfuse batch async path: <5% overhead reported in #9766
- Source: https://github.com/langfuse/langfuse/issues/9429

---

### bench_throughput.py

**What it measures:** Spans created and finished per second at batch sizes 1, 10, 100, 1000.

**Method:**
For each batch size, 20 timing runs are performed. Each run creates `batch_size` spans
sequentially. Total spans/sec = `batch_size * n_batches / total_elapsed`.

**Key metrics:**
- `spans_per_second` per batch size

**Note:** No OTel exporter is configured. This measures pure in-process span lifecycle cost.

---

### bench_memory_usage.py

**What it measures:** Memory allocated per span at 100, 1K, and 10K span counts.

**Method:**
Uses Python's `tracemalloc` stdlib module to snapshot heap before and after span creation.
Reports net bytes and bytes-per-span.

**Key metrics:**
- `bytes_per_span` — marginal heap cost of one AgentSpan

**Competitor reference:**
- Langfuse #9766: unbounded queue growth under stalled export threads
- Source: https://github.com/langfuse/langfuse/issues/9766

## Interpreting Results

- All benchmarks use the `AgentTracer` NoOp backend (OTel SDK not required).
- Results are deterministic: no external calls, fixed synthetic workloads.
- Run on the same hardware for fair comparisons across versions.

## Competitor Numbers (public only)

| Competitor | Metric | Value | Source |
|------------|--------|-------|--------|
| Langfuse sync SDK | Latency overhead | 2-4x | GitHub #9429, #9766 (2024) |
| Langfuse async batch | Latency overhead | <5% | GitHub #9766 (2024) |
| OpenLLMetry no-op | Spans/sec | 10K-50K | Community reports |
