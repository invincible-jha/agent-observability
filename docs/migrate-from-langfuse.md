# Migrating from Langfuse to agent-observability

agent-observability is an OpenTelemetry-native tracing library built specifically for multi-agent
systems. If you are using Langfuse for LLM tracing today, this guide explains how to migrate fully
or how to run both tools side-by-side using the built-in LangChain adapter bridge.

---

## Feature Comparison

| Capability | Langfuse | agent-observability |
|---|---|---|
| Trace LLM calls | Yes | Yes |
| Prompt management | Yes (hosted UI) | Yes — `PromptRegistry` with versioning, diffing, and rendering (in-memory or custom backend) |
| Human evaluation / scoring | Yes | No |
| OpenTelemetry native | Partial (OTLP export only) | Full — spans are first-class OTel |
| Agent-semantic span kinds | No — generic spans | 8 kinds: `llm_call`, `tool_invoke`, `memory_read`, `memory_write`, `reasoning_step`, `agent_delegate`, `human_approval`, `agent_error` |
| Delegation tracing | No | Yes — `agent_delegate` span with target agent ID |
| Reasoning step tracing | No | Yes — `reasoning_step` with confidence and strategy |
| Per-span cost attribution | No | Yes — USD cost and token counts on each span |
| Behavioral drift detection | No | Yes — Z-score analysis against a baseline |
| PII redaction pipeline | No | Yes — configurable redactor on export |
| Async / zero-overhead mode | Optional | Default — spans never block the hot path |
| Self-hosted required | No (cloud or self-hosted) | No — works with any OTLP backend or in-process |
| Replay debugger | No | Yes — re-execute a recorded trace offline |
| LangChain adapter | Yes (first-party) | Yes — `LangChainTracer` adapter |

---

## Installation

```bash
# Full install with OpenTelemetry and LangChain adapter
pip install "agent-observability[otel,langchain]"

# Minimal install for LangChain-free code
pip install agent-observability
```

---

## Step 1 — Replace Langfuse Initialization

**Before (Langfuse):**

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)

@observe()
def run_agent(prompt: str) -> str:
    response = call_llm(prompt)
    return response
```

**After (agent-observability):**

```python
from agent_observability import Tracer

tracer = Tracer(service_name="my-agent", agent_id="agent-001")

def run_agent(prompt: str) -> str:
    with tracer.trace_llm(model="claude-sonnet-4", provider="anthropic") as span:
        response = call_llm(prompt)
        span.set_tokens(input_tokens=len(prompt.split()), output_tokens=50)
        span.set_cost(cost_usd=0.0004)
    return response
```

No API keys. No hosted service. Spans go to any OTLP-compatible backend (Jaeger, Grafana Tempo,
Honeycomb, Datadog) or stay in-process for testing.

---

## Step 2 — Replace Generic Spans with Agent-Semantic Spans

Langfuse treats everything as a generic observation. agent-observability has dedicated span kinds
for each agent action, which means your observability platform can filter, alert, and dashboard
on agent behavior without custom parsing.

**Before (Langfuse — tool call as observation):**

```python
from langfuse.decorators import observe

@observe(name="tool-web-search")
def web_search(query: str) -> str:
    return search(query)
```

**After (agent-observability — typed tool span):**

```python
from agent_observability.spans.types import AgentTracer

tracer = AgentTracer(agent_id="agent-001", framework="custom")

def web_search(query: str) -> str:
    with tracer.tool_invoke("web_search") as span:
        result = search(query)
        span.set_tool("web_search", success=True)
    return result
```

**Delegation between agents:**

```python
with tracer.agent_delegate(target_agent="summarizer-agent") as span:
    span.set_delegation(
        target_agent="summarizer-agent",
        task_id="task-789",
        strategy="capability-match",
    )
    result = delegate_to_summarizer(content)
```

**Reasoning step with confidence:**

```python
with tracer.reasoning_step(step_index=2, step_type="hypothesis") as span:
    span.set_reasoning(
        step_index=2,
        step_type="hypothesis",
        confidence=0.87,
        strategy="chain-of-thought",
    )
    conclusion = derive_hypothesis(evidence)
```

---

## Step 3 — Add Cost Attribution Per Span

Langfuse reports aggregate cost per trace from the LLM provider response. agent-observability
attributes cost to each individual span, so you can see which sub-step of a multi-agent workflow
is responsible for the most spend.

```python
from agent_observability.spans.types import AgentTracer, CostAnnotation

tracer = AgentTracer(agent_id="agent-001")

with tracer.llm_call(model="gpt-4o", provider="openai") as span:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    span.set_tokens(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
    )
    # Cost is computed and stored on the span — no separate aggregation needed
    span.set_cost(cost_usd=usage.prompt_tokens * 2.5e-6 + usage.completion_tokens * 10e-6)
```

---

## Step 4 — Enable Behavioral Drift Detection

This has no equivalent in Langfuse. Drift detection compares recent agent behavior against a
recorded baseline using Z-score analysis across behavioral feature vectors.

```python
from agent_observability.drift.detector import DriftDetector
from agent_observability.drift.baseline import AgentBaseline
from agent_observability.drift.features import SpanRecord

# Build a baseline from a known-good set of span records
baseline = AgentBaseline.from_span_records(
    agent_id="agent-001",
    spans=known_good_spans,  # list[SpanRecord]
)

detector = DriftDetector(sigma_threshold=3.0, min_window_spans=10)

# Check current window against baseline
result = detector.check(baseline=baseline, window=recent_spans)

if result.drifted:
    print(f"Drift detected! severity={result.severity}, max_z={result.max_z_score:.2f}")
    print(f"Drifted features: {list(result.drifted_features.keys())}")
```

---

## Step 5 — Configure PII-Safe Telemetry Export

```python
from agent_observability.pii.redactor import PIIRedactor
from agent_observability import Tracer

# Redactor strips patterns (email, phone, SSN) before spans leave the process
redactor = PIIRedactor(patterns=["email", "phone", "ssn"])
tracer = Tracer(service_name="my-agent", agent_id="agent-001")

# Export with PII stripped
raw_spans = tracer.export()
clean_spans = [redactor.redact_span(span) for span in raw_spans]
```

---

## Coexistence: Enriching Langfuse Traces with Agent Spans

If you want to keep Langfuse for its hosted UI and human evaluation workflows while adding
agent-semantic spans alongside, use the `LangChainTracer` adapter as a bridge.

```bash
pip install "agent-observability[langchain]"
```

```python
from agent_observability.adapters.langchain import LangChainTracer
from agent_observability.spans.types import AgentTracer

# The LangChainTracer adapter emits OTel span dicts for each LangChain lifecycle event.
# Feed those dicts to your OTLP exporter while Langfuse continues to receive its own events.

lc_adapter = LangChainTracer()
agent_tracer = AgentTracer(agent_id="agent-001", framework="langchain")

# On LLM start — emit both a Langfuse observation AND an OTel span
def on_llm_start(model: str, prompt: str) -> None:
    span_data = lc_adapter.on_llm_start(model=model, prompt=prompt)
    # span_data is a serializable dict — forward to your OTLP exporter
    forward_to_otel_exporter(span_data)
    # Langfuse continues to receive events via its own callback handler
```

This pattern lets you evaluate agent-observability incrementally without a flag day: Langfuse
handles scoring and the hosted UI; agent-observability adds delegation tracing, drift detection,
and per-span cost attribution in parallel.

---

## What You Gain by Switching

1. **Agent-semantic spans** — 8 typed span kinds give your observability platform structured data
   about decisions, delegations, and reasoning steps without custom parsing.
2. **Per-span cost attribution** — every span carries USD cost and token counts, so you can
   identify exactly which sub-step is driving your spend.
3. **Behavioral drift detection** — Z-score analysis catches when an agent's behavior deviates
   from baseline without requiring manual alert rules.
4. **PII-safe telemetry** — a configurable redaction pipeline strips sensitive data before spans
   leave the process, which is useful for regulated industries.
5. **No hosted dependency** — agent-observability works with any OTLP backend or purely in-process.
   You own all trace data.
6. **Replay debugger** — recorded span traces can be re-executed offline for root cause analysis.

## What You Keep

- All your existing OTLP backends (Jaeger, Grafana, Datadog, Honeycomb) work without changes —
  agent-observability emits standard OTel spans.
- LangChain callbacks and chain structure are preserved — the `LangChainTracer` adapter translates
  lifecycle events without wrapping your chains.
- Langfuse prompt management and human evaluation workflows are independent of tracing and remain
  available if you run both tools in coexistence mode.
