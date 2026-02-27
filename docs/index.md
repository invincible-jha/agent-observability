# agent-observability

OpenTelemetry-Native Agent Tracing — 8 semantic span types, cost attribution, drift detection.

[![CI](https://github.com/invincible-jha/agent-observability/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-observability/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/aumos-agent-observability.svg)](https://pypi.org/project/aumos-agent-observability/)
[![Python versions](https://img.shields.io/pypi/pyversions/aumos-agent-observability.svg)](https://pypi.org/project/aumos-agent-observability/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Installation

```bash
pip install aumos-agent-observability
```

Verify the installation:

```bash
agent-observability version
```

---

## Quick Start

```python
from agent_observability import AgentTracer, instrument

# Auto-instrument all supported frameworks with one call
instrument()  # Covers LangChain, CrewAI, AutoGen, Anthropic SDK, OpenAI SDK, MCP

# Or use the AgentTracer context manager directly
tracer = AgentTracer(service_name="my-agent")

with tracer.llm_call(model="gpt-4o", prompt="Summarize this document") as span:
    # Your LLM call here
    response = call_llm(prompt="Summarize this document")
    span.set_tokens(input_tokens=512, output_tokens=128)
    span.set_cost(input_cost=0.0025, output_cost=0.0010)

with tracer.tool_invoke(tool_name="web_search") as span:
    results = web_search("quarterly earnings report")
    span.set_result_count(len(results))

with tracer.reasoning_step(step_name="plan") as span:
    plan = agent.plan(results)
    span.set_reasoning_tokens(256)

# Drift detection
from agent_observability import BaselineProfiler, DriftDetector

profiler = BaselineProfiler(agent_id="agent-alpha")
profiler.record_observation(tool_calls=3, prompt_length=512, error_rate=0.0)

detector = DriftDetector(profiler)
alerts = detector.check(tool_calls=12, prompt_length=4096, error_rate=0.25)
for alert in alerts:
    print(f"Drift detected: {alert.metric} deviated by {alert.deviation:.1f}x")
```

---

## Key Features

- **Eight agent-semantic OTel span kinds** — `llm_call`, `tool_invoke`, `memory_read`, `memory_write`, `reasoning_step`, `agent_delegate`, `human_approval`, and `agent_error` — each with typed fluent setters for tokens, cost, model, and domain-specific fields
- **`AgentTracer` context-manager factory** — produces `AgentSpan` instances backed by a real OTel tracer when the SDK is installed, or a zero-dependency no-op fallback when it is not
- **Per-call cost attribution** — model pricing tables and `CostAnnotation` dataclasses recorded directly on LLM call spans
- **Behavioral drift detection** — `BaselineProfiler` accumulates metrics, and `DriftDetector` raises alerts when observations deviate beyond configurable thresholds
- **PII-safe telemetry** — configurable `Redactor` that scrubs span attributes before they are exported
- **Auto-instrumentation** — for LangChain, CrewAI, AutoGen, Anthropic SDK, OpenAI SDK, and MCP — call `instrument()` once and all framework calls emit spans automatically
- **Pre-built Grafana dashboards** — for agent throughput, cost trends, error rates, and drift alerts

---

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-observability)
- [PyPI Package](https://pypi.org/project/aumos-agent-observability/)
- [Architecture](architecture.md)
- [Migration from Langfuse](migrate-from-langfuse.md)
- [Changelog](https://github.com/invincible-jha/agent-observability/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/invincible-jha/agent-observability/blob/main/CONTRIBUTING.md)

---

> Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.
