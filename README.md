# agent-observability

OpenTelemetry-native agent tracing, cost attribution, and drift detection

[![CI](https://github.com/aumos-ai/agent-observability/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-observability/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-observability.svg)](https://pypi.org/project/agent-observability/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-observability.svg)](https://pypi.org/project/agent-observability/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- Eight agent-semantic OTel span kinds — `llm_call`, `tool_invoke`, `memory_read`, `memory_write`, `reasoning_step`, `agent_delegate`, `human_approval`, and `agent_error` — each with typed fluent setters for tokens, cost, model, and domain-specific fields
- `AgentTracer` context-manager factory produces `AgentSpan` instances backed by a real OTel tracer when the SDK is installed, or a zero-dependency no-op fallback when it is not
- Per-call cost attribution with model pricing tables and `CostAnnotation` dataclasses recorded directly on LLM call spans
- Behavioral drift detection: `BaselineProfiler` accumulates metrics (tool call frequency, prompt length, error rate), and `DriftDetector` raises alerts when observations deviate beyond configurable thresholds
- PII-safe telemetry via a configurable `Redactor` that scrubs span attributes before they are exported
- Auto-instrumentation modules for LangChain, CrewAI, AutoGen, Anthropic SDK, OpenAI SDK, and MCP — call `instrument()` once and all framework calls emit spans automatically
- Pre-built Grafana dashboard definitions for agent throughput, cost trends, error rates, and drift alerts

## Current Limitations

> **Transparency note**: We list known limitations to help you evaluate fit.

- **Storage**: In-memory trace storage only — no persistent backend. Traces lost on restart.
- **Export**: OTLP export only. No direct Jaeger/Zipkin/Datadog export.
- **UI**: No built-in web dashboard. Requires external tools (Grafana, Jaeger UI) for visualization.

## Quick Start

Install from PyPI:

```bash
pip install agent-observability
```

Verify the installation:

```bash
agent-observability version
```

Basic usage:

```python
import agent_observability

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

For production deployments requiring SLA-backed support and advanced
integrations, contact the maintainers or see the commercial extensions documentation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
