# Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Quickstart](01_quickstart.py) | Minimal working example with the Tracer convenience class |
| 02 | [Decision Tracing](02_decision_tracing.py) | Record and query agent decision points |
| 03 | [Cost Tracking](03_cost_tracking.py) | Hierarchical cost attribution across agents |
| 04 | [Prompt Registry](04_prompt_registry.py) | Versioned prompt management with diffs |
| 05 | [Trace Correlation](05_trace_correlation.py) | Link multi-agent spans into a unified trace tree |
| 06 | [LangChain Tracing](06_langchain_tracing.py) | Trace LangChain chains with cost attribution |
| 07 | [OTel Export](07_otel_export.py) | Record, replay, and diff agent traces |

## Running the examples

```bash
pip install agent-observability
python examples/01_quickstart.py
```

For framework integrations:

```bash
pip install langchain langchain-openai   # for example 06
```
