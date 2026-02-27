"""Example: Using agent-observability with Langfuse.

Install with:
    pip install "aumos-agent-observability[langfuse]"

This example demonstrates bridging AumOS agent traces to Langfuse.
The adapter maps agent-semantic span kinds (llm_call, tool_invoke, etc.)
to Langfuse observation types (generation, span, event).
"""
from __future__ import annotations

# from agent_observability.integrations.langfuse_adapter import (
#     LangfuseAgentTracer,
# )

# --- Initialize the tracer bridge ---
# tracer = LangfuseAgentTracer(
#     public_key="pk-lf-...",     # Your Langfuse public key
#     secret_key="sk-lf-...",     # Your Langfuse secret key
#     host="https://cloud.langfuse.com",  # or self-hosted URL
# )

# --- Start an agent trace ---
# trace = tracer.start_trace(
#     agent_id="support-bot-v2",
#     metadata={"user_id": "u-12345", "session": "s-001"},
# )

# --- Record an LLM call (maps to Langfuse "generation") ---
# span = trace.start_span(
#     kind="llm_call",
#     name="classify-intent",
#     metadata={"model": "gpt-4o-mini", "temperature": 0.0},
# )
# span.end(output={"intent": "billing_inquiry"})

# --- Record a tool invocation (maps to Langfuse "span") ---
# tool_span = trace.start_span(
#     kind="tool_invoke",
#     name="lookup-account",
#     metadata={"tool": "crm_api"},
# )
# tool_span.end(output={"account_id": "A-789"})

# --- Flush traces to Langfuse ---
# tracer.flush()

print("Example: use_with_langfuse.py")
print("Uncomment the code above and install langfuse to run.")
