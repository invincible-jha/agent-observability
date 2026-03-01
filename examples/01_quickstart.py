#!/usr/bin/env python3
"""Example: Quickstart

Demonstrates the minimal setup for agent-observability using the
Tracer convenience class to trace agent operations.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-observability
"""
from __future__ import annotations

import agent_observability
from agent_observability import Tracer


def simulate_agent_call(tracer: Tracer, user_input: str) -> str:
    """Simulate a traced agent call."""
    with tracer.span("agent.process", attributes={"input_length": len(user_input)}) as span:
        # Simulate LLM call
        with tracer.span("llm.call", attributes={"model": "claude-haiku-4"}):
            response = f"Processed: {user_input[:40]}"
        span.set_attribute("output_length", len(response))
    return response


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    # Step 1: Create a zero-config tracer
    tracer = Tracer(service_name="my-agent")
    print(f"Tracer created for service: {tracer.service_name}")

    # Step 2: Trace some agent operations
    inputs = [
        "What is the capital of France?",
        "Summarise the quarterly report.",
        "Schedule a meeting for next Monday.",
    ]

    print("\nTracing agent calls:")
    for user_input in inputs:
        try:
            response = simulate_agent_call(tracer, user_input)
            print(f"  Input: '{user_input[:40]}' -> Response: '{response[:40]}'")
        except Exception as error:
            print(f"  Error tracing call: {error}")

    # Step 3: Print trace summary
    summary = tracer.summary()
    print(f"\nTrace summary:")
    print(f"  Total spans: {summary.total_spans}")
    print(f"  Errors: {summary.error_count}")


if __name__ == "__main__":
    main()
