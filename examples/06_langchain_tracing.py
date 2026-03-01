#!/usr/bin/env python3
"""Example: LangChain Tracing Integration

Demonstrates wrapping a LangChain chain with the agent-observability
Tracer so every LLM call is traced and costs attributed.

Usage:
    python examples/06_langchain_tracing.py

Requirements:
    pip install agent-observability langchain langchain-openai
"""
from __future__ import annotations

try:
    from langchain.schema import HumanMessage
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

import agent_observability
from agent_observability import Tracer, HierarchicalCostAttributor


def traced_langchain_call(
    tracer: Tracer,
    attributor: HierarchicalCostAttributor,
    user_input: str,
    call_index: int,
) -> str:
    """Execute a LangChain call wrapped in observability tracing."""
    span_id = f"lc-call-{call_index}"
    with tracer.span(f"langchain.invoke.{call_index}", attributes={"input": user_input[:50]}):
        if _LANGCHAIN_AVAILABLE:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            result = llm.invoke([HumanMessage(content=user_input)])
            response = result.content
            usage = getattr(result, "usage_metadata", None)
            input_tokens = usage.get("input_tokens", 100) if usage else 100
            output_tokens = usage.get("output_tokens", 50) if usage else 50
        else:
            response = f"[stub] Response to: {user_input[:40]}"
            input_tokens, output_tokens = 100, 50

        attributor.record_cost(
            node_id="langchain-agent",
            model="gpt-4o-mini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(input_tokens * 0.00000015 + output_tokens * 0.0000006),
            label=f"Call {call_index}: {user_input[:30]}",
        )
    return response


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    if not _LANGCHAIN_AVAILABLE:
        print("LangChain not installed — using stub responses.")
        print("Install with: pip install langchain langchain-openai")

    # Step 1: Set up observability
    tracer = Tracer(service_name="langchain-demo")
    attributor = HierarchicalCostAttributor()
    attributor.create_node(node_id="langchain-agent", label="LangChain Agent", parent_id=None)

    # Step 2: Run traced LangChain calls
    questions = [
        "What is the boiling point of water?",
        "Name three programming paradigms.",
        "What does API stand for?",
    ]

    print("\nRunning traced LangChain calls:")
    for i, question in enumerate(questions):
        response = traced_langchain_call(tracer, attributor, question, i + 1)
        print(f"  [{i + 1}] Q: '{question[:40]}' -> A: '{response[:50]}'")

    # Step 3: Report trace summary
    trace_summary = tracer.summary()
    print(f"\nTrace summary: {trace_summary.total_spans} spans, {trace_summary.error_count} errors")

    # Step 4: Report cost summary
    rollup = attributor.rollup(root_id="langchain-agent")
    print(f"Cost summary: ${rollup.total_cost_usd:.6f} "
          f"| {rollup.total_input_tokens:,} in / {rollup.total_output_tokens:,} out tokens")


if __name__ == "__main__":
    main()
