#!/usr/bin/env python3
"""Example: Multi-Agent Trace Correlation

Demonstrates how TraceCorrelator links spans from multiple agents
into a unified trace tree for debugging distributed workflows.

Usage:
    python examples/05_trace_correlation.py

Requirements:
    pip install agent-observability
"""
from __future__ import annotations

import agent_observability
from agent_observability import (
    TraceCorrelator,
    CorrelationContext,
    BaggageItem,
    SpanRelationship,
)


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    # Step 1: Create a trace correlator
    correlator = TraceCorrelator()

    # Step 2: Create a root correlation context for a workflow
    root_context = correlator.create_context(
        trace_id="wf-001",
        span_id="span-root",
        agent_id="orchestrator-agent",
        baggage=[
            BaggageItem(key="workflow.type", value="data-analysis"),
            BaggageItem(key="workflow.priority", value="high"),
        ],
    )
    print(f"Root context: trace_id={root_context.trace_id}, agent={root_context.agent_id}")

    # Step 3: Create child spans for sub-agents
    child_contexts: list[CorrelationContext] = []
    sub_agents = [
        ("data-retrieval-agent", "span-retrieval"),
        ("analysis-agent", "span-analysis"),
        ("reporting-agent", "span-report"),
    ]

    for agent_id, span_id in sub_agents:
        child_ctx = correlator.create_child(
            parent_context=root_context,
            span_id=span_id,
            agent_id=agent_id,
            relationship=SpanRelationship.CHILD_OF,
        )
        child_contexts.append(child_ctx)
        print(f"  Child span: agent={agent_id} parent={root_context.span_id}")

    # Step 4: Add a sibling relationship between sub-agents
    if len(child_contexts) >= 2:
        correlator.add_relationship(
            from_span=child_contexts[0].span_id,
            to_span=child_contexts[1].span_id,
            relationship=SpanRelationship.FOLLOWS_FROM,
        )

    # Step 5: Build the trace tree
    try:
        tree = correlator.build_tree(trace_id="wf-001")
        print(f"\nTrace tree for workflow 'wf-001':")
        print(f"  Total spans: {tree.total_spans}")
        print(f"  Root agent: {tree.root.agent_id}")
        print(f"  Children: {len(tree.root.children)}")
        for child in tree.root.children:
            print(f"    - {child.agent_id} (relationship: {child.relationship.value})")
    except Exception as error:
        print(f"Tree build error: {error}")

    # Step 6: Propagate baggage to child agents
    child_baggage = correlator.get_baggage(child_contexts[0])
    print(f"\nBaggage propagated to '{child_contexts[0].agent_id}':")
    for item in child_baggage:
        print(f"  {item.key}={item.value}")


if __name__ == "__main__":
    main()
