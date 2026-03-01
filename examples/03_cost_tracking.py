#!/usr/bin/env python3
"""Example: Cost Tracking

Demonstrates hierarchical cost attribution across multiple agent
calls using HierarchicalCostAttributor.

Usage:
    python examples/03_cost_tracking.py

Requirements:
    pip install agent-observability
"""
from __future__ import annotations

import agent_observability
from agent_observability import (
    HierarchicalCostAttributor,
    AttributionNode,
    CostRollup,
)


def simulate_project_pipeline(attributor: HierarchicalCostAttributor) -> None:
    """Simulate cost attribution across a multi-agent project pipeline."""

    # Top-level: project
    project_node = attributor.create_node(
        node_id="project-q4-analysis",
        label="Q4 Analysis Project",
        parent_id=None,
    )

    # Sub-agents
    research_node = attributor.create_node(
        node_id="agent-researcher",
        label="Research Agent",
        parent_id="project-q4-analysis",
    )
    writer_node = attributor.create_node(
        node_id="agent-writer",
        label="Writing Agent",
        parent_id="project-q4-analysis",
    )

    # Record LLM call costs
    attributor.record_cost(
        node_id="agent-researcher",
        model="claude-haiku-4",
        input_tokens=2000,
        output_tokens=800,
        cost_usd=0.0045,
        label="Market research query",
    )
    attributor.record_cost(
        node_id="agent-researcher",
        model="claude-haiku-4",
        input_tokens=1500,
        output_tokens=600,
        cost_usd=0.0034,
        label="Competitor analysis",
    )
    attributor.record_cost(
        node_id="agent-writer",
        model="claude-sonnet-4",
        input_tokens=3000,
        output_tokens=2000,
        cost_usd=0.0210,
        label="Report drafting",
    )


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    # Step 1: Create the cost attributor
    attributor = HierarchicalCostAttributor()
    print("HierarchicalCostAttributor created.")

    # Step 2: Simulate a pipeline
    print("\nSimulating project pipeline costs...")
    simulate_project_pipeline(attributor)

    # Step 3: Roll up costs
    rollup: CostRollup = attributor.rollup(root_id="project-q4-analysis")
    print(f"\nCost rollup for 'Q4 Analysis Project':")
    print(f"  Total cost: ${rollup.total_cost_usd:.4f}")
    print(f"  Total input tokens: {rollup.total_input_tokens:,}")
    print(f"  Total output tokens: {rollup.total_output_tokens:,}")

    print(f"\nPer-agent breakdown:")
    for node_id, node_rollup in rollup.by_node.items():
        print(f"  [{node_id}] ${node_rollup.total_cost_usd:.4f} "
              f"({node_rollup.call_count} calls)")

    # Step 4: Per-model breakdown
    print(f"\nPer-model breakdown:")
    for model, model_rollup in rollup.by_model.items():
        print(f"  [{model}] ${model_rollup.total_cost_usd:.4f} "
              f"| in={model_rollup.total_input_tokens:,} "
              f"| out={model_rollup.total_output_tokens:,}")


if __name__ == "__main__":
    main()
