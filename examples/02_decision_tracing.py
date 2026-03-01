#!/usr/bin/env python3
"""Example: Decision Tracing

Demonstrates how to use DecisionTracker to record and query
agent decision points throughout a workflow.

Usage:
    python examples/02_decision_tracing.py

Requirements:
    pip install agent-observability
"""
from __future__ import annotations

import agent_observability
from agent_observability import (
    DecisionTracker,
    DecisionSpan,
    DecisionStatus,
    DecisionQuery,
)


def run_workflow_with_decisions(tracker: DecisionTracker) -> None:
    """Simulate a multi-step agent workflow with decision tracking."""

    # Decision 1: Route the request
    routing_span = tracker.record(
        decision_id="routing-001",
        description="Route user request to appropriate handler",
        options=["search", "generate", "retrieve"],
        chosen="search",
        rationale="Query contains a question mark — treat as search",
        status=DecisionStatus.ACCEPTED,
        metadata={"confidence": 0.92},
    )
    print(f"  Decision [{routing_span.decision_id}]: route=search (confidence=0.92)")

    # Decision 2: Select data source
    source_span = tracker.record(
        decision_id="datasource-002",
        description="Select the most appropriate data source",
        options=["vector_store", "sql_db", "web_search"],
        chosen="vector_store",
        rationale="Query matches known document corpus",
        status=DecisionStatus.ACCEPTED,
        metadata={"source_count": 3},
    )
    print(f"  Decision [{source_span.decision_id}]: source=vector_store")

    # Decision 3: Response format (rejected option for audit trail)
    format_span = tracker.record(
        decision_id="format-003",
        description="Choose response format",
        options=["markdown", "plain_text", "json"],
        chosen="markdown",
        rationale="User interface supports rich text rendering",
        status=DecisionStatus.ACCEPTED,
        metadata={"user_interface": "web"},
    )
    print(f"  Decision [{format_span.decision_id}]: format=markdown")


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    # Step 1: Create a decision tracker
    tracker = DecisionTracker(agent_id="research-agent-v1")
    print(f"Decision tracker created for: {tracker.agent_id}")

    # Step 2: Run a workflow with decision recording
    print("\nRunning workflow with decision tracking:")
    run_workflow_with_decisions(tracker)

    # Step 3: Query decisions
    print("\nQuerying decisions:")
    query = DecisionQuery(agent_id="research-agent-v1", status=DecisionStatus.ACCEPTED)
    result = tracker.query(query)
    print(f"  Accepted decisions: {result.total_count}")

    for decision in result.decisions:
        print(f"  [{decision.decision_id}] chosen='{decision.chosen}' | "
              f"confidence={decision.metadata.get('confidence', 'N/A')}")

    # Step 4: Get full decision history
    history = tracker.history()
    print(f"\nFull decision history: {len(history)} entries")
    print(f"  Agent: {tracker.agent_id}")
    print(f"  Decisions recorded: {tracker.count()}")


if __name__ == "__main__":
    main()
