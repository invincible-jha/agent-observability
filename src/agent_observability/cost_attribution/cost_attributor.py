"""HierarchicalCostAttributor — tree-based cost rollup by agent → task → call.

Builds an AttributionNode tree where leaf nodes represent individual LLM calls
and interior nodes aggregate costs from their children. Roll-up computation
propagates costs from leaves to the root.

Example
-------
::

    from agent_observability.cost_attribution.cost_attributor import (
        HierarchicalCostAttributor,
    )

    attributor = HierarchicalCostAttributor()
    agent_id = attributor.add_agent("orchestrator")
    task_id = attributor.add_task(agent_id, "plan_and_execute")
    attributor.add_call(task_id, "gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=0.015)
    attributor.add_call(task_id, "gpt-4o", input_tokens=500, output_tokens=200, cost_usd=0.008)
    rollup = attributor.rollup(agent_id)
    print(rollup.total_cost_usd, rollup.total_tokens)
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """The level of the attribution tree node."""

    ROOT = "root"
    AGENT = "agent"
    TASK = "task"
    CALL = "call"


@dataclass
class AttributionNode:
    """A node in the cost attribution tree.

    Attributes
    ----------
    node_id:
        Unique identifier for this node.
    node_type:
        Level in the hierarchy (root, agent, task, call).
    name:
        Human-readable label (agent name, task name, or call ID).
    parent_id:
        ID of the parent node. None for root.
    direct_cost_usd:
        Cost directly associated with this node (leaf-level only).
    input_tokens:
        LLM input tokens consumed at this node (leaf-level only).
    output_tokens:
        LLM output tokens produced at this node (leaf-level only).
    model:
        Model used for this call (leaf-level only).
    timestamp_utc:
        When this node was created.
    metadata:
        Arbitrary additional data.
    """

    node_id: str
    node_type: NodeType
    name: str
    parent_id: Optional[str] = None
    direct_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    timestamp_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens at this node (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "parent_id": self.parent_id,
            "direct_cost_usd": self.direct_cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "timestamp_utc": self.timestamp_utc.isoformat(),
        }


@dataclass
class CostRollup:
    """Rolled-up cost summary for a node and all its descendants.

    Attributes
    ----------
    node_id:
        The root node of this rollup.
    node_name:
        Human-readable name of the root node.
    total_cost_usd:
        Total cost including all descendants.
    total_input_tokens:
        Total input tokens across all descendants.
    total_output_tokens:
        Total output tokens across all descendants.
    call_count:
        Number of leaf (call) nodes included.
    by_model:
        Cost breakdown per model.
    by_agent:
        Cost breakdown per agent (for root-level rollups).
    by_task:
        Cost breakdown per task.
    child_rollups:
        Rollups for direct children of this node.
    """

    node_id: str
    node_name: str
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    by_agent: dict[str, float] = field(default_factory=dict)
    by_task: dict[str, float] = field(default_factory=dict)
    child_rollups: list["CostRollup"] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cost_per_call(self) -> float:
        """Average cost per LLM call. Returns 0.0 if no calls."""
        return self.total_cost_usd / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "cost_per_call": round(self.cost_per_call, 6),
            "by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "by_agent": {k: round(v, 6) for k, v in self.by_agent.items()},
            "by_task": {k: round(v, 6) for k, v in self.by_task.items()},
            "children": [c.to_dict() for c in self.child_rollups],
        }


class HierarchicalCostAttributor:
    """Build and roll up a cost attribution tree: agent → task → call.

    Maintains an in-memory tree of AttributionNodes. Agent nodes are
    children of the root, task nodes are children of agents, and call
    nodes are children of tasks. Call :meth:`rollup` to compute aggregated
    costs from any node level.

    Example
    -------
    ::

        attributor = HierarchicalCostAttributor()
        agent_id = attributor.add_agent("orchestrator")
        task_id = attributor.add_task(agent_id, "plan_and_execute")
        attributor.add_call(task_id, "gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=0.015)
        rollup = attributor.rollup()
        print(rollup.total_cost_usd)
    """

    def __init__(self) -> None:
        self._root = AttributionNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.ROOT,
            name="root",
        )
        self._nodes: dict[str, AttributionNode] = {self._root.node_id: self._root}
        self._children: dict[str, list[str]] = {self._root.node_id: []}
        self._lock = threading.Lock()

    @property
    def root_id(self) -> str:
        """The ID of the root node."""
        return self._root.node_id

    def add_agent(
        self,
        name: str,
        *,
        metadata: Optional[dict[str, object]] = None,
    ) -> str:
        """Add an agent node as a child of the root.

        Parameters
        ----------
        name:
            Human-readable agent name.
        metadata:
            Optional additional data.

        Returns
        -------
        str
            The new agent node's ID.
        """
        return self._add_node(
            parent_id=self._root.node_id,
            node_type=NodeType.AGENT,
            name=name,
            metadata=metadata,
        )

    def add_task(
        self,
        agent_id: str,
        name: str,
        *,
        metadata: Optional[dict[str, object]] = None,
    ) -> str:
        """Add a task node as a child of an agent node.

        Parameters
        ----------
        agent_id:
            The parent agent's node ID.
        name:
            Human-readable task name.
        metadata:
            Optional additional data.

        Returns
        -------
        str
            The new task node's ID.

        Raises
        ------
        KeyError
            If the agent_id is not found.
        """
        return self._add_node(
            parent_id=agent_id,
            node_type=NodeType.TASK,
            name=name,
            metadata=metadata,
        )

    def add_call(
        self,
        task_id: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[dict[str, object]] = None,
    ) -> str:
        """Add an LLM call node as a child of a task node.

        Parameters
        ----------
        task_id:
            The parent task's node ID.
        model:
            Model identifier used for this call.
        input_tokens:
            Number of input tokens consumed.
        output_tokens:
            Number of output tokens produced.
        cost_usd:
            Cost of this call in US dollars.
        metadata:
            Optional additional data.

        Returns
        -------
        str
            The new call node's ID.
        """
        node_id = self._add_node(
            parent_id=task_id,
            node_type=NodeType.CALL,
            name=f"call-{model}",
            metadata=metadata,
        )
        with self._lock:
            node = self._nodes[node_id]
            node.model = model
            node.input_tokens = input_tokens
            node.output_tokens = output_tokens
            node.direct_cost_usd = cost_usd
        return node_id

    def _add_node(
        self,
        parent_id: str,
        node_type: NodeType,
        name: str,
        metadata: Optional[dict[str, object]] = None,
    ) -> str:
        """Internal helper to add a node to the tree."""
        with self._lock:
            if parent_id not in self._nodes:
                raise KeyError(f"Parent node {parent_id!r} not found.")
            node_id = str(uuid.uuid4())
            node = AttributionNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                parent_id=parent_id,
                metadata=dict(metadata or {}),
            )
            self._nodes[node_id] = node
            self._children.setdefault(parent_id, []).append(node_id)
            self._children[node_id] = []
        return node_id

    def rollup(self, node_id: Optional[str] = None) -> CostRollup:
        """Compute a hierarchical cost rollup from a given node.

        Parameters
        ----------
        node_id:
            The node to roll up from. Defaults to the root node.

        Returns
        -------
        CostRollup
            Aggregated cost summary for the node and all descendants.

        Raises
        ------
        KeyError
            If the node_id is not found.
        """
        target_id = node_id or self._root.node_id
        with self._lock:
            if target_id not in self._nodes:
                raise KeyError(f"Node {target_id!r} not found.")
            # Work with copies to avoid lock contention during recursion
            nodes = dict(self._nodes)
            children = {k: list(v) for k, v in self._children.items()}

        return self._build_rollup(target_id, nodes, children)

    def _build_rollup(
        self,
        node_id: str,
        nodes: dict[str, AttributionNode],
        children: dict[str, list[str]],
    ) -> CostRollup:
        """Recursively build a cost rollup for a node."""
        node = nodes[node_id]
        child_ids = children.get(node_id, [])

        # Leaf node (call)
        if node.node_type == NodeType.CALL:
            by_model: dict[str, float] = {}
            if node.model:
                by_model[node.model] = node.direct_cost_usd
            return CostRollup(
                node_id=node_id,
                node_name=node.name,
                total_cost_usd=node.direct_cost_usd,
                total_input_tokens=node.input_tokens,
                total_output_tokens=node.output_tokens,
                call_count=1,
                by_model=by_model,
            )

        # Interior node: aggregate from children
        child_rollups = [
            self._build_rollup(child_id, nodes, children)
            for child_id in child_ids
        ]

        total_cost = sum(cr.total_cost_usd for cr in child_rollups)
        total_input = sum(cr.total_input_tokens for cr in child_rollups)
        total_output = sum(cr.total_output_tokens for cr in child_rollups)
        call_count = sum(cr.call_count for cr in child_rollups)

        # Aggregate by_model
        by_model_agg: dict[str, float] = {}
        for cr in child_rollups:
            for model, cost in cr.by_model.items():
                by_model_agg[model] = by_model_agg.get(model, 0.0) + cost

        # Aggregate by_agent (only meaningful at root level)
        by_agent: dict[str, float] = {}
        by_task: dict[str, float] = {}

        if node.node_type == NodeType.ROOT:
            for cr in child_rollups:
                # Each child rollup at root level is an agent
                by_agent[cr.node_name] = cr.total_cost_usd
                for inner_cr in cr.child_rollups:
                    by_task[inner_cr.node_name] = inner_cr.total_cost_usd

        elif node.node_type == NodeType.AGENT:
            for cr in child_rollups:
                by_task[cr.node_name] = cr.total_cost_usd

        return CostRollup(
            node_id=node_id,
            node_name=node.name,
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            call_count=call_count,
            by_model=by_model_agg,
            by_agent=by_agent,
            by_task=by_task,
            child_rollups=child_rollups,
        )

    def get_node(self, node_id: str) -> Optional[AttributionNode]:
        """Return a node by ID, or None if not found."""
        with self._lock:
            return self._nodes.get(node_id)

    def list_agents(self) -> list[AttributionNode]:
        """Return all agent-level nodes."""
        with self._lock:
            return [
                node for node in self._nodes.values()
                if node.node_type == NodeType.AGENT
            ]

    def list_tasks(self, agent_id: Optional[str] = None) -> list[AttributionNode]:
        """Return all task-level nodes, optionally filtered by agent.

        Parameters
        ----------
        agent_id:
            If provided, only return tasks that are children of this agent.
        """
        with self._lock:
            tasks = [
                node for node in self._nodes.values()
                if node.node_type == NodeType.TASK
            ]
        if agent_id is not None:
            tasks = [t for t in tasks if t.parent_id == agent_id]
        return tasks
