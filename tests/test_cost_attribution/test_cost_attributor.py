"""Tests for HierarchicalCostAttributor."""
from __future__ import annotations

import pytest

from agent_observability.cost_attribution.cost_attributor import (
    AttributionNode,
    CostRollup,
    HierarchicalCostAttributor,
    NodeType,
)


class TestAttributionNode:
    def test_total_tokens(self) -> None:
        node = AttributionNode(
            node_id="n1", node_type=NodeType.CALL, name="call-gpt",
            input_tokens=1000, output_tokens=500,
        )
        assert node.total_tokens == 1500

    def test_to_dict_structure(self) -> None:
        node = AttributionNode(
            node_id="n1", node_type=NodeType.AGENT, name="agent-1",
        )
        d = node.to_dict()
        assert d["node_id"] == "n1"
        assert d["node_type"] == "agent"
        assert "total_tokens" in d


class TestCostRollup:
    def test_total_tokens(self) -> None:
        rollup = CostRollup(
            node_id="n1", node_name="root",
            total_input_tokens=1000, total_output_tokens=500,
        )
        assert rollup.total_tokens == 1500

    def test_cost_per_call_zero_when_no_calls(self) -> None:
        rollup = CostRollup(node_id="n1", node_name="root", call_count=0)
        assert rollup.cost_per_call == 0.0

    def test_cost_per_call_computed(self) -> None:
        rollup = CostRollup(
            node_id="n1", node_name="root",
            total_cost_usd=0.030, call_count=3,
        )
        assert rollup.cost_per_call == pytest.approx(0.010)

    def test_to_dict_structure(self) -> None:
        rollup = CostRollup(node_id="n1", node_name="root")
        d = rollup.to_dict()
        assert "total_cost_usd" in d
        assert "call_count" in d
        assert "by_model" in d
        assert "by_agent" in d
        assert "by_task" in d
        assert "children" in d


class TestHierarchicalCostAttributor:
    def setup_method(self) -> None:
        self.attributor = HierarchicalCostAttributor()

    def test_initial_rollup_is_empty(self) -> None:
        rollup = self.attributor.rollup()
        assert rollup.total_cost_usd == 0.0
        assert rollup.call_count == 0

    def test_add_agent_returns_id(self) -> None:
        agent_id = self.attributor.add_agent("orchestrator")
        assert isinstance(agent_id, str)
        assert agent_id != ""

    def test_add_agent_creates_node(self) -> None:
        agent_id = self.attributor.add_agent("orchestrator")
        node = self.attributor.get_node(agent_id)
        assert node is not None
        assert node.node_type == NodeType.AGENT
        assert node.name == "orchestrator"

    def test_add_task_returns_id(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "plan_execute")
        assert isinstance(task_id, str)
        assert task_id != agent_id

    def test_add_task_unknown_parent_raises(self) -> None:
        with pytest.raises(KeyError):
            self.attributor.add_task("nonexistent-id", "task")

    def test_add_call_returns_id(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        call_id = self.attributor.add_call(task_id, "gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        assert isinstance(call_id, str)

    def test_add_call_stores_model(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        call_id = self.attributor.add_call(task_id, "claude-sonnet", input_tokens=1000, output_tokens=500, cost_usd=0.015)
        node = self.attributor.get_node(call_id)
        assert node is not None
        assert node.model == "claude-sonnet"
        assert node.input_tokens == 1000
        assert node.output_tokens == 500
        assert node.direct_cost_usd == pytest.approx(0.015)

    def test_rollup_single_call(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        self.attributor.add_call(task_id, "gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=0.025)
        rollup = self.attributor.rollup()
        assert rollup.total_cost_usd == pytest.approx(0.025)
        assert rollup.call_count == 1
        assert rollup.total_input_tokens == 1000
        assert rollup.total_output_tokens == 500

    def test_rollup_multiple_calls_aggregates(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        self.attributor.add_call(task_id, "gpt-4o", input_tokens=500, output_tokens=200, cost_usd=0.010)
        self.attributor.add_call(task_id, "gpt-4o", input_tokens=300, output_tokens=100, cost_usd=0.005)
        rollup = self.attributor.rollup()
        assert rollup.total_cost_usd == pytest.approx(0.015)
        assert rollup.call_count == 2

    def test_rollup_by_model(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        self.attributor.add_call(task_id, "gpt-4o", cost_usd=0.010)
        self.attributor.add_call(task_id, "claude-sonnet", cost_usd=0.008)
        rollup = self.attributor.rollup()
        assert "gpt-4o" in rollup.by_model
        assert "claude-sonnet" in rollup.by_model
        assert rollup.by_model["gpt-4o"] == pytest.approx(0.010)
        assert rollup.by_model["claude-sonnet"] == pytest.approx(0.008)

    def test_rollup_by_agent(self) -> None:
        agent1_id = self.attributor.add_agent("agent-1")
        agent2_id = self.attributor.add_agent("agent-2")
        task1_id = self.attributor.add_task(agent1_id, "task-1")
        task2_id = self.attributor.add_task(agent2_id, "task-2")
        self.attributor.add_call(task1_id, "gpt-4o", cost_usd=0.010)
        self.attributor.add_call(task2_id, "gpt-4o", cost_usd=0.020)
        rollup = self.attributor.rollup()
        assert "agent-1" in rollup.by_agent
        assert "agent-2" in rollup.by_agent
        assert rollup.by_agent["agent-1"] == pytest.approx(0.010)
        assert rollup.by_agent["agent-2"] == pytest.approx(0.020)

    def test_rollup_from_agent_node(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        self.attributor.add_call(task_id, "gpt-4o", cost_usd=0.015)
        rollup = self.attributor.rollup(agent_id)
        assert rollup.total_cost_usd == pytest.approx(0.015)
        assert rollup.node_name == "agent-1"

    def test_rollup_unknown_node_raises(self) -> None:
        with pytest.raises(KeyError):
            self.attributor.rollup("nonexistent-node")

    def test_rollup_to_dict_structure(self) -> None:
        rollup = self.attributor.rollup()
        d = rollup.to_dict()
        assert "total_cost_usd" in d
        assert "total_tokens" in d
        assert "call_count" in d
        assert "children" in d

    def test_list_agents(self) -> None:
        self.attributor.add_agent("agent-1")
        self.attributor.add_agent("agent-2")
        agents = self.attributor.list_agents()
        assert len(agents) == 2
        names = {a.name for a in agents}
        assert "agent-1" in names
        assert "agent-2" in names

    def test_list_tasks(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        self.attributor.add_task(agent_id, "task-a")
        self.attributor.add_task(agent_id, "task-b")
        tasks = self.attributor.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_filtered_by_agent(self) -> None:
        agent1_id = self.attributor.add_agent("agent-1")
        agent2_id = self.attributor.add_agent("agent-2")
        self.attributor.add_task(agent1_id, "task-a")
        self.attributor.add_task(agent2_id, "task-b")
        tasks = self.attributor.list_tasks(agent_id=agent1_id)
        assert len(tasks) == 1
        assert tasks[0].name == "task-a"

    def test_child_rollups_in_result(self) -> None:
        agent_id = self.attributor.add_agent("agent-1")
        task_id = self.attributor.add_task(agent_id, "task-1")
        self.attributor.add_call(task_id, "gpt-4o", cost_usd=0.010)
        rollup = self.attributor.rollup()
        # root has children (agents), agents have children (tasks), tasks have children (calls)
        assert len(rollup.child_rollups) >= 1
        agent_rollup = rollup.child_rollups[0]
        assert len(agent_rollup.child_rollups) >= 1
