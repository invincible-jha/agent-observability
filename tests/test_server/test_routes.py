"""Tests for agent_observability.server.routes."""
from __future__ import annotations

import pytest

from agent_observability.server import routes


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Reset module-level state before each test."""
    routes.reset_state()


class TestHandleCreateTrace:
    def test_creates_trace_with_provider_model(self) -> None:
        body = {
            "agent_id": "agent-001",
            "session_id": "sess-001",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "operation": "llm_call",
        }
        status, data = routes.handle_create_trace(body)

        assert status == 201
        assert "trace_id" in data
        assert data["agent_id"] == "agent-001"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"
        assert data["input_tokens"] == 100

    def test_creates_trace_without_provider(self) -> None:
        body = {"agent_id": "agent-002", "session_id": "sess-002"}
        status, data = routes.handle_create_trace(body)

        assert status == 201
        assert "trace_id" in data
        assert data["agent_id"] == "agent-002"

    def test_rejects_invalid_body(self) -> None:
        body = {"input_tokens": "not-a-number"}
        status, data = routes.handle_create_trace(body)

        assert status == 422
        assert "error" in data

    def test_trace_stored_in_store(self) -> None:
        body = {"agent_id": "agent-003", "provider": "anthropic", "model": "claude-3"}
        _, data = routes.handle_create_trace(body)

        trace_id = data["trace_id"]
        status2, data2 = routes.handle_get_trace(trace_id)
        assert status2 == 200
        assert data2["trace_id"] == trace_id

    def test_tags_preserved(self) -> None:
        body = {"tags": {"env": "prod", "region": "us-east"}}
        status, data = routes.handle_create_trace(body)

        assert status == 201
        assert data["tags"]["env"] == "prod"

    def test_cost_recorded_when_provider_model_present(self) -> None:
        body = {
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 500,
            "output_tokens": 200,
        }
        routes.handle_create_trace(body)

        tracker = routes.get_cost_tracker()
        assert len(tracker) == 1


class TestHandleGetTrace:
    def test_returns_404_for_missing_trace(self) -> None:
        status, data = routes.handle_get_trace("nonexistent-trace-id")

        assert status == 404
        assert data["error"] == "Not found"

    def test_returns_200_for_existing_trace(self) -> None:
        _, created = routes.handle_create_trace({"agent_id": "a1"})
        trace_id = created["trace_id"]

        status, data = routes.handle_get_trace(trace_id)

        assert status == 200
        assert data["trace_id"] == trace_id
        assert data["agent_id"] == "a1"


class TestHandleListTraces:
    def test_returns_empty_list_when_no_traces(self) -> None:
        status, data = routes.handle_list_traces()

        assert status == 200
        assert data["traces"] == []
        assert data["total"] == 0

    def test_returns_all_traces(self) -> None:
        routes.handle_create_trace({"agent_id": "a1"})
        routes.handle_create_trace({"agent_id": "a2"})

        status, data = routes.handle_list_traces()

        assert status == 200
        assert data["total"] == 2

    def test_filters_by_agent_id(self) -> None:
        routes.handle_create_trace({"agent_id": "agent-alpha"})
        routes.handle_create_trace({"agent_id": "agent-beta"})

        status, data = routes.handle_list_traces(agent_id="agent-alpha")

        assert status == 200
        assert data["total"] == 1
        assert data["traces"][0]["agent_id"] == "agent-alpha"

    def test_respects_limit(self) -> None:
        for i in range(5):
            routes.handle_create_trace({"agent_id": f"agent-{i}"})

        status, data = routes.handle_list_traces(limit=3)

        assert status == 200
        assert len(data["traces"]) == 3


class TestHandleGetCosts:
    def test_returns_zero_summary_initially(self) -> None:
        status, data = routes.handle_get_costs()

        assert status == 200
        assert data["total_cost_usd"] == 0.0
        assert data["record_count"] == 0

    def test_returns_cost_after_trace_with_provider(self) -> None:
        routes.handle_create_trace(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost_usd": 0.025,
            }
        )

        status, data = routes.handle_get_costs()

        assert status == 200
        assert data["record_count"] == 1
        assert data["total_cost_usd"] > 0

    def test_by_model_breakdown(self) -> None:
        routes.handle_create_trace(
            {"provider": "openai", "model": "gpt-4o", "input_tokens": 100, "output_tokens": 50}
        )
        routes.handle_create_trace(
            {
                "provider": "anthropic",
                "model": "claude-3-opus",
                "input_tokens": 100,
                "output_tokens": 50,
            }
        )

        status, data = routes.handle_get_costs()

        assert status == 200
        assert "gpt-4o" in data["by_model"] or "claude-3-opus" in data["by_model"]


class TestHandleHealth:
    def test_returns_ok_status(self) -> None:
        status, data = routes.handle_health()

        assert status == 200
        assert data["status"] == "ok"
        assert data["service"] == "agent-observability"

    def test_reports_trace_count(self) -> None:
        routes.handle_create_trace({"agent_id": "a1"})

        status, data = routes.handle_health()

        assert status == 200
        assert data["trace_count"] == 1
