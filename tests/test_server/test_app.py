"""Tests for agent_observability.server.app — HTTP handler integration."""
from __future__ import annotations

import io
import json
from http.server import HTTPServer
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.server import routes
from agent_observability.server.app import AgentObservabilityHandler, create_server


def _make_handler(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[AgentObservabilityHandler, io.BytesIO]:
    """Build a handler instance wired to a fake socket for testing.

    Returns the handler and the output buffer to read the response from.
    """
    body_bytes = json.dumps(body or {}).encode("utf-8") if body is not None else b""

    request = MagicMock()
    request.makefile = MagicMock()

    # Build input stream
    request_line = f"{method} {path} HTTP/1.1\r\n"
    headers = (
        f"Host: localhost\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"\r\n"
    )
    full_input = (request_line + headers).encode() + body_bytes
    input_stream = io.BytesIO(full_input)
    output_stream = io.BytesIO()

    request.makefile.side_effect = lambda mode, **kwargs: (
        input_stream if "rb" in mode or "r" in mode else output_stream
    )

    server = MagicMock()
    server.server_address = ("127.0.0.1", 8080)

    handler = AgentObservabilityHandler.__new__(AgentObservabilityHandler)
    handler.request = request
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = server
    handler.rfile = input_stream
    handler.wfile = output_stream

    return handler, output_stream


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Reset module-level state before each test."""
    routes.reset_state()


class TestAgentObservabilityHandlerHealth:
    def test_health_endpoint_returns_200(self) -> None:
        status, data = routes.handle_health()
        assert status == 200
        assert data["status"] == "ok"

    def test_health_service_name(self) -> None:
        status, data = routes.handle_health()
        assert data["service"] == "agent-observability"


class TestAgentObservabilityHandlerTraces:
    def test_create_trace_route(self) -> None:
        body = {
            "agent_id": "test-agent",
            "provider": "openai",
            "model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 50,
        }
        status, data = routes.handle_create_trace(body)
        assert status == 201
        assert "trace_id" in data

    def test_get_trace_route_missing(self) -> None:
        status, data = routes.handle_get_trace("no-such-id")
        assert status == 404

    def test_get_trace_route_found(self) -> None:
        _, created = routes.handle_create_trace({"agent_id": "xyz"})
        trace_id = created["trace_id"]

        status, data = routes.handle_get_trace(trace_id)
        assert status == 200
        assert data["agent_id"] == "xyz"

    def test_list_traces_route(self) -> None:
        routes.handle_create_trace({"agent_id": "a1"})
        routes.handle_create_trace({"agent_id": "a2"})

        status, data = routes.handle_list_traces()
        assert status == 200
        assert data["total"] == 2

    def test_create_trace_invalid_input_tokens(self) -> None:
        body = {"input_tokens": -1}
        status, data = routes.handle_create_trace(body)
        assert status == 422


class TestAgentObservabilityHandlerCosts:
    def test_get_costs_empty(self) -> None:
        status, data = routes.handle_get_costs()
        assert status == 200
        assert data["total_cost_usd"] == 0.0

    def test_get_costs_after_trace(self) -> None:
        routes.handle_create_trace(
            {"provider": "openai", "model": "gpt-4o", "input_tokens": 100, "output_tokens": 50}
        )
        status, data = routes.handle_get_costs()
        assert status == 200
        assert data["record_count"] == 1

    def test_get_costs_agent_filter(self) -> None:
        routes.handle_create_trace(
            {
                "agent_id": "a1",
                "provider": "openai",
                "model": "gpt-4",
                "input_tokens": 100,
                "output_tokens": 50,
            }
        )
        routes.handle_create_trace(
            {
                "agent_id": "a2",
                "provider": "openai",
                "model": "gpt-4",
                "input_tokens": 200,
                "output_tokens": 100,
            }
        )

        status, data = routes.handle_get_costs(agent_id="a1")
        assert status == 200
        assert data["record_count"] == 1


class TestCreateServer:
    def test_create_server_returns_http_server(self) -> None:
        server = create_server(host="127.0.0.1", port=0)
        try:
            assert isinstance(server, HTTPServer)
        finally:
            server.server_close()

    def test_create_server_uses_correct_handler(self) -> None:
        server = create_server(host="127.0.0.1", port=0)
        try:
            assert server.RequestHandlerClass is AgentObservabilityHandler
        finally:
            server.server_close()


class TestServerModels:
    def test_create_trace_request_defaults(self) -> None:
        from agent_observability.server.models import CreateTraceRequest

        req = CreateTraceRequest(provider="openai", model="gpt-4")
        assert req.input_tokens == 0
        assert req.output_tokens == 0
        assert req.operation == "llm_call"

    def test_trace_response_fields(self) -> None:
        from agent_observability.server.models import TraceResponse

        resp = TraceResponse(
            trace_id="t1",
            agent_id="a1",
            session_id="s1",
            task_id="",
            service_name="agent",
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=0,
            cost_usd=0.01,
            operation="llm_call",
            timestamp=1234567890.0,
        )
        assert resp.trace_id == "t1"

    def test_cost_summary_response_fields(self) -> None:
        from agent_observability.server.models import CostSummaryResponse

        resp = CostSummaryResponse(
            total_cost_usd=0.5,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_tokens=1500,
            record_count=3,
        )
        assert resp.total_cost_usd == 0.5
        assert resp.record_count == 3

    def test_health_response_defaults(self) -> None:
        from agent_observability.server.models import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.service == "agent-observability"
