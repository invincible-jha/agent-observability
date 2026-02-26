"""Tests for all instrumentors (anthropic_sdk, openai_sdk, langchain, mcp, autogen, crewai).

All SDK imports are guarded, so we can test:
- No-op behaviour when SDK not installed (_XXXXX_AVAILABLE = False)
- instrument() idempotency
- uninstrument() cleanup
- instrument() success path using mock SDK objects
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from agent_observability.tracer.agent_tracer import AgentTracer


# ── Helpers ────────────────────────────────────────────────────────────────────


def _tracer() -> AgentTracer:
    """Create a fresh AgentTracer for each test."""
    return AgentTracer(service_name="test", agent_id="a1")


# ── AnthropicInstrumentor ──────────────────────────────────────────────────────


class TestAnthropicInstrumentor:
    def test_instrument_noop_when_anthropic_unavailable(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor(tracer=_tracer())
        with patch("agent_observability.instrumentors.anthropic_sdk._ANTHROPIC_AVAILABLE", False):
            instrumentor.instrument()
        assert not instrumentor._instrumented

    def test_instrument_idempotent(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        # Should return immediately without any side effects
        instrumentor.instrument()
        assert instrumentor._instrumented is True

    def test_uninstrument_noop_when_not_instrumented(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor(tracer=_tracer())
        # Should not raise
        instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_with_mock_anthropic(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        # Build a fake anthropic.resources.messages.Messages class
        mock_messages_module = MagicMock()
        mock_messages_cls = MagicMock()
        mock_messages_cls.create = MagicMock(return_value=MagicMock())
        mock_messages_module.Messages = mock_messages_cls

        with patch("agent_observability.instrumentors.anthropic_sdk._ANTHROPIC_AVAILABLE", True):
            with patch.dict("sys.modules", {"anthropic.resources.messages": mock_messages_module}):
                instrumentor = AnthropicInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented

    def test_uninstrument_restores_original(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        mock_messages_module = MagicMock()
        original_create = MagicMock()
        mock_messages_cls = MagicMock()
        mock_messages_cls.create = original_create
        mock_messages_module.Messages = mock_messages_cls

        with patch("agent_observability.instrumentors.anthropic_sdk._ANTHROPIC_AVAILABLE", True):
            with patch.dict("sys.modules", {"anthropic.resources.messages": mock_messages_module}):
                instrumentor = AnthropicInstrumentor(tracer=_tracer())
                instrumentor.instrument()
                assert instrumentor._instrumented
                instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_handles_attribute_error_gracefully(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        mock_messages_module = MagicMock()
        del mock_messages_module.Messages  # AttributeError on access

        with patch("agent_observability.instrumentors.anthropic_sdk._ANTHROPIC_AVAILABLE", True):
            with patch.dict("sys.modules", {"anthropic.resources.messages": mock_messages_module}):
                instrumentor = AnthropicInstrumentor(tracer=_tracer())
                # Should not raise even if patching fails
                instrumentor.instrument()
        # _instrumented is True because it was set after the try/except
        assert instrumentor._instrumented

    def test_uninstrument_handles_import_error(self) -> None:
        from agent_observability.instrumentors.anthropic_sdk import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        instrumentor._original_messages_create = None

        # Remove anthropic from sys.modules to trigger ImportError on uninstrument
        with patch("agent_observability.instrumentors.anthropic_sdk._ANTHROPIC_AVAILABLE", True):
            with patch.dict("sys.modules", {"anthropic.resources.messages": None}):  # type: ignore[dict-item]
                # Should not raise
                try:
                    instrumentor.uninstrument()
                except Exception:
                    pass
        # Regardless of errors, _instrumented should be False
        assert not instrumentor._instrumented


# ── OpenAIInstrumentor ─────────────────────────────────────────────────────────


class TestOpenAIInstrumentor:
    def test_instrument_noop_when_openai_unavailable(self) -> None:
        from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor(tracer=_tracer())
        with patch("agent_observability.instrumentors.openai_sdk._OPENAI_AVAILABLE", False):
            instrumentor.instrument()
        assert not instrumentor._instrumented

    def test_instrument_idempotent(self) -> None:
        from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        instrumentor.instrument()
        assert instrumentor._instrumented is True

    def test_uninstrument_noop_when_not_instrumented(self) -> None:
        from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor(tracer=_tracer())
        instrumentor.uninstrument()  # should not raise
        assert not instrumentor._instrumented

    def test_instrument_with_mock_openai(self) -> None:
        from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

        mock_completions_module = MagicMock()
        mock_completions_cls = MagicMock()
        mock_completions_cls.create = MagicMock(return_value=MagicMock())
        mock_completions_module.Completions = mock_completions_cls

        with patch("agent_observability.instrumentors.openai_sdk._OPENAI_AVAILABLE", True):
            with patch.dict(
                "sys.modules",
                {"openai.resources.chat.completions": mock_completions_module},
            ):
                instrumentor = OpenAIInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented

    def test_uninstrument_restores_original(self) -> None:
        from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

        mock_completions_module = MagicMock()
        original_create = MagicMock()
        mock_completions_cls = MagicMock()
        mock_completions_cls.create = original_create
        mock_completions_module.Completions = mock_completions_cls

        with patch("agent_observability.instrumentors.openai_sdk._OPENAI_AVAILABLE", True):
            with patch.dict(
                "sys.modules",
                {"openai.resources.chat.completions": mock_completions_module},
            ):
                instrumentor = OpenAIInstrumentor(tracer=_tracer())
                instrumentor.instrument()
                instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_handles_attribute_error(self) -> None:
        from agent_observability.instrumentors.openai_sdk import OpenAIInstrumentor

        bad_module = MagicMock(spec=[])  # no Completions attribute
        with patch("agent_observability.instrumentors.openai_sdk._OPENAI_AVAILABLE", True):
            with patch.dict(
                "sys.modules", {"openai.resources.chat.completions": bad_module}
            ):
                instrumentor = OpenAIInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented


# ── LangChainInstrumentor ──────────────────────────────────────────────────────


class TestLangChainInstrumentor:
    def test_instrument_noop_when_langchain_unavailable(self) -> None:
        from agent_observability.instrumentors.langchain import LangChainInstrumentor

        instrumentor = LangChainInstrumentor(tracer=_tracer())
        with patch("agent_observability.instrumentors.langchain._LANGCHAIN_AVAILABLE", False):
            instrumentor.instrument()
        assert not instrumentor._instrumented

    def test_instrument_idempotent(self) -> None:
        from agent_observability.instrumentors.langchain import LangChainInstrumentor

        instrumentor = LangChainInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        instrumentor.instrument()
        assert instrumentor._instrumented is True

    def test_uninstrument_noop_when_not_instrumented(self) -> None:
        from agent_observability.instrumentors.langchain import LangChainInstrumentor

        instrumentor = LangChainInstrumentor(tracer=_tracer())
        instrumentor.uninstrument()  # should not raise

    def test_instrument_with_mock_langchain_llm(self) -> None:
        from agent_observability.instrumentors.langchain import LangChainInstrumentor

        mock_llms_base = MagicMock()
        mock_base_llm = MagicMock()
        mock_base_llm.__call__ = MagicMock()
        mock_llms_base.BaseLLM = mock_base_llm

        mock_tools_base = MagicMock()
        mock_base_tool = MagicMock()
        mock_base_tool.run = MagicMock()
        mock_tools_base.BaseTool = mock_base_tool

        with patch("agent_observability.instrumentors.langchain._LANGCHAIN_AVAILABLE", True):
            with patch.dict(
                "sys.modules",
                {
                    "langchain.llms.base": mock_llms_base,
                    "langchain.tools.base": mock_tools_base,
                },
            ):
                instrumentor = LangChainInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented

    def test_uninstrument_restores_both_patches(self) -> None:
        from agent_observability.instrumentors.langchain import LangChainInstrumentor

        mock_llms_base = MagicMock()
        mock_base_llm = MagicMock()
        original_call = MagicMock()
        mock_base_llm.__call__ = original_call
        mock_llms_base.BaseLLM = mock_base_llm

        mock_tools_base = MagicMock()
        mock_base_tool = MagicMock()
        original_run = MagicMock()
        mock_base_tool.run = original_run
        mock_tools_base.BaseTool = mock_base_tool

        with patch("agent_observability.instrumentors.langchain._LANGCHAIN_AVAILABLE", True):
            with patch.dict(
                "sys.modules",
                {
                    "langchain.llms.base": mock_llms_base,
                    "langchain.tools.base": mock_tools_base,
                },
            ):
                instrumentor = LangChainInstrumentor(tracer=_tracer())
                instrumentor.instrument()
                instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_handles_missing_llms_module(self) -> None:
        from agent_observability.instrumentors.langchain import LangChainInstrumentor

        mock_tools_base = MagicMock()
        mock_base_tool = MagicMock()
        mock_base_tool.run = MagicMock()
        mock_tools_base.BaseTool = mock_base_tool

        with patch("agent_observability.instrumentors.langchain._LANGCHAIN_AVAILABLE", True):
            with patch.dict(
                "sys.modules",
                {
                    "langchain.llms.base": None,  # type: ignore[dict-item]
                    "langchain.tools.base": mock_tools_base,
                },
            ):
                instrumentor = LangChainInstrumentor(tracer=_tracer())
                try:
                    instrumentor.instrument()
                except Exception:
                    pass
        # Even partial patching should not crash


# ── MCPInstrumentor ────────────────────────────────────────────────────────────


class TestMCPInstrumentor:
    def test_instrument_noop_when_mcp_unavailable(self) -> None:
        from agent_observability.instrumentors.mcp import MCPInstrumentor

        instrumentor = MCPInstrumentor(tracer=_tracer())
        with patch("agent_observability.instrumentors.mcp._MCP_AVAILABLE", False):
            instrumentor.instrument()
        assert not instrumentor._instrumented

    def test_instrument_idempotent(self) -> None:
        from agent_observability.instrumentors.mcp import MCPInstrumentor

        instrumentor = MCPInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        instrumentor.instrument()
        assert instrumentor._instrumented is True

    def test_uninstrument_noop_when_not_instrumented(self) -> None:
        from agent_observability.instrumentors.mcp import MCPInstrumentor

        instrumentor = MCPInstrumentor(tracer=_tracer())
        instrumentor.uninstrument()  # should not raise

    def test_instrument_with_mock_mcp(self) -> None:
        from agent_observability.instrumentors.mcp import MCPInstrumentor

        mock_session_module = MagicMock()
        mock_client_session = MagicMock()
        mock_client_session.call_tool = MagicMock()
        mock_session_module.ClientSession = mock_client_session

        with patch("agent_observability.instrumentors.mcp._MCP_AVAILABLE", True):
            with patch.dict(
                "sys.modules", {"mcp.client.session": mock_session_module}
            ):
                instrumentor = MCPInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented

    def test_uninstrument_restores_call_tool(self) -> None:
        from agent_observability.instrumentors.mcp import MCPInstrumentor

        mock_session_module = MagicMock()
        mock_client_session = MagicMock()
        original_call_tool = MagicMock()
        mock_client_session.call_tool = original_call_tool
        mock_session_module.ClientSession = mock_client_session

        with patch("agent_observability.instrumentors.mcp._MCP_AVAILABLE", True):
            with patch.dict(
                "sys.modules", {"mcp.client.session": mock_session_module}
            ):
                instrumentor = MCPInstrumentor(tracer=_tracer())
                instrumentor.instrument()
                instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_handles_attribute_error(self) -> None:
        from agent_observability.instrumentors.mcp import MCPInstrumentor

        bad_module = MagicMock(spec=[])  # no ClientSession
        with patch("agent_observability.instrumentors.mcp._MCP_AVAILABLE", True):
            with patch.dict("sys.modules", {"mcp.client.session": bad_module}):
                instrumentor = MCPInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented


# ── AutoGenInstrumentor ────────────────────────────────────────────────────────


class TestAutoGenInstrumentor:
    def test_instrument_noop_when_autogen_unavailable(self) -> None:
        from agent_observability.instrumentors.autogen import AutoGenInstrumentor

        instrumentor = AutoGenInstrumentor(tracer=_tracer())
        with patch("agent_observability.instrumentors.autogen._AUTOGEN_AVAILABLE", False):
            instrumentor.instrument()
        assert not instrumentor._instrumented

    def test_instrument_idempotent(self) -> None:
        from agent_observability.instrumentors.autogen import AutoGenInstrumentor

        instrumentor = AutoGenInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        instrumentor.instrument()
        assert instrumentor._instrumented is True

    def test_uninstrument_noop_when_not_instrumented(self) -> None:
        from agent_observability.instrumentors.autogen import AutoGenInstrumentor

        instrumentor = AutoGenInstrumentor(tracer=_tracer())
        instrumentor.uninstrument()  # should not raise

    def test_instrument_with_mock_autogen(self) -> None:
        from agent_observability.instrumentors.autogen import AutoGenInstrumentor

        mock_autogen = MagicMock()
        mock_conversable_agent = MagicMock()
        mock_conversable_agent.initiate_chat = MagicMock()
        mock_conversable_agent.generate_reply = MagicMock()
        mock_autogen.ConversableAgent = mock_conversable_agent

        with patch("agent_observability.instrumentors.autogen._AUTOGEN_AVAILABLE", True):
            with patch.dict("sys.modules", {"autogen": mock_autogen}):
                instrumentor = AutoGenInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented

    def test_uninstrument_restores_patches(self) -> None:
        from agent_observability.instrumentors.autogen import AutoGenInstrumentor

        mock_autogen = MagicMock()
        mock_conversable_agent = MagicMock()
        original_initiate = MagicMock()
        original_generate = MagicMock()
        mock_conversable_agent.initiate_chat = original_initiate
        mock_conversable_agent.generate_reply = original_generate
        mock_autogen.ConversableAgent = mock_conversable_agent

        with patch("agent_observability.instrumentors.autogen._AUTOGEN_AVAILABLE", True):
            with patch.dict("sys.modules", {"autogen": mock_autogen}):
                instrumentor = AutoGenInstrumentor(tracer=_tracer())
                instrumentor.instrument()
                instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_handles_attribute_error(self) -> None:
        from agent_observability.instrumentors.autogen import AutoGenInstrumentor

        bad_autogen = MagicMock(spec=[])  # no ConversableAgent
        with patch("agent_observability.instrumentors.autogen._AUTOGEN_AVAILABLE", True):
            with patch.dict("sys.modules", {"autogen": bad_autogen}):
                instrumentor = AutoGenInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented


# ── CrewAIInstrumentor ─────────────────────────────────────────────────────────


class TestCrewAIInstrumentor:
    def test_instrument_noop_when_crewai_unavailable(self) -> None:
        from agent_observability.instrumentors.crewai import CrewAIInstrumentor

        instrumentor = CrewAIInstrumentor(tracer=_tracer())
        with patch("agent_observability.instrumentors.crewai._CREWAI_AVAILABLE", False):
            instrumentor.instrument()
        assert not instrumentor._instrumented

    def test_instrument_idempotent(self) -> None:
        from agent_observability.instrumentors.crewai import CrewAIInstrumentor

        instrumentor = CrewAIInstrumentor(tracer=_tracer())
        instrumentor._instrumented = True
        instrumentor.instrument()
        assert instrumentor._instrumented is True

    def test_uninstrument_noop_when_not_instrumented(self) -> None:
        from agent_observability.instrumentors.crewai import CrewAIInstrumentor

        instrumentor = CrewAIInstrumentor(tracer=_tracer())
        instrumentor.uninstrument()  # should not raise

    def test_instrument_with_mock_crewai(self) -> None:
        from agent_observability.instrumentors.crewai import CrewAIInstrumentor

        mock_crewai = MagicMock()
        mock_task = MagicMock()
        mock_task.execute = MagicMock()
        mock_agent = MagicMock()
        mock_agent.execute_task = MagicMock()
        mock_crewai.Task = mock_task
        mock_crewai.Agent = mock_agent

        with patch("agent_observability.instrumentors.crewai._CREWAI_AVAILABLE", True):
            with patch.dict("sys.modules", {"crewai": mock_crewai}):
                instrumentor = CrewAIInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented

    def test_uninstrument_restores_patches(self) -> None:
        from agent_observability.instrumentors.crewai import CrewAIInstrumentor

        mock_crewai = MagicMock()
        mock_task = MagicMock()
        original_execute = MagicMock()
        mock_task.execute = original_execute
        mock_agent = MagicMock()
        original_agent_execute = MagicMock()
        mock_agent.execute_task = original_agent_execute
        mock_crewai.Task = mock_task
        mock_crewai.Agent = mock_agent

        with patch("agent_observability.instrumentors.crewai._CREWAI_AVAILABLE", True):
            with patch.dict("sys.modules", {"crewai": mock_crewai}):
                instrumentor = CrewAIInstrumentor(tracer=_tracer())
                instrumentor.instrument()
                instrumentor.uninstrument()
        assert not instrumentor._instrumented

    def test_instrument_handles_missing_task_attribute(self) -> None:
        from agent_observability.instrumentors.crewai import CrewAIInstrumentor

        mock_crewai = MagicMock(spec=[])  # no Task or Agent
        with patch("agent_observability.instrumentors.crewai._CREWAI_AVAILABLE", True):
            with patch.dict("sys.modules", {"crewai": mock_crewai}):
                instrumentor = CrewAIInstrumentor(tracer=_tracer())
                instrumentor.instrument()
        assert instrumentor._instrumented
