"""Unit tests for agent_observability framework adapters."""
from __future__ import annotations

import pytest

from agent_observability.adapters import (
    AnthropicTracer,
    CrewAITracer,
    LangChainTracer,
    MicrosoftAgentTracer,
    OpenAIAgentsTracer,
)


# ---------------------------------------------------------------------------
# LangChainTracer
# ---------------------------------------------------------------------------


class TestLangChainTracer:
    def test_construction_no_args(self) -> None:
        tracer = LangChainTracer()
        assert tracer.tracer_provider is None

    def test_construction_with_provider(self) -> None:
        sentinel = object()
        tracer = LangChainTracer(tracer_provider=sentinel)
        assert tracer.tracer_provider is sentinel

    def test_on_llm_start_returns_dict(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_llm_start("gpt-4o", "Hello world")
        assert isinstance(result, dict)

    def test_on_llm_start_has_name_and_attributes(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_llm_start("gpt-4o", "Hello world")
        assert "name" in result
        assert "attributes" in result
        assert result["attributes"]["llm.model"] == "gpt-4o"
        assert result["attributes"]["llm.prompt_length"] == len("Hello world")

    def test_on_llm_end_returns_dict(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_llm_end("some response")
        assert isinstance(result, dict)
        assert "name" in result

    def test_on_tool_start_returns_dict(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_tool_start("search", {"query": "foo"})
        assert isinstance(result, dict)
        assert result["attributes"]["tool.name"] == "search"

    def test_on_tool_end_returns_dict(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_tool_end(["result1", "result2"])
        assert isinstance(result, dict)
        assert "attributes" in result

    def test_on_chain_start_returns_dict(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_chain_start("MyChain", {"input": "foo"})
        assert isinstance(result, dict)
        assert result["attributes"]["chain.name"] == "MyChain"
        assert "input" in result["attributes"]["chain.input_keys"]

    def test_on_chain_end_returns_dict(self) -> None:
        tracer = LangChainTracer()
        result = tracer.on_chain_end({"output": "bar"})
        assert isinstance(result, dict)
        assert "output" in result["attributes"]["chain.output_keys"]

    def test_framework_attribute_on_all_methods(self) -> None:
        tracer = LangChainTracer()
        assert tracer.on_llm_start("m", "p")["attributes"]["framework"] == "langchain"
        assert tracer.on_llm_end("r")["attributes"]["framework"] == "langchain"
        assert tracer.on_tool_start("t", {})["attributes"]["framework"] == "langchain"
        assert tracer.on_tool_end("out")["attributes"]["framework"] == "langchain"
        assert tracer.on_chain_start("c", {})["attributes"]["framework"] == "langchain"
        assert tracer.on_chain_end({})["attributes"]["framework"] == "langchain"


# ---------------------------------------------------------------------------
# CrewAITracer
# ---------------------------------------------------------------------------


class TestCrewAITracer:
    def test_construction_no_args(self) -> None:
        tracer = CrewAITracer()
        assert tracer.tracer_provider is None

    def test_construction_with_provider(self) -> None:
        sentinel = object()
        tracer = CrewAITracer(tracer_provider=sentinel)
        assert tracer.tracer_provider is sentinel

    def test_on_task_start_returns_dict(self) -> None:
        tracer = CrewAITracer()
        result = tracer.on_task_start("analyse_data")
        assert isinstance(result, dict)
        assert result["attributes"]["task.name"] == "analyse_data"

    def test_on_task_end_returns_dict(self) -> None:
        tracer = CrewAITracer()
        result = tracer.on_task_end("analyse_data", "done")
        assert isinstance(result, dict)
        assert result["attributes"]["task.name"] == "analyse_data"

    def test_on_agent_action_returns_dict(self) -> None:
        tracer = CrewAITracer()
        result = tracer.on_agent_action("researcher", "web_search")
        assert isinstance(result, dict)
        assert result["attributes"]["agent.name"] == "researcher"
        assert result["attributes"]["agent.action"] == "web_search"

    def test_on_crew_start_returns_dict(self) -> None:
        tracer = CrewAITracer()
        result = tracer.on_crew_start("MyCrew")
        assert isinstance(result, dict)
        assert result["attributes"]["crew.name"] == "MyCrew"

    def test_on_crew_end_returns_dict(self) -> None:
        tracer = CrewAITracer()
        result = tracer.on_crew_end("MyCrew")
        assert isinstance(result, dict)
        assert result["attributes"]["crew.name"] == "MyCrew"

    def test_framework_attribute(self) -> None:
        tracer = CrewAITracer()
        assert tracer.on_task_start("t")["attributes"]["framework"] == "crewai"


# ---------------------------------------------------------------------------
# OpenAIAgentsTracer
# ---------------------------------------------------------------------------


class TestOpenAIAgentsTracer:
    def test_construction_no_args(self) -> None:
        tracer = OpenAIAgentsTracer()
        assert tracer.tracer_provider is None

    def test_construction_with_provider(self) -> None:
        sentinel = object()
        tracer = OpenAIAgentsTracer(tracer_provider=sentinel)
        assert tracer.tracer_provider is sentinel

    def test_on_agent_start_returns_dict(self) -> None:
        tracer = OpenAIAgentsTracer()
        result = tracer.on_agent_start("Agent1", "Summarise this document.")
        assert isinstance(result, dict)
        assert result["attributes"]["agent.name"] == "Agent1"

    def test_on_agent_end_returns_dict(self) -> None:
        tracer = OpenAIAgentsTracer()
        result = tracer.on_agent_end("Agent1", {"summary": "..."})
        assert isinstance(result, dict)
        assert result["attributes"]["agent.name"] == "Agent1"

    def test_on_tool_call_returns_dict(self) -> None:
        tracer = OpenAIAgentsTracer()
        result = tracer.on_tool_call("web_search", {"query": "latest news"})
        assert isinstance(result, dict)
        assert result["attributes"]["tool.name"] == "web_search"
        assert result["attributes"]["tool.arg_count"] == 1

    def test_on_handoff_returns_dict(self) -> None:
        tracer = OpenAIAgentsTracer()
        result = tracer.on_handoff("AgentA", "AgentB")
        assert isinstance(result, dict)
        assert result["attributes"]["handoff.from_agent"] == "AgentA"
        assert result["attributes"]["handoff.to_agent"] == "AgentB"

    def test_framework_attribute(self) -> None:
        tracer = OpenAIAgentsTracer()
        assert tracer.on_agent_start("a", "t")["attributes"]["framework"] == "openai_agents"


# ---------------------------------------------------------------------------
# AnthropicTracer
# ---------------------------------------------------------------------------


class TestAnthropicTracer:
    def test_construction_no_args(self) -> None:
        tracer = AnthropicTracer()
        assert tracer.tracer_provider is None

    def test_construction_with_provider(self) -> None:
        sentinel = object()
        tracer = AnthropicTracer(tracer_provider=sentinel)
        assert tracer.tracer_provider is sentinel

    def test_on_message_start_returns_dict(self) -> None:
        tracer = AnthropicTracer()
        result = tracer.on_message_start("claude-sonnet-4-5", [{"role": "user", "content": "Hi"}])
        assert isinstance(result, dict)
        assert result["attributes"]["llm.model"] == "claude-sonnet-4-5"
        assert result["attributes"]["llm.message_count"] == 1

    def test_on_message_end_returns_dict(self) -> None:
        tracer = AnthropicTracer()
        result = tracer.on_message_end({"id": "msg_123"})
        assert isinstance(result, dict)

    def test_on_tool_use_returns_dict(self) -> None:
        tracer = AnthropicTracer()
        result = tracer.on_tool_use("calculator", {"expression": "2+2"})
        assert isinstance(result, dict)
        assert result["attributes"]["tool.name"] == "calculator"

    def test_on_content_block_returns_dict(self) -> None:
        tracer = AnthropicTracer()
        result = tracer.on_content_block("text", "Here is the answer.")
        assert isinstance(result, dict)
        assert result["attributes"]["content.block_type"] == "text"
        assert result["attributes"]["content.length"] == len("Here is the answer.")

    def test_framework_attribute(self) -> None:
        tracer = AnthropicTracer()
        assert tracer.on_message_start("m", [])["attributes"]["framework"] == "anthropic"


# ---------------------------------------------------------------------------
# MicrosoftAgentTracer
# ---------------------------------------------------------------------------


class TestMicrosoftAgentTracer:
    def test_construction_no_args(self) -> None:
        tracer = MicrosoftAgentTracer()
        assert tracer.tracer_provider is None

    def test_construction_with_provider(self) -> None:
        sentinel = object()
        tracer = MicrosoftAgentTracer(tracer_provider=sentinel)
        assert tracer.tracer_provider is sentinel

    def test_on_turn_start_returns_dict(self) -> None:
        tracer = MicrosoftAgentTracer()
        result = tracer.on_turn_start("turn-001")
        assert isinstance(result, dict)
        assert result["attributes"]["turn.id"] == "turn-001"

    def test_on_turn_end_returns_dict(self) -> None:
        tracer = MicrosoftAgentTracer()
        result = tracer.on_turn_end("turn-001")
        assert isinstance(result, dict)
        assert result["attributes"]["turn.id"] == "turn-001"

    def test_on_activity_returns_dict(self) -> None:
        tracer = MicrosoftAgentTracer()
        result = tracer.on_activity("message", {"text": "Hello"})
        assert isinstance(result, dict)
        assert result["attributes"]["activity.type"] == "message"

    def test_on_dialog_step_returns_dict(self) -> None:
        tracer = MicrosoftAgentTracer()
        result = tracer.on_dialog_step("main_dialog", "prompt_for_name")
        assert isinstance(result, dict)
        assert result["attributes"]["dialog.id"] == "main_dialog"
        assert result["attributes"]["dialog.step"] == "prompt_for_name"

    def test_framework_attribute(self) -> None:
        tracer = MicrosoftAgentTracer()
        assert tracer.on_turn_start("t")["attributes"]["framework"] == "microsoft_agents"

    def test_all_classes_importable_from_init(self) -> None:
        from agent_observability.adapters import (
            AnthropicTracer,
            CrewAITracer,
            LangChainTracer,
            MicrosoftAgentTracer,
            OpenAIAgentsTracer,
        )
        assert LangChainTracer is not None
        assert CrewAITracer is not None
        assert OpenAIAgentsTracer is not None
        assert AnthropicTracer is not None
        assert MicrosoftAgentTracer is not None
