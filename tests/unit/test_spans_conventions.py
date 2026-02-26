"""Unit tests for spans.conventions — semantic attribute key constants."""
from __future__ import annotations

import agent_observability.spans.conventions as conv


class TestLlmAttributeKeys:
    def test_agent_span_kind_key(self) -> None:
        assert conv.AGENT_SPAN_KIND == "agent.span.kind"

    def test_llm_provider_key(self) -> None:
        assert conv.LLM_PROVIDER == "llm.provider"

    def test_llm_model_key(self) -> None:
        assert conv.LLM_MODEL == "llm.model"

    def test_llm_tokens_input_key(self) -> None:
        assert conv.LLM_TOKENS_INPUT == "llm.tokens.input"

    def test_llm_tokens_output_key(self) -> None:
        assert conv.LLM_TOKENS_OUTPUT == "llm.tokens.output"

    def test_llm_tokens_total_key(self) -> None:
        assert conv.LLM_TOKENS_TOTAL == "llm.tokens.total"

    def test_llm_cost_usd_key(self) -> None:
        assert conv.LLM_COST_USD == "llm.cost.usd"

    def test_llm_temperature_key(self) -> None:
        assert conv.LLM_TEMPERATURE == "llm.temperature"

    def test_llm_system_prompt_hash_key(self) -> None:
        assert conv.LLM_SYSTEM_PROMPT_HASH == "llm.system_prompt.hash"

    def test_llm_response_finish_reason_key(self) -> None:
        assert conv.LLM_RESPONSE_FINISH_REASON == "llm.response.finish_reason"


class TestToolAttributeKeys:
    def test_tool_name_key(self) -> None:
        assert conv.TOOL_NAME == "tool.name"

    def test_tool_input_hash_key(self) -> None:
        assert conv.TOOL_INPUT_HASH == "tool.input.hash"

    def test_tool_output_size_bytes_key(self) -> None:
        assert conv.TOOL_OUTPUT_SIZE_BYTES == "tool.output.size_bytes"

    def test_tool_success_key(self) -> None:
        assert conv.TOOL_SUCCESS == "tool.success"

    def test_tool_error_type_key(self) -> None:
        assert conv.TOOL_ERROR_TYPE == "tool.error.type"


class TestMemoryAttributeKeys:
    def test_memory_key(self) -> None:
        assert conv.MEMORY_KEY == "memory.key"

    def test_memory_operation_key(self) -> None:
        assert conv.MEMORY_OPERATION == "memory.operation"

    def test_memory_backend_key(self) -> None:
        assert conv.MEMORY_BACKEND == "memory.backend"

    def test_memory_hit_key(self) -> None:
        assert conv.MEMORY_HIT == "memory.hit"


class TestAgentIdentityKeys:
    def test_agent_id_key(self) -> None:
        assert conv.AGENT_ID == "agent.id"

    def test_agent_name_key(self) -> None:
        assert conv.AGENT_NAME == "agent.name"

    def test_agent_framework_key(self) -> None:
        assert conv.AGENT_FRAMEWORK == "agent.framework"

    def test_agent_session_id_key(self) -> None:
        assert conv.AGENT_SESSION_ID == "agent.session_id"

    def test_agent_environment_key(self) -> None:
        assert conv.AGENT_ENVIRONMENT == "agent.environment"


class TestPrivacyAttributeKeys:
    def test_privacy_redacted_key(self) -> None:
        assert conv.PRIVACY_REDACTED == "privacy.redacted"

    def test_privacy_redaction_count_key(self) -> None:
        assert conv.PRIVACY_REDACTION_COUNT == "privacy.redaction_count"

    def test_privacy_jurisdiction_key(self) -> None:
        assert conv.PRIVACY_JURISDICTION == "privacy.jurisdiction"


class TestDelegationAttributeKeys:
    def test_delegation_target_agent_key(self) -> None:
        assert conv.DELEGATION_TARGET_AGENT == "delegation.target_agent"

    def test_delegation_task_id_key(self) -> None:
        assert conv.DELEGATION_TASK_ID == "delegation.task_id"

    def test_delegation_strategy_key(self) -> None:
        assert conv.DELEGATION_STRATEGY == "delegation.strategy"
