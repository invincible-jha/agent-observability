"""Tests for OTel GenAI convention alignment."""
from __future__ import annotations

import pytest
from agent_observability.conventions.semantic import (
    LEGACY_TO_OTEL,
    AumOSAttributes,
    AumOSSpanExtension,
    GenAIAttributes,
    OTelGenAISpanName,
)


class TestOTelConventionMapping:
    """Verify legacy-to-OTel span type mapping."""

    def test_all_legacy_types_mapped(self) -> None:
        legacy_types = {
            "llm.call",
            "tool.invoke",
            "agent.delegate",
            "memory.read",
            "memory.write",
            "reasoning.step",
            "human.approval",
            "agent.error",
        }
        assert set(LEGACY_TO_OTEL.keys()) == legacy_types

    def test_official_otel_mappings(self) -> None:
        assert LEGACY_TO_OTEL["llm.call"] == "gen_ai.chat"
        assert LEGACY_TO_OTEL["tool.invoke"] == "gen_ai.execute_tool"
        assert LEGACY_TO_OTEL["agent.delegate"] == "invoke_agent"

    def test_extension_mappings_use_namespace(self) -> None:
        for key in [
            "memory.read",
            "memory.write",
            "reasoning.step",
            "human.approval",
            "agent.error",
        ]:
            assert LEGACY_TO_OTEL[key].startswith("gen_ai.agent.")

    def test_genai_attributes_match_spec(self) -> None:
        assert GenAIAttributes.SYSTEM == "gen_ai.system"
        assert GenAIAttributes.AGENT_NAME == "gen_ai.agent.name"
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"

    def test_otel_span_names_are_strings(self) -> None:
        for member in OTelGenAISpanName:
            assert isinstance(member.value, str)

    def test_aumos_extensions_are_strings(self) -> None:
        for member in AumOSSpanExtension:
            assert isinstance(member.value, str)

    def test_aumos_attributes_are_strings(self) -> None:
        for attr in [
            AumOSAttributes.COST_USD,
            AumOSAttributes.MEMORY_LAYER,
            AumOSAttributes.DRIFT_SCORE,
        ]:
            assert isinstance(attr, str)
            assert attr.startswith("aumos.")

    def test_legacy_mapping_is_complete(self) -> None:
        """Every legacy span type should have a non-empty OTel mapping."""
        for legacy, otel in LEGACY_TO_OTEL.items():
            assert otel, f"Empty OTel mapping for legacy span type: {legacy}"

    def test_no_duplicate_otel_values(self) -> None:
        """No two legacy types should map to the same OTel convention."""
        values = list(LEGACY_TO_OTEL.values())
        assert len(values) == len(set(values)), "Duplicate OTel mappings found"


class TestOTelSpanNameEnum:
    """Verify OTelGenAISpanName enum values match the spec."""

    def test_chat_span_name(self) -> None:
        assert OTelGenAISpanName.CHAT == "gen_ai.chat"
        assert OTelGenAISpanName.CHAT.value == "gen_ai.chat"

    def test_execute_tool_span_name(self) -> None:
        assert OTelGenAISpanName.EXECUTE_TOOL == "gen_ai.execute_tool"

    def test_invoke_agent_span_name(self) -> None:
        assert OTelGenAISpanName.INVOKE_AGENT == "invoke_agent"

    def test_enum_members_count(self) -> None:
        assert len(OTelGenAISpanName) == 3

    def test_enum_is_str_subclass(self) -> None:
        assert issubclass(OTelGenAISpanName, str)

    def test_enum_string_comparison(self) -> None:
        """str(Enum) should equal the value for str-based enums."""
        assert OTelGenAISpanName.CHAT == OTelGenAISpanName.CHAT.value


class TestAumOSSpanExtensionEnum:
    """Verify AumOSSpanExtension enum values use gen_ai.agent namespace."""

    def test_memory_read(self) -> None:
        assert AumOSSpanExtension.MEMORY_READ == "gen_ai.agent.memory.read"

    def test_memory_write(self) -> None:
        assert AumOSSpanExtension.MEMORY_WRITE == "gen_ai.agent.memory.write"

    def test_reasoning_step(self) -> None:
        assert AumOSSpanExtension.REASONING_STEP == "gen_ai.agent.reasoning.step"

    def test_human_approval(self) -> None:
        assert AumOSSpanExtension.HUMAN_APPROVAL == "gen_ai.agent.human_approval"

    def test_agent_error(self) -> None:
        assert AumOSSpanExtension.AGENT_ERROR == "gen_ai.agent.error"

    def test_all_extensions_use_agent_namespace(self) -> None:
        for member in AumOSSpanExtension:
            assert member.value.startswith("gen_ai.agent."), (
                f"{member.name} does not use gen_ai.agent namespace: {member.value}"
            )

    def test_enum_members_count(self) -> None:
        assert len(AumOSSpanExtension) == 5


class TestGenAIAttributes:
    """Verify GenAIAttributes class constants match the OTel spec."""

    def test_system(self) -> None:
        assert GenAIAttributes.SYSTEM == "gen_ai.system"

    def test_request_model(self) -> None:
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"

    def test_request_max_tokens(self) -> None:
        assert GenAIAttributes.REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"

    def test_request_temperature(self) -> None:
        assert GenAIAttributes.REQUEST_TEMPERATURE == "gen_ai.request.temperature"

    def test_response_model(self) -> None:
        assert GenAIAttributes.RESPONSE_MODEL == "gen_ai.response.model"

    def test_response_finish_reasons(self) -> None:
        assert GenAIAttributes.RESPONSE_FINISH_REASONS == "gen_ai.response.finish_reasons"

    def test_usage_input_tokens(self) -> None:
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"

    def test_usage_output_tokens(self) -> None:
        assert GenAIAttributes.USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"

    def test_agent_name(self) -> None:
        assert GenAIAttributes.AGENT_NAME == "gen_ai.agent.name"

    def test_agent_description(self) -> None:
        assert GenAIAttributes.AGENT_DESCRIPTION == "gen_ai.agent.description"

    def test_tool_name(self) -> None:
        assert GenAIAttributes.TOOL_NAME == "gen_ai.tool.name"

    def test_tool_description(self) -> None:
        assert GenAIAttributes.TOOL_DESCRIPTION == "gen_ai.tool.description"

    def test_all_attributes_use_gen_ai_prefix(self) -> None:
        attrs = [
            name
            for name in dir(GenAIAttributes)
            if not name.startswith("_")
        ]
        for attr_name in attrs:
            value = getattr(GenAIAttributes, attr_name)
            assert isinstance(value, str), f"{attr_name} is not a string"
            assert value.startswith("gen_ai."), f"{attr_name} = {value!r} missing gen_ai. prefix"


class TestAumOSAttributes:
    """Verify AumOSAttributes class constants use the aumos namespace."""

    def test_cost_usd(self) -> None:
        assert AumOSAttributes.COST_USD == "aumos.cost.usd"

    def test_memory_layer(self) -> None:
        assert AumOSAttributes.MEMORY_LAYER == "aumos.memory.layer"

    def test_memory_key(self) -> None:
        assert AumOSAttributes.MEMORY_KEY == "aumos.memory.key"

    def test_reasoning_confidence(self) -> None:
        assert AumOSAttributes.REASONING_CONFIDENCE == "aumos.reasoning.confidence"

    def test_approval_required(self) -> None:
        assert AumOSAttributes.APPROVAL_REQUIRED == "aumos.approval.required"

    def test_approval_granted(self) -> None:
        assert AumOSAttributes.APPROVAL_GRANTED == "aumos.approval.granted"

    def test_error_category(self) -> None:
        assert AumOSAttributes.ERROR_CATEGORY == "aumos.error.category"

    def test_drift_score(self) -> None:
        assert AumOSAttributes.DRIFT_SCORE == "aumos.drift.score"

    def test_all_attributes_use_aumos_prefix(self) -> None:
        attrs = [name for name in dir(AumOSAttributes) if not name.startswith("_")]
        for attr_name in attrs:
            value = getattr(AumOSAttributes, attr_name)
            assert isinstance(value, str), f"{attr_name} is not a string"
            assert value.startswith("aumos."), f"{attr_name} = {value!r} missing aumos. prefix"
