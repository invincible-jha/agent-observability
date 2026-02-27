"""OpenTelemetry GenAI Semantic Convention constants (v1.37+).

Maps agent-observability span types to official OTel GenAI conventions.
Includes proposed extensions for agent-specific operations not yet
covered by the official specification.
"""
from __future__ import annotations

from enum import Enum


class OTelGenAISpanName(str, Enum):
    """Official OTel GenAI span names (v1.37)."""

    CHAT = "gen_ai.chat"
    EXECUTE_TOOL = "gen_ai.execute_tool"
    INVOKE_AGENT = "invoke_agent"


class AumOSSpanExtension(str, Enum):
    """Proposed AumOS extensions to OTel GenAI conventions.

    These span types fill gaps in the current OTel specification
    for agent-specific operations. Submitted as convention proposals.
    """

    MEMORY_READ = "gen_ai.agent.memory.read"
    MEMORY_WRITE = "gen_ai.agent.memory.write"
    REASONING_STEP = "gen_ai.agent.reasoning.step"
    HUMAN_APPROVAL = "gen_ai.agent.human_approval"
    AGENT_ERROR = "gen_ai.agent.error"


# Mapping from legacy span types to OTel conventions
LEGACY_TO_OTEL: dict[str, str] = {
    "llm.call": OTelGenAISpanName.CHAT.value,
    "tool.invoke": OTelGenAISpanName.EXECUTE_TOOL.value,
    "agent.delegate": OTelGenAISpanName.INVOKE_AGENT.value,
    "memory.read": AumOSSpanExtension.MEMORY_READ.value,
    "memory.write": AumOSSpanExtension.MEMORY_WRITE.value,
    "reasoning.step": AumOSSpanExtension.REASONING_STEP.value,
    "human.approval": AumOSSpanExtension.HUMAN_APPROVAL.value,
    "agent.error": AumOSSpanExtension.AGENT_ERROR.value,
}


class GenAIAttributes:
    """Standard OTel GenAI semantic attributes."""

    SYSTEM = "gen_ai.system"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_DESCRIPTION = "gen_ai.tool.description"


class AumOSAttributes:
    """AumOS extension attributes for agent observability."""

    COST_USD = "aumos.cost.usd"
    MEMORY_LAYER = "aumos.memory.layer"
    MEMORY_KEY = "aumos.memory.key"
    REASONING_CONFIDENCE = "aumos.reasoning.confidence"
    APPROVAL_REQUIRED = "aumos.approval.required"
    APPROVAL_GRANTED = "aumos.approval.granted"
    ERROR_CATEGORY = "aumos.error.category"
    DRIFT_SCORE = "aumos.drift.score"
