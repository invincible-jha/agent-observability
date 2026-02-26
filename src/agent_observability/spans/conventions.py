"""Semantic conventions for agent observability spans.

These constants define the attribute keys used across all span types,
following OpenTelemetry semantic convention naming patterns.
"""
from __future__ import annotations

# ── Span kind ─────────────────────────────────────────────────────────────────
AGENT_SPAN_KIND: str = "agent.span.kind"

# ── LLM attributes ────────────────────────────────────────────────────────────
LLM_PROVIDER: str = "llm.provider"
LLM_MODEL: str = "llm.model"
LLM_TOKENS_INPUT: str = "llm.tokens.input"
LLM_TOKENS_OUTPUT: str = "llm.tokens.output"
LLM_TOKENS_TOTAL: str = "llm.tokens.total"
LLM_COST_USD: str = "llm.cost.usd"
LLM_TEMPERATURE: str = "llm.temperature"
LLM_MAX_TOKENS: str = "llm.max_tokens"
LLM_SYSTEM_PROMPT_HASH: str = "llm.system_prompt.hash"
LLM_RESPONSE_FINISH_REASON: str = "llm.response.finish_reason"

# ── Tool attributes ───────────────────────────────────────────────────────────
TOOL_NAME: str = "tool.name"
TOOL_INPUT_HASH: str = "tool.input.hash"
TOOL_OUTPUT_SIZE_BYTES: str = "tool.output.size_bytes"
TOOL_SUCCESS: str = "tool.success"
TOOL_ERROR_TYPE: str = "tool.error.type"

# ── Memory attributes ─────────────────────────────────────────────────────────
MEMORY_KEY: str = "memory.key"
MEMORY_OPERATION: str = "memory.operation"
MEMORY_BACKEND: str = "memory.backend"
MEMORY_HIT: str = "memory.hit"
MEMORY_SIZE_BYTES: str = "memory.size_bytes"
MEMORY_TTL_SECONDS: str = "memory.ttl_seconds"

# ── Reasoning attributes ──────────────────────────────────────────────────────
REASONING_STEP_INDEX: str = "reasoning.step.index"
REASONING_STEP_TYPE: str = "reasoning.step.type"
REASONING_CONFIDENCE: str = "reasoning.confidence"
REASONING_STRATEGY: str = "reasoning.strategy"

# ── Delegation attributes ─────────────────────────────────────────────────────
DELEGATION_TARGET_AGENT: str = "delegation.target_agent"
DELEGATION_TASK_ID: str = "delegation.task_id"
DELEGATION_STRATEGY: str = "delegation.strategy"
DELEGATION_SUCCESS: str = "delegation.success"

# ── Human approval attributes ─────────────────────────────────────────────────
HUMAN_APPROVAL_REQUESTED_BY: str = "human_approval.requested_by"
HUMAN_APPROVAL_STATUS: str = "human_approval.status"
HUMAN_APPROVAL_TIMEOUT_SECONDS: str = "human_approval.timeout_seconds"

# ── Error attributes ──────────────────────────────────────────────────────────
AGENT_ERROR_TYPE: str = "agent.error.type"
AGENT_ERROR_RECOVERABLE: str = "agent.error.recoverable"
AGENT_ERROR_RETRY_COUNT: str = "agent.error.retry_count"

# ── Agent identity ────────────────────────────────────────────────────────────
AGENT_ID: str = "agent.id"
AGENT_NAME: str = "agent.name"
AGENT_VERSION: str = "agent.version"
AGENT_FRAMEWORK: str = "agent.framework"
AGENT_SESSION_ID: str = "agent.session_id"
AGENT_TASK_ID: str = "agent.task_id"
AGENT_RUN_ID: str = "agent.run_id"
AGENT_ENVIRONMENT: str = "agent.environment"

# ── Privacy attributes ────────────────────────────────────────────────────────
PRIVACY_REDACTED: str = "privacy.redacted"
PRIVACY_REDACTION_COUNT: str = "privacy.redaction_count"
PRIVACY_JURISDICTION: str = "privacy.jurisdiction"
