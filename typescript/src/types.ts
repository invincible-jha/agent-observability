/**
 * TypeScript interfaces for the agent-observability framework.
 *
 * Mirrors the Python models defined in:
 *   agent_observability.spans.types     — AgentSpanKind, AgentSpan
 *   agent_observability.cost.tracker    — CostRecord, CostSummary
 *   agent_observability.drift.detector  — DriftResult
 *   agent_observability.server.models   — Trace, TraceListResponse
 *
 * All interfaces use readonly fields to match Python frozen models.
 */

// ---------------------------------------------------------------------------
// Agent span semantic types
// ---------------------------------------------------------------------------

/**
 * The 8 semantic span kinds for agent observability.
 * Maps to AgentSpanKind enum in agent_observability.spans.types.
 *
 * Span kind values follow the dot-separated naming convention used
 * by the OpenTelemetry semantic conventions for agents:
 *   - llm.call        — a single LLM request/response cycle
 *   - tool.invoke     — an external tool or function call
 *   - memory.read     — reading from any memory backend
 *   - memory.write    — writing to any memory backend
 *   - reasoning.step  — a single step in a chain-of-thought
 *   - agent.delegate  — delegating a sub-task to another agent
 *   - human.approval  — a human-in-the-loop approval gate
 *   - agent.error     — a recoverable or unrecoverable agent error
 */
export type AgentSpanKind =
  | "llm.call"
  | "tool.invoke"
  | "memory.read"
  | "memory.write"
  | "reasoning.step"
  | "agent.delegate"
  | "human.approval"
  | "agent.error";

/**
 * A single span within a trace, enriched with agent-semantic attributes.
 * Combines the OTel span structure with agent-specific metadata.
 */
export interface AgentSpan {
  /** OTel span ID (hex string). */
  readonly span_id: string;
  /** OTel trace ID this span belongs to (hex string). */
  readonly trace_id: string;
  /** Human-readable span name (e.g. "llm.call", "tool.invoke"). */
  readonly name: string;
  /** One of the 8 agent-semantic span kinds. */
  readonly kind: AgentSpanKind;
  /** Epoch milliseconds when the span started. */
  readonly start_time_ms: number;
  /** Epoch milliseconds when the span ended (null if still active). */
  readonly end_time_ms: number | null;
  /** Wall-clock duration in milliseconds (null if still active). */
  readonly duration_ms: number | null;
  /** Agent ID attached to this span. */
  readonly agent_id: string;
  /** Session ID attached to this span. */
  readonly session_id: string;
  /** Framework name (e.g. "langchain", "crewai"). */
  readonly framework: string;
  /** All OTel and agent-semantic attributes on the span. */
  readonly attributes: Readonly<Record<string, string | number | boolean>>;
}

// ---------------------------------------------------------------------------
// Trace export
// ---------------------------------------------------------------------------

/**
 * A complete exported trace — one root trace with all child spans.
 * Maps to the trace response body from the observability server.
 */
export interface TraceExport {
  /** OTel trace ID (hex string). */
  readonly trace_id: string;
  /** Primary agent ID for this trace. */
  readonly agent_id: string;
  /** Session identifier. */
  readonly session_id: string;
  /** Task or run identifier. */
  readonly task_id: string;
  /** Service or framework name. */
  readonly service_name: string;
  /** Ordered list of spans in this trace (root span first). */
  readonly spans: readonly AgentSpan[];
  /** Epoch unix timestamp (seconds) when the trace was recorded. */
  readonly timestamp: number;
  /** Arbitrary metadata attached to the trace. */
  readonly tags: Readonly<Record<string, string>>;
}

/**
 * Response body for the list-traces endpoint.
 */
export interface TraceListResponse {
  readonly traces: readonly TraceExport[];
  readonly total: number;
}

// ---------------------------------------------------------------------------
// Cost attribution
// ---------------------------------------------------------------------------

/**
 * A single recorded LLM cost event.
 * Maps to CostRecord in agent_observability.cost.tracker.
 */
export interface CostRecord {
  /** Unix epoch timestamp (seconds) of this record. */
  readonly timestamp: number;
  /** Agent that made this LLM call. */
  readonly agent_id: string;
  /** Session this call belongs to. */
  readonly session_id: string;
  /** Task or run identifier. */
  readonly task_id: string;
  /** LLM provider name (e.g. "anthropic", "openai"). */
  readonly provider: string;
  /** Model name (e.g. "claude-sonnet-4-6", "gpt-4o"). */
  readonly model: string;
  /** Number of prompt/input tokens consumed. */
  readonly input_tokens: number;
  /** Number of completion/output tokens generated. */
  readonly output_tokens: number;
  /** Cached prompt tokens (charged at reduced rate by some providers). */
  readonly cached_input_tokens: number;
  /** Computed cost in USD. */
  readonly cost_usd: number;
  /** Semantic label for the operation (e.g. "llm_call", "embedding"). */
  readonly operation: string;
  /** OTel trace ID hex string. */
  readonly trace_id: string;
  /** OTel span ID hex string. */
  readonly span_id: string;
  /** Arbitrary key/value metadata. */
  readonly tags: Readonly<Record<string, string>>;
}

/**
 * Aggregated cost summary over a collection of CostRecord instances.
 * Maps to CostSummary in agent_observability.cost.tracker.
 */
export interface CostAttribution {
  /** Total cost in USD across all matching records. */
  readonly total_cost_usd: number;
  /** Total prompt/input tokens consumed. */
  readonly total_input_tokens: number;
  /** Total completion/output tokens generated. */
  readonly total_output_tokens: number;
  /** Combined token count. */
  readonly total_tokens: number;
  /** Number of cost records contributing to this summary. */
  readonly record_count: number;
  /** Cost breakdown by model name. */
  readonly by_model: Readonly<Record<string, number>>;
  /** Cost breakdown by provider name. */
  readonly by_provider: Readonly<Record<string, number>>;
  /** Cost breakdown by agent ID. */
  readonly by_agent: Readonly<Record<string, number>>;
  /** Cost breakdown by operation type. */
  readonly by_operation: Readonly<Record<string, number>>;
}

// ---------------------------------------------------------------------------
// Drift detection
// ---------------------------------------------------------------------------

/**
 * Qualitative severity label for a drift event.
 */
export type DriftSeverity = "none" | "low" | "medium" | "high";

/**
 * Result of a single agent behaviour drift check.
 * Maps to DriftResult in agent_observability.drift.detector.
 */
export interface DriftReport {
  /** Agent ID that was checked. */
  readonly agent_id: string;
  /** Unix epoch timestamp (seconds) when the check ran. */
  readonly timestamp: number;
  /** True when at least one feature exceeded the Z-score threshold. */
  readonly drifted: boolean;
  /** Maximum absolute Z-score across all behavioural features. */
  readonly max_z_score: number;
  /** Configured sigma threshold used in this check. */
  readonly threshold: number;
  /** Feature names and their Z-scores for features that exceeded threshold. */
  readonly drifted_features: Readonly<Record<string, number>>;
  /** Z-scores for all behavioural features checked. */
  readonly all_z_scores: Readonly<Record<string, number>>;
  /** Age of the baseline in seconds at check time. */
  readonly baseline_age_seconds: number;
  /** Number of spans in the evaluation window. */
  readonly window_span_count: number;
  /** Qualitative severity label derived from max_z_score vs. threshold. */
  readonly severity: DriftSeverity;
  /** Optional human-readable notes (e.g. skip reason). */
  readonly notes: string;
}

// ---------------------------------------------------------------------------
// Fleet status
// ---------------------------------------------------------------------------

/**
 * Per-agent health summary within the fleet status response.
 */
export interface AgentHealthSummary {
  /** Agent identifier. */
  readonly agent_id: string;
  /** Whether this agent is currently drifting. */
  readonly drifting: boolean;
  /** Drift severity for this agent. */
  readonly drift_severity: DriftSeverity;
  /** Total cost attributed to this agent in USD. */
  readonly total_cost_usd: number;
  /** Total number of spans recorded for this agent. */
  readonly span_count: number;
  /** ISO-8601 UTC timestamp of the most recent span. */
  readonly last_seen: string;
}

/**
 * Fleet-level health status aggregating all tracked agents.
 */
export interface FleetStatus {
  /** Total number of agents tracked. */
  readonly agent_count: number;
  /** Number of agents currently drifting. */
  readonly drifting_count: number;
  /** Cumulative cost across all agents in USD. */
  readonly total_cost_usd: number;
  /** Total span count across all agents. */
  readonly total_span_count: number;
  /** Per-agent health summaries. */
  readonly agents: readonly AgentHealthSummary[];
  /** ISO-8601 UTC timestamp when this status was generated. */
  readonly generated_at: string;
}

// ---------------------------------------------------------------------------
// Error and result wrapper
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-observability API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
