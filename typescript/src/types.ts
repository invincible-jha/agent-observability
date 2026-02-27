/**
 * TypeScript interfaces for agent-observability server API types.
 *
 * Mirrors the Pydantic models defined in:
 *   agent_observability.server.models
 */

// ---------------------------------------------------------------------------
// Trace types
// ---------------------------------------------------------------------------

/** Request body for POST /traces. */
export interface CreateTraceRequest {
  agent_id?: string;
  session_id?: string;
  task_id?: string;
  service_name?: string;
  provider?: string;
  model?: string;
  input_tokens?: number;
  output_tokens?: number;
  cached_input_tokens?: number;
  cost_usd?: number | null;
  operation?: string;
  tags?: Record<string, string>;
}

/** A single span within a trace response. */
export interface SpanData {
  readonly span_id: string;
  readonly name: string;
  readonly kind: string;
  readonly attributes: Readonly<Record<string, string | number | boolean>>;
}

/** Response body for a single trace. */
export interface Trace {
  readonly trace_id: string;
  readonly agent_id: string;
  readonly session_id: string;
  readonly task_id: string;
  readonly service_name: string;
  readonly provider: string;
  readonly model: string;
  readonly input_tokens: number;
  readonly output_tokens: number;
  readonly cached_input_tokens: number;
  readonly cost_usd: number;
  readonly operation: string;
  readonly tags: Readonly<Record<string, string>>;
  readonly timestamp: number;
  readonly spans: readonly SpanData[];
}

/** Response body for the trace list endpoint. */
export interface TraceListResponse {
  readonly traces: readonly Trace[];
  readonly total: number;
}

// ---------------------------------------------------------------------------
// Cost types
// ---------------------------------------------------------------------------

/** Aggregated cost summary. */
export interface CostSummary {
  readonly total_cost_usd: number;
  readonly total_input_tokens: number;
  readonly total_output_tokens: number;
  readonly total_tokens: number;
  readonly record_count: number;
  readonly by_model: Readonly<Record<string, number>>;
  readonly by_provider: Readonly<Record<string, number>>;
  readonly by_agent: Readonly<Record<string, number>>;
  readonly by_operation: Readonly<Record<string, number>>;
}

// ---------------------------------------------------------------------------
// Health types
// ---------------------------------------------------------------------------

/** Health check response. */
export interface HealthStatus {
  readonly status: "ok" | "degraded";
  readonly service: string;
  readonly version: string;
  readonly trace_count: number;
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/** Standard error response. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
