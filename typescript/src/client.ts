/**
 * HTTP client for the agent-observability API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentObservabilityClient } from "@aumos/agent-observability";
 *
 * const client = createAgentObservabilityClient({ baseUrl: "http://localhost:8080" });
 *
 * const traces = await client.getTraces({ agentId: "my-agent", limit: 50 });
 * if (traces.ok) {
 *   console.log(`Found ${traces.data.total} traces`);
 * }
 *
 * const costs = await client.getCostReport({ agentId: "my-agent" });
 * if (costs.ok) {
 *   console.log(`Total cost: $${costs.data.total_cost_usd.toFixed(4)}`);
 * }
 * ```
 */

import type {
  AgentSpanKind,
  ApiError,
  ApiResult,
  CostAttribution,
  DriftReport,
  FleetStatus,
  TraceExport,
  TraceListResponse,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentObservabilityClient. */
export interface AgentObservabilityClientConfig {
  /** Base URL of the agent-observability server (e.g. "http://localhost:8080"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-observability server. */
export interface AgentObservabilityClient {
  /**
   * Retrieve a paginated list of traces with optional filtering.
   *
   * Results are ordered by timestamp descending (most recent first).
   * Use the `kind` filter to narrow to a specific agent span type such
   * as "llm.call" or "tool.invoke".
   *
   * @param options - Optional filter parameters.
   * @returns A TraceListResponse with matching traces and total count.
   */
  getTraces(options?: {
    agentId?: string;
    sessionId?: string;
    kind?: AgentSpanKind;
    since?: number;
    until?: number;
    limit?: number;
  }): Promise<ApiResult<TraceListResponse>>;

  /**
   * Retrieve aggregated cost attribution data.
   *
   * Returns cost breakdowns by model, provider, agent, and operation
   * across all LLM calls matching the filter criteria.
   *
   * @param options - Optional filter parameters.
   * @returns A CostAttribution record with per-dimension breakdowns.
   */
  getCostReport(options?: {
    agentId?: string;
    since?: number;
    until?: number;
  }): Promise<ApiResult<CostAttribution>>;

  /**
   * Run a behavioural drift analysis for an agent.
   *
   * Compares the agent's recent spans against the stored baseline using
   * Z-score analysis. Returns a DriftReport with per-feature Z-scores
   * and a qualitative severity label.
   *
   * @param agentId - The agent to analyse.
   * @param options - Optional window and threshold parameters.
   * @returns A DriftReport with drift status, Z-scores, and severity.
   */
  getDriftAnalysis(
    agentId: string,
    options?: {
      windowSpans?: number;
      sigmaThreshold?: number;
    },
  ): Promise<ApiResult<DriftReport>>;

  /**
   * Retrieve fleet-level health status for all tracked agents.
   *
   * Aggregates drift status, cost totals, and span counts per agent.
   * Useful for dashboard views and alerting on fleet-wide anomalies.
   *
   * @returns A FleetStatus record with per-agent health summaries.
   */
  getFleetStatus(): Promise<ApiResult<FleetStatus>>;

  /**
   * Export all spans for a specific trace in OTLP-compatible JSON format.
   *
   * @param traceId - The OTel trace ID (hex string).
   * @returns The full TraceExport with all child spans.
   */
  exportTraces(traceId: string): Promise<ApiResult<TraceExport>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-observability server.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentObservabilityClient instance.
 */
export function createAgentObservabilityClient(
  config: AgentObservabilityClientConfig,
): AgentObservabilityClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async getTraces(
      options: {
        agentId?: string;
        sessionId?: string;
        kind?: AgentSpanKind;
        since?: number;
        until?: number;
        limit?: number;
      } = {},
    ): Promise<ApiResult<TraceListResponse>> {
      const params = new URLSearchParams();
      if (options.agentId !== undefined) {
        params.set("agent_id", options.agentId);
      }
      if (options.sessionId !== undefined) {
        params.set("session_id", options.sessionId);
      }
      if (options.kind !== undefined) {
        params.set("kind", options.kind);
      }
      if (options.since !== undefined) {
        params.set("since", String(options.since));
      }
      if (options.until !== undefined) {
        params.set("until", String(options.until));
      }
      if (options.limit !== undefined) {
        params.set("limit", String(options.limit));
      }

      const queryString = params.toString();
      const url = queryString
        ? `${baseUrl}/traces?${queryString}`
        : `${baseUrl}/traces`;

      return fetchJson<TraceListResponse>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getCostReport(
      options: { agentId?: string; since?: number; until?: number } = {},
    ): Promise<ApiResult<CostAttribution>> {
      const params = new URLSearchParams();
      if (options.agentId !== undefined) {
        params.set("agent_id", options.agentId);
      }
      if (options.since !== undefined) {
        params.set("since", String(options.since));
      }
      if (options.until !== undefined) {
        params.set("until", String(options.until));
      }

      const queryString = params.toString();
      const url = queryString
        ? `${baseUrl}/costs?${queryString}`
        : `${baseUrl}/costs`;

      return fetchJson<CostAttribution>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getDriftAnalysis(
      agentId: string,
      options: { windowSpans?: number; sigmaThreshold?: number } = {},
    ): Promise<ApiResult<DriftReport>> {
      const params = new URLSearchParams({ agent_id: agentId });
      if (options.windowSpans !== undefined) {
        params.set("window_spans", String(options.windowSpans));
      }
      if (options.sigmaThreshold !== undefined) {
        params.set("sigma_threshold", String(options.sigmaThreshold));
      }

      return fetchJson<DriftReport>(
        `${baseUrl}/drift/analysis?${params.toString()}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getFleetStatus(): Promise<ApiResult<FleetStatus>> {
      return fetchJson<FleetStatus>(
        `${baseUrl}/fleet/status`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async exportTraces(traceId: string): Promise<ApiResult<TraceExport>> {
      return fetchJson<TraceExport>(
        `${baseUrl}/traces/${encodeURIComponent(traceId)}/export`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

