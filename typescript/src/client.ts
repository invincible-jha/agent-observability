/**
 * HTTP client for the agent-observability server mode API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createObservabilityClient } from "@aumos/agent-observability-client";
 *
 * const client = createObservabilityClient({ baseUrl: "http://localhost:8080" });
 *
 * const result = await client.createTrace({
 *   agent_id: "my-agent",
 *   provider: "openai",
 *   model: "gpt-4o",
 *   input_tokens: 500,
 *   output_tokens: 200,
 * });
 *
 * if (result.ok) {
 *   console.log("Trace created:", result.data.trace_id);
 * }
 * ```
 */

import type {
  ApiError,
  ApiResult,
  CostSummary,
  CreateTraceRequest,
  HealthStatus,
  Trace,
  TraceListResponse,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the ObservabilityClient. */
export interface ObservabilityClientConfig {
  /** Base URL of the agent-observability server (e.g. "http://localhost:8080"). */
  readonly baseUrl: string;

  /** Optional request timeout in milliseconds (default: 10000). */
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
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "Accept": "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-observability server. */
export interface ObservabilityClient {
  /**
   * Create a new trace record.
   *
   * @param request - Trace creation parameters.
   * @returns The created trace with its assigned trace_id.
   */
  createTrace(request: CreateTraceRequest): Promise<ApiResult<Trace>>;

  /**
   * Retrieve a single trace by ID.
   *
   * @param traceId - The trace identifier.
   * @returns The full trace record.
   */
  getTrace(traceId: string): Promise<ApiResult<Trace>>;

  /**
   * List traces with optional filtering.
   *
   * @param options - Optional filter parameters.
   * @returns Paginated list of traces.
   */
  listTraces(options?: {
    agentId?: string;
    limit?: number;
  }): Promise<ApiResult<TraceListResponse>>;

  /**
   * Retrieve aggregated cost summary.
   *
   * @param options - Optional filter parameters.
   * @returns Aggregated cost summary.
   */
  getCosts(options?: {
    agentId?: string;
    since?: number;
    until?: number;
  }): Promise<ApiResult<CostSummary>>;

  /**
   * Check server health.
   *
   * @returns Health status of the agent-observability server.
   */
  health(): Promise<ApiResult<HealthStatus>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-observability server.
 *
 * @param config - Client configuration including base URL.
 * @returns An ObservabilityClient instance.
 */
export function createObservabilityClient(
  config: ObservabilityClientConfig
): ObservabilityClient {
  const { baseUrl, timeoutMs = 10_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async createTrace(request: CreateTraceRequest): Promise<ApiResult<Trace>> {
      return fetchJson<Trace>(
        `${baseUrl}/traces`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async getTrace(traceId: string): Promise<ApiResult<Trace>> {
      return fetchJson<Trace>(
        `${baseUrl}/traces/${encodeURIComponent(traceId)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async listTraces(
      options: { agentId?: string; limit?: number } = {}
    ): Promise<ApiResult<TraceListResponse>> {
      const params = new URLSearchParams();
      if (options.agentId !== undefined) params.set("agent_id", options.agentId);
      if (options.limit !== undefined) params.set("limit", String(options.limit));

      const queryString = params.toString();
      const url = queryString ? `${baseUrl}/traces?${queryString}` : `${baseUrl}/traces`;

      return fetchJson<TraceListResponse>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getCosts(
      options: { agentId?: string; since?: number; until?: number } = {}
    ): Promise<ApiResult<CostSummary>> {
      const params = new URLSearchParams();
      if (options.agentId !== undefined) params.set("agent_id", options.agentId);
      if (options.since !== undefined) params.set("since", String(options.since));
      if (options.until !== undefined) params.set("until", String(options.until));

      const queryString = params.toString();
      const url = queryString ? `${baseUrl}/costs?${queryString}` : `${baseUrl}/costs`;

      return fetchJson<CostSummary>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async health(): Promise<ApiResult<HealthStatus>> {
      return fetchJson<HealthStatus>(
        `${baseUrl}/health`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

/** Type alias re-export for convenience. */
export type { CreateTraceRequest, Trace, TraceListResponse, CostSummary, HealthStatus };
