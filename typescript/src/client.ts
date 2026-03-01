/**
 * HTTP client for the agent-observability API.
 *
 * Backed by @aumos/sdk-core's createHttpClient which provides automatic retry
 * with exponential backoff, typed error hierarchy, request lifecycle events,
 * and abort signal support.
 *
 * The public API surface is unchanged — all methods still return ApiResult<T>
 * so existing callers require no migration work.
 *
 * @example
 * ```ts
 * import { createAgentObservabilityClient } from "@aumos/agent-observability";
 *
 * const client = createAgentObservabilityClient({ baseUrl: "http://localhost:8080" });
 *
 * // Observe retry events from sdk-core
 * client.events.on("request:retry", ({ payload }) => {
 *   console.warn(`Observability API retry ${payload.attempt}`);
 * });
 *
 * const traces = await client.getTraces({ agentId: "my-agent", limit: 50 });
 * if (traces.ok) {
 *   console.log(`Found ${traces.data.total} traces`);
 * }
 * ```
 */

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  RateLimitError,
  ServerError,
  ValidationError,
  AumosError,
} from "@aumos/sdk-core";

import type { HttpClient, SdkEventEmitter } from "@aumos/sdk-core";

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
  /** Optional maximum retry count. Defaults to 3. */
  readonly maxRetries?: number;
}

// ---------------------------------------------------------------------------
// Internal adapter — bridges HttpClient throws into ApiResult<T>
// ---------------------------------------------------------------------------

function extractApiError(body: unknown, fallbackMessage: string): ApiError {
  if (
    body !== null &&
    typeof body === "object" &&
    "error" in body &&
    typeof (body as Record<string, unknown>)["error"] === "string"
  ) {
    const candidate = body as Partial<{ error: string; detail: string }>;
    return {
      error: candidate.error ?? fallbackMessage,
      detail: candidate.detail ?? "",
    };
  }
  return { error: fallbackMessage, detail: "" };
}

async function executeApiCall<T>(
  call: () => Promise<T>,
): Promise<ApiResult<T>> {
  try {
    const data = await call();
    return { ok: true, data };
  } catch (error: unknown) {
    if (error instanceof RateLimitError) {
      return {
        ok: false,
        error: extractApiError(error.body, "Rate limit exceeded"),
        status: 429,
      };
    }
    if (error instanceof ValidationError) {
      return {
        ok: false,
        error: {
          error: "Validation failed",
          detail: Object.entries(error.fields)
            .map(([field, messages]) => `${field}: ${messages.join(", ")}`)
            .join("; "),
        },
        status: 422,
      };
    }
    if (error instanceof ServerError) {
      return {
        ok: false,
        error: extractApiError(error.body, `Server error: HTTP ${error.statusCode}`),
        status: error.statusCode,
      };
    }
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: extractApiError(error.body, `HTTP error: ${error.statusCode}`),
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: {
          error: "Network error",
          detail: error instanceof Error ? error.message : String(error),
        },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unknown error", detail: message },
      status: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-observability server. */
export interface AgentObservabilityClient {
  /**
   * Typed event emitter exposed from the underlying sdk-core HttpClient.
   * Attach listeners here to observe request lifecycle, retries, and errors.
   *
   * @example
   * ```ts
   * client.events.on("request:retry", ({ payload }) => {
   *   console.warn(`Retry attempt ${payload.attempt}, delay ${payload.delayMs}ms`);
   * });
   * ```
   */
  readonly events: SdkEventEmitter;

  /**
   * Retrieve a paginated list of traces with optional filtering.
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
 * Internally uses @aumos/sdk-core's createHttpClient for automatic retry,
 * typed errors, and request lifecycle events. The public API remains identical
 * to the previous version — all methods return ApiResult<T>.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentObservabilityClient instance.
 */
export function createAgentObservabilityClient(
  config: AgentObservabilityClientConfig,
): AgentObservabilityClient {
  const httpClient: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    maxRetries: config.maxRetries ?? 3,
    defaultHeaders: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(config.headers as Record<string, string> | undefined),
    },
  });

  return {
    events: httpClient.events,

    getTraces(
      options: {
        agentId?: string;
        sessionId?: string;
        kind?: AgentSpanKind;
        since?: number;
        until?: number;
        limit?: number;
      } = {},
    ): Promise<ApiResult<TraceListResponse>> {
      const queryParams: Record<string, string> = {};
      if (options.agentId !== undefined) {
        queryParams["agent_id"] = options.agentId;
      }
      if (options.sessionId !== undefined) {
        queryParams["session_id"] = options.sessionId;
      }
      if (options.kind !== undefined) {
        queryParams["kind"] = options.kind;
      }
      if (options.since !== undefined) {
        queryParams["since"] = String(options.since);
      }
      if (options.until !== undefined) {
        queryParams["until"] = String(options.until);
      }
      if (options.limit !== undefined) {
        queryParams["limit"] = String(options.limit);
      }

      return executeApiCall(() =>
        httpClient
          .get<TraceListResponse>("/traces", { queryParams })
          .then((r) => r.data),
      );
    },

    getCostReport(
      options: { agentId?: string; since?: number; until?: number } = {},
    ): Promise<ApiResult<CostAttribution>> {
      const queryParams: Record<string, string> = {};
      if (options.agentId !== undefined) {
        queryParams["agent_id"] = options.agentId;
      }
      if (options.since !== undefined) {
        queryParams["since"] = String(options.since);
      }
      if (options.until !== undefined) {
        queryParams["until"] = String(options.until);
      }

      return executeApiCall(() =>
        httpClient
          .get<CostAttribution>("/costs", { queryParams })
          .then((r) => r.data),
      );
    },

    getDriftAnalysis(
      agentId: string,
      options: { windowSpans?: number; sigmaThreshold?: number } = {},
    ): Promise<ApiResult<DriftReport>> {
      const queryParams: Record<string, string> = { agent_id: agentId };
      if (options.windowSpans !== undefined) {
        queryParams["window_spans"] = String(options.windowSpans);
      }
      if (options.sigmaThreshold !== undefined) {
        queryParams["sigma_threshold"] = String(options.sigmaThreshold);
      }

      return executeApiCall(() =>
        httpClient
          .get<DriftReport>("/drift/analysis", { queryParams })
          .then((r) => r.data),
      );
    },

    getFleetStatus(): Promise<ApiResult<FleetStatus>> {
      return executeApiCall(() =>
        httpClient.get<FleetStatus>("/fleet/status").then((r) => r.data),
      );
    },

    exportTraces(traceId: string): Promise<ApiResult<TraceExport>> {
      return executeApiCall(() =>
        httpClient
          .get<TraceExport>(
            `/traces/${encodeURIComponent(traceId)}/export`,
          )
          .then((r) => r.data),
      );
    },
  };
}
