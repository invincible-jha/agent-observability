/**
 * @aumos/agent-observability-client
 *
 * TypeScript HTTP client for the agent-observability server mode API.
 */

export type {
  ObservabilityClientConfig,
  ObservabilityClient,
  CreateTraceRequest,
  Trace,
  TraceListResponse,
  CostSummary,
  HealthStatus,
} from "./client.js";

export { createObservabilityClient } from "./client.js";

export type {
  SpanData,
  ApiError,
  ApiResult,
} from "./types.js";
