/**
 * @aumos/agent-observability
 *
 * TypeScript client for the AumOS agent-observability framework.
 * Provides distributed tracing, cost attribution, drift detection,
 * and fleet health monitoring for AI agents.
 *
 * The client is now backed by @aumos/sdk-core for automatic retry,
 * typed error hierarchy, and request lifecycle events.
 */

// Client and configuration
export type {
  AgentObservabilityClient,
  AgentObservabilityClientConfig,
} from "./client.js";
export { createAgentObservabilityClient } from "./client.js";

// Core observability types
export type {
  AgentSpanKind,
  AgentSpan,
  TraceExport,
  TraceListResponse,
  CostRecord,
  CostAttribution,
  DriftSeverity,
  DriftReport,
  AgentHealthSummary,
  FleetStatus,
  ApiError,
  ApiResult,
} from "./types.js";

// Re-export sdk-core error hierarchy for callers that want to instanceof-check
export {
  AumosError,
  NetworkError,
  TimeoutError,
  HttpError,
  RateLimitError,
  ValidationError,
  ServerError,
  AbortError,
} from "@aumos/sdk-core";

// Re-export event emitter type for listeners attached via client.events
export type { SdkEventEmitter, SdkEventMap } from "@aumos/sdk-core";
