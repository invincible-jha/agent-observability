/**
 * @aumos/agent-observability
 *
 * TypeScript client for the AumOS agent-observability framework.
 * Provides distributed tracing, cost attribution, drift detection,
 * and fleet health monitoring for AI agents.
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
