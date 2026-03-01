/**
 * Tests for @aumos/agent-observability client.
 *
 * Covers:
 * - getTraces, getCostReport, getDriftAnalysis, getFleetStatus, exportTraces
 * - sdk-core error hierarchy: HttpError, NetworkError, RateLimitError, ServerError
 * - Request lifecycle events and retry behavior
 * - Query parameter construction for all filter options
 * - Backward compatibility of ApiResult<T> shape
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createAgentObservabilityClient } from "../src/client.js";
import type { TraceListResponse, CostAttribution, DriftReport, FleetStatus, TraceExport } from "../src/types.js";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function makeSuccessResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-type" ? "application/json" : null,
      forEach: (cb: (v: string, k: string) => void) => {
        cb("application/json", "content-type");
      },
    },
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

function makeErrorResponse(
  status: number,
  body: unknown,
  extraHeaders: Record<string, string> = {},
) {
  return {
    ok: false,
    status,
    statusText: `Error ${status}`,
    headers: {
      get: (name: string) => {
        if (name.toLowerCase() === "content-type") return "application/json";
        return extraHeaders[name.toLowerCase()] ?? null;
      },
      forEach: (cb: (v: string, k: string) => void) => {
        cb("application/json", "content-type");
        for (const [k, v] of Object.entries(extraHeaders)) {
          cb(v, k);
        }
      },
    },
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

const BASE_URL = "http://localhost:18080";

const SAMPLE_TRACE_LIST: TraceListResponse = {
  traces: [],
  total: 0,
};

const SAMPLE_COST: CostAttribution = {
  total_cost_usd: 1.23,
  total_input_tokens: 10000,
  total_output_tokens: 5000,
  total_tokens: 15000,
  record_count: 42,
  by_model: { "claude-sonnet-4-6": 1.23 },
  by_provider: { anthropic: 1.23 },
  by_agent: { "agent-001": 1.23 },
  by_operation: { llm_call: 1.23 },
};

const SAMPLE_DRIFT: DriftReport = {
  agent_id: "agent-001",
  timestamp: 1704067200,
  drifted: false,
  max_z_score: 1.2,
  threshold: 3.0,
  drifted_features: {},
  all_z_scores: { token_count: 1.2 },
  baseline_age_seconds: 3600,
  window_span_count: 50,
  severity: "none",
  notes: "",
};

const SAMPLE_FLEET: FleetStatus = {
  agent_count: 3,
  drifting_count: 0,
  total_cost_usd: 5.67,
  total_span_count: 1000,
  agents: [],
  generated_at: "2024-01-01T00:00:00Z",
};

const SAMPLE_TRACE_EXPORT: TraceExport = {
  trace_id: "abc123",
  agent_id: "agent-001",
  session_id: "session-001",
  task_id: "task-001",
  service_name: "test-service",
  spans: [],
  timestamp: 1704067200,
  tags: {},
};

// ---------------------------------------------------------------------------
// getTraces()
// ---------------------------------------------------------------------------

describe("createAgentObservabilityClient — getTraces()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with TraceListResponse on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_TRACE_LIST)));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getTraces();

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.total).toBe(0);
      expect(result.data.traces).toHaveLength(0);
    }
  });

  it("passes all filter params as query params", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_TRACE_LIST));
      }),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getTraces({
      agentId: "agent-001",
      sessionId: "session-001",
      kind: "llm.call",
      since: 1000,
      until: 2000,
      limit: 25,
    });

    expect(capturedUrl).toContain("agent_id=agent-001");
    expect(capturedUrl).toContain("session_id=session-001");
    expect(capturedUrl).toContain("kind=llm.call");
    expect(capturedUrl).toContain("since=1000");
    expect(capturedUrl).toContain("until=2000");
    expect(capturedUrl).toContain("limit=25");
  });

  it("omits optional params when not provided", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_TRACE_LIST));
      }),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getTraces();

    expect(capturedUrl).not.toContain("agent_id");
    expect(capturedUrl).not.toContain("session_id");
    expect(capturedUrl).not.toContain("kind");
  });
});

// ---------------------------------------------------------------------------
// getCostReport()
// ---------------------------------------------------------------------------

describe("createAgentObservabilityClient — getCostReport()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with CostAttribution on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_COST)));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getCostReport({ agentId: "agent-001" });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.total_cost_usd).toBe(1.23);
      expect(result.data.record_count).toBe(42);
    }
  });

  it("passes agentId, since, and until as query params", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_COST));
      }),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getCostReport({ agentId: "agent-001", since: 1000, until: 2000 });

    expect(capturedUrl).toContain("agent_id=agent-001");
    expect(capturedUrl).toContain("since=1000");
    expect(capturedUrl).toContain("until=2000");
  });
});

// ---------------------------------------------------------------------------
// getDriftAnalysis()
// ---------------------------------------------------------------------------

describe("createAgentObservabilityClient — getDriftAnalysis()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with DriftReport on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_DRIFT)));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getDriftAnalysis("agent-001");

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.drifted).toBe(false);
      expect(result.data.severity).toBe("none");
    }
  });

  it("passes agentId and options as query params", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_DRIFT));
      }),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.getDriftAnalysis("agent-001", { windowSpans: 100, sigmaThreshold: 3.5 });

    expect(capturedUrl).toContain("agent_id=agent-001");
    expect(capturedUrl).toContain("window_spans=100");
    expect(capturedUrl).toContain("sigma_threshold=3.5");
  });
});

// ---------------------------------------------------------------------------
// getFleetStatus()
// ---------------------------------------------------------------------------

describe("createAgentObservabilityClient — getFleetStatus()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with FleetStatus on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_FLEET)));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getFleetStatus();

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.agent_count).toBe(3);
      expect(result.data.drifting_count).toBe(0);
    }
  });
});

// ---------------------------------------------------------------------------
// exportTraces()
// ---------------------------------------------------------------------------

describe("createAgentObservabilityClient — exportTraces()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with TraceExport on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_TRACE_EXPORT)));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.exportTraces("abc123");

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.trace_id).toBe("abc123");
    }
  });

  it("URL-encodes the traceId in the path", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_TRACE_EXPORT));
      }),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.exportTraces("trace/with-slash");

    expect(capturedUrl).toContain(encodeURIComponent("trace/with-slash"));
  });
});

// ---------------------------------------------------------------------------
// sdk-core error handling integration
// ---------------------------------------------------------------------------

describe("createAgentObservabilityClient — sdk-core error handling", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:false with status 404 on not-found", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(404, { error: "Trace not found", detail: "" })),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.exportTraces("missing-trace");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(404);
      expect(result.error.error).toBe("Trace not found");
    }
  });

  it("returns ok:false with status 429 on rate limit", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(429, { error: "Rate limit exceeded", detail: "" }, {
          "retry-after": "60",
        }),
      ),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getFleetStatus();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(429);
    }
  });

  it("returns ok:false with status 500 on server error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(500, { error: "Internal error", detail: "DB unavailable" }),
      ),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getFleetStatus();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(500);
    }
  });

  it("returns ok:false with status 0 on network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Network error")));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getTraces();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(0);
      expect(result.error.error).toMatch(/network error/i);
    }
  });

  it("exposes SdkEventEmitter on client.events", () => {
    const client = createAgentObservabilityClient({ baseUrl: BASE_URL });
    expect(typeof client.events.on).toBe("function");
    expect(typeof client.events.off).toBe("function");
    expect(typeof client.events.emit).toBe("function");
  });

  it("fires request:start and request:end on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_FLEET)));

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const fired: string[] = [];

    client.events.on("request:start", () => fired.push("start"));
    client.events.on("request:end", () => fired.push("end"));

    await client.getFleetStatus();

    expect(fired).toContain("start");
    expect(fired).toContain("end");
  });

  it("fires request:error on final failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(503, { error: "Down", detail: "" })),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const errorEvents: unknown[] = [];
    client.events.on("request:error", ({ payload }) => errorEvents.push(payload.error));

    await client.getFleetStatus();

    expect(errorEvents).toHaveLength(1);
  });

  it("retries on 503 and succeeds on third attempt", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount += 1;
        if (callCount < 3) {
          return Promise.resolve(makeErrorResponse(503, { error: "Down", detail: "" }));
        }
        return Promise.resolve(makeSuccessResponse(SAMPLE_FLEET));
      }),
    );

    const retried: number[] = [];
    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 3 });
    client.events.on("request:retry", ({ payload }) => retried.push(payload.attempt));

    const result = await client.getFleetStatus();

    expect(result.ok).toBe(true);
    expect(callCount).toBe(3);
    expect(retried).toHaveLength(2);
  });

  it("returns ok:false after all retries exhausted on persistent 502", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(502, { error: "Bad gateway", detail: "" })),
    );

    const client = createAgentObservabilityClient({ baseUrl: BASE_URL, maxRetries: 1 });
    const result = await client.getFleetStatus();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(502);
    }
  });
});
