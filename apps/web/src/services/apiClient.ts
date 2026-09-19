import type {
  ApiList,
  ErrorResponse,
  HealthResponse,
  Session,
  SessionCommandResult,
  SessionCreate,
  SessionDetail,
  SessionStatus,
  SuspiciousEvent,
  EventDetail,
} from "../types/api.generated";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly payload: ErrorResponse,
  ) {
    super(payload.message);
    this.name = "ApiError";
  }
}

export type TokenProvider = () => string | undefined;
export type FetchLike = typeof fetch;

export class ApiClient {
  constructor(
    private readonly baseUrl = "",
    private readonly fetchImpl: FetchLike = fetch,
    private readonly tokenProvider?: TokenProvider,
  ) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const token = this.tokenProvider?.();
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, { ...init, headers });
    if (!response.ok) {
      let payload: ErrorResponse;
      try {
        payload = (await response.json()) as ErrorResponse;
      } catch {
        payload = { code: "HTTP_ERROR", message: response.statusText, request_id: "unknown" };
      }
      throw new ApiError(response.status, payload);
    }
    return (await response.json()) as T;
  }

  health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/api/health");
  }

  listSessions(params: { status?: SessionStatus; cursor?: string; limit?: number } = {}): Promise<ApiList<Session>> {
    const query = new URLSearchParams();
    if (params.status) query.set("status", params.status);
    if (params.cursor) query.set("cursor", params.cursor);
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    const encoded = query.toString();
    const suffix = encoded ? `?${encoded}` : "";
    return this.request<ApiList<Session>>(`/api/sessions${suffix}`);
  }

  createSession(input: SessionCreate): Promise<Session> {
    return this.request<Session>("/api/sessions", { method: "POST", body: JSON.stringify(input) });
  }

  getSession(sessionId: string): Promise<SessionDetail> {
    return this.request<SessionDetail>(`/api/sessions/${encodeURIComponent(sessionId)}`);
  }

  startSession(sessionId: string, idempotencyKey: string): Promise<SessionCommandResult> {
    return this.command(`/api/sessions/${encodeURIComponent(sessionId)}/start`, idempotencyKey);
  }

  stopSession(sessionId: string, idempotencyKey: string): Promise<SessionCommandResult> {
    return this.command(`/api/sessions/${encodeURIComponent(sessionId)}/stop`, idempotencyKey);
  }

  listEvents(sessionId: string, params: { cursor?: string; limit?: number } = {}): Promise<ApiList<SuspiciousEvent>> {
    const query = new URLSearchParams();
    if (params.cursor) query.set("cursor", params.cursor);
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    const encoded = query.toString();
    const suffix = encoded ? `?${encoded}` : "";
    return this.request<ApiList<SuspiciousEvent>>(
      `/api/sessions/${encodeURIComponent(sessionId)}/events${suffix}`,
    );
  }

  getEvent(eventId: string): Promise<EventDetail> {
    return this.request<EventDetail>(`/api/events/${encodeURIComponent(eventId)}`);
  }

  private command(path: string, idempotencyKey: string): Promise<SessionCommandResult> {
    const headers = new Headers({ "Idempotency-Key": idempotencyKey });
    return this.request<SessionCommandResult>(path, { method: "POST", headers });
  }
}
