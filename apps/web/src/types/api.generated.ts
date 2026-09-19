/**
 * Generated from specs/001-smart-exam-proctoring/contracts/openapi.yaml.
 * Do not hand-edit fields without updating the contract and running `npm run contract:check`.
 */

export const API_CONTRACT_VERSION = "0.1.0" as const;

export type SessionStatus = "CREATED" | "STARTING" | "RUNNING" | "STOPPING" | "STOPPED" | "FAILED";
export type SessionMode = "SINGLE_PERSON" | "MULTI_PERSON";
export type RiskLevel = "NORMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type CameraStatus = "UNKNOWN" | "CONNECTED" | "DEGRADED" | "DISCONNECTED";
export type ProcessingStatus = "IDLE" | "LOADING" | "READY" | "DEGRADED" | "FAILED";
export type TrackStatus = "ACTIVE" | "OCCLUDED" | "LOST" | "CLOSED";
export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type EventType =
  | "HEAD_TURN_LEFT"
  | "HEAD_TURN_RIGHT"
  | "LOOK_DOWN"
  | "ABNORMAL_HEAD_MOVEMENT"
  | "PHONE_DETECTED"
  | "DOCUMENT_DETECTED"
  | "MULTIPLE_PERSON_DETECTED"
  | "PERSON_MISSING"
  | "LEAVING_SEAT"
  | "CAMERA_BLOCKED"
  | "FACE_NOT_VISIBLE"
  | "SUSPICIOUS_HAND_ACTIVITY"
  | "POSSIBLE_TALKING";

export interface ErrorResponse {
  code: string;
  message: string;
  request_id: string;
  details?: Record<string, unknown>;
}

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  database: "ready" | "degraded" | "unavailable";
  storage: "ready" | "degraded" | "unavailable";
  analyzers: Record<string, "ready" | "disabled" | "degraded" | "unavailable">;
}

export interface SessionCreate {
  name: string;
  mode: SessionMode;
  camera_source: string;
  policy_id: string;
}

export interface Session {
  id: string;
  name: string;
  mode: SessionMode;
  status: SessionStatus;
  current_risk_score: number;
  current_risk_level: RiskLevel;
  camera_status: CameraStatus;
  processing_status: ProcessingStatus;
  started_at: string | null;
  ended_at: string | null;
  failure_code?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CandidateTrack {
  id: string;
  runtime_track_id: number;
  status: TrackStatus;
  seat_region_id?: string | null;
  bbox?: { x: number; y: number; width: number; height: number } | null;
  visibility?: number | null;
  first_seen_at: string;
  last_seen_at: string;
  current_risk_score: number;
  current_risk_level: RiskLevel;
}

export interface RuntimeMetrics {
  capture_fps: number;
  processed_fps: number;
  dropped_frames: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  analyzer_latency_ms?: Record<string, number>;
}

export interface SessionDetail extends Session {
  policy_version: number;
  config_version: number;
  latest_sequence: number;
  active_tracks: CandidateTrack[];
  metrics?: RuntimeMetrics;
}

export interface SessionCommandResult {
  session_id: string;
  status: SessionStatus;
  request_id: string;
}

export interface SuspiciousEvent {
  id: string;
  session_id: string;
  track_id?: string | null;
  event_type: EventType;
  status: "OPEN" | "CLOSED";
  severity: Severity;
  confidence: number;
  risk_delta: number;
  risk_score_after: number;
  started_at: string;
  ended_at?: string | null;
  duration_ms?: number | null;
  reason_codes: string[];
  explanation: string;
  metrics?: Record<string, unknown>;
  evidence_status?: "NONE" | "PENDING" | "AVAILABLE" | "PARTIAL" | "FAILED" | "DELETED";
  created_at: string;
  updated_at: string;
}

export interface EvidenceMetadata {
  id: string;
  kind: "FRAME" | "CLIP";
  status: "PENDING" | "AVAILABLE" | "FAILED" | "DELETED";
  media_type?: string | null;
  size_bytes?: number | null;
  captured_from?: string | null;
  captured_to?: string | null;
  retention_until: string;
  failure_code?: string | null;
  created_at: string;
}

export interface EventDetail extends SuspiciousEvent {
  evidence: EvidenceMetadata[];
  reviews: ReviewDisposition[];
}

export interface ReviewDisposition {
  id: string;
  event_id: string;
  disposition: "VALID_SIGNAL" | "FALSE_POSITIVE" | "INCONCLUSIVE";
  note?: string;
  reviewed_by: string;
  reviewed_at: string;
}

export interface ApiList<T> {
  items: T[];
  next_cursor?: string | null;
}

