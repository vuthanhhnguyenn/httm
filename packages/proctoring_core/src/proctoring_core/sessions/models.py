"""Session and anonymous-track domain models with explicit lifecycle guards."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

from ..types import (
    BoundingBox,
    CameraStatus,
    ProcessingStatus,
    RiskLevel,
    RiskScore,
    SessionMode,
    SessionStatus,
    TrackStatus,
    ensure_utc,
)

_BIOMETRIC_TERMS = re.compile(r"(?:biometric|fingerprint|face\s*id|identity|căn\s*cước|sinh\s*trắc)", re.I)


class SessionStateError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SessionCommandResult:
    session_id: UUID
    status: SessionStatus
    request_id: str
    replayed: bool = False


@dataclass
class ExamSession:
    id: UUID
    name: str
    mode: SessionMode
    camera_source: str
    policy_id: UUID
    policy_snapshot: Mapping[str, Any]
    config_version_id: UUID
    config_snapshot: Mapping[str, Any]
    status: SessionStatus = SessionStatus.CREATED
    started_at: datetime | None = None
    ended_at: datetime | None = None
    current_risk_score: float = 0.0
    current_risk_level: RiskLevel = RiskLevel.NORMAL
    camera_status: CameraStatus = CameraStatus.UNKNOWN
    processing_status: ProcessingStatus = ProcessingStatus.IDLE
    failure_code: str | None = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _commands: dict[str, SessionCommandResult] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= len(self.name.strip()) <= 120:
            raise ValueError("session name must contain 1..120 characters")
        if _BIOMETRIC_TERMS.search(self.name):
            raise ValueError("session name must not contain biometric identity data")
        self.mode = SessionMode(self.mode)
        self.policy_snapshot = _freeze_mapping(self.policy_snapshot)
        self.config_snapshot = _freeze_mapping(self.config_snapshot)
        self.current_risk_score = float(RiskScore(self.current_risk_score).value)
        self.created_at = ensure_utc(self.created_at)
        self.updated_at = ensure_utc(self.updated_at)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        mode: SessionMode,
        camera_source: str,
        policy_id: UUID,
        policy_snapshot: Mapping[str, Any],
        config_version_id: UUID,
        config_snapshot: Mapping[str, Any],
        created_by: str,
        session_id: UUID | None = None,
    ) -> "ExamSession":
        return cls(
            id=session_id or uuid4(),
            name=name,
            mode=mode,
            camera_source=camera_source,
            policy_id=policy_id,
            policy_snapshot=policy_snapshot,
            config_version_id=config_version_id,
            config_snapshot=config_snapshot,
            created_by=created_by,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "mode": self.mode.value,
            "camera_source": self.camera_source,
            "status": self.status.value,
            "current_risk_score": self.current_risk_score,
            "current_risk_level": self.current_risk_level.value,
            "camera_status": self.camera_status.value,
            "processing_status": self.processing_status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "failure_code": self.failure_code,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def _remember(self, key: str, request_id: str) -> SessionCommandResult:
        result = SessionCommandResult(self.id, self.status, request_id)
        self._commands[key] = result
        return result

    def start(self, idempotency_key: str, request_id: str) -> SessionCommandResult:
        if not idempotency_key:
            raise SessionStateError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required")
        previous = self._commands.get(idempotency_key)
        if previous is not None:
            return SessionCommandResult(previous.session_id, previous.status, previous.request_id, replayed=True)
        if self.status is SessionStatus.STOPPED:
            raise SessionStateError("SESSION_TERMINAL", "STOPPED session cannot be started")
        if self.status is not SessionStatus.CREATED:
            raise SessionStateError("SESSION_STATE_CONFLICT", f"cannot start session in {self.status.value}")
        self.status = SessionStatus.STARTING
        self.processing_status = ProcessingStatus.LOADING
        self._touch()
        return self._remember(idempotency_key, request_id)

    def mark_running(self) -> None:
        if self.status is not SessionStatus.STARTING:
            raise SessionStateError("SESSION_STATE_CONFLICT", "only STARTING session can become RUNNING")
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.now(UTC)
        self.camera_status = CameraStatus.CONNECTED
        self.processing_status = ProcessingStatus.READY
        self._touch()

    def stop(self, idempotency_key: str, request_id: str) -> SessionCommandResult:
        if not idempotency_key:
            raise SessionStateError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required")
        previous = self._commands.get(idempotency_key)
        if previous is not None:
            return SessionCommandResult(previous.session_id, previous.status, previous.request_id, replayed=True)
        if self.status is SessionStatus.STOPPED:
            raise SessionStateError("SESSION_TERMINAL", "session is already stopped")
        if self.status not in {SessionStatus.STARTING, SessionStatus.RUNNING, SessionStatus.STOPPING, SessionStatus.FAILED}:
            raise SessionStateError("SESSION_STATE_CONFLICT", f"cannot stop session in {self.status.value}")
        self.status = SessionStatus.STOPPING
        self._touch()
        return self._remember(idempotency_key, request_id)

    def mark_stopped(self) -> None:
        if self.status not in {SessionStatus.STOPPING, SessionStatus.FAILED}:
            raise SessionStateError("SESSION_STATE_CONFLICT", "only STOPPING or FAILED can become STOPPED")
        self.status = SessionStatus.STOPPED
        self.ended_at = datetime.now(UTC)
        self.camera_status = CameraStatus.DISCONNECTED
        self.processing_status = ProcessingStatus.IDLE
        self._touch()

    def fail(self, failure_code: str) -> None:
        if self.status is SessionStatus.STOPPED:
            return
        self.status = SessionStatus.FAILED
        self.failure_code = failure_code[:120]
        self.processing_status = ProcessingStatus.FAILED
        self.ended_at = datetime.now(UTC)
        self._touch()

    def update_risk(self, score: float, level: RiskLevel) -> None:
        self.current_risk_score = RiskScore(score).value
        self.current_risk_level = RiskLevel(level)
        self._touch()


@dataclass
class CandidateTrack:
    id: UUID
    session_id: UUID
    runtime_track_id: int
    status: TrackStatus = TrackStatus.ACTIVE
    first_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lost_at: datetime | None = None
    closed_at: datetime | None = None
    current_bbox: BoundingBox | None = None
    visibility: float | None = None
    current_risk_score: float = 0.0
    current_risk_level: RiskLevel = RiskLevel.NORMAL

    def __post_init__(self) -> None:
        if self.runtime_track_id < 1:
            raise ValueError("runtime_track_id must be positive")
        self.status = TrackStatus(self.status)
        self.first_seen_at = ensure_utc(self.first_seen_at)
        self.last_seen_at = ensure_utc(self.last_seen_at)
        self.current_risk_score = RiskScore(self.current_risk_score).value
        if self.visibility is not None and not 0 <= self.visibility <= 1:
            raise ValueError("visibility must be between 0 and 1")

    def update(self, *, bbox: BoundingBox | None, visibility: float | None, occurred_at: datetime) -> None:
        self.current_bbox = bbox
        self.visibility = visibility
        self.last_seen_at = ensure_utc(occurred_at)
        self.status = TrackStatus.ACTIVE

    def mark_lost(self, occurred_at: datetime) -> None:
        self.status = TrackStatus.LOST
        self.lost_at = ensure_utc(occurred_at)

    def close(self, occurred_at: datetime) -> None:
        self.status = TrackStatus.CLOSED
        self.closed_at = ensure_utc(occurred_at)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(nested) for key, nested in item.items()})
        if isinstance(item, (list, tuple)):
            return tuple(freeze(nested) for nested in item)
        return item

    frozen = freeze(dict(value))
    return frozen if isinstance(frozen, Mapping) else MappingProxyType({})
