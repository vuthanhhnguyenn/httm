"""Runtime observation and suspicious-event models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

from ..types import EventType, RiskScore, Severity, ensure_utc


@dataclass(frozen=True, slots=True)
class ObservationHealth:
    status: str
    latency_ms: float = 0.0
    reason: str | None = None


@dataclass
class ObservationBundle:
    frame_id: int
    session_id: UUID
    captured_at_utc: datetime
    captured_at_monotonic_ms: int
    frame_size: tuple[int, int]
    people: list[dict[str, Any]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    faces: list[dict[str, Any]] = field(default_factory=list)
    poses: list[dict[str, Any]] = field(default_factory=list)
    hands: list[dict[str, Any]] = field(default_factory=list)
    image_quality: dict[str, Any] = field(default_factory=dict)
    analyzer_health: dict[str, ObservationHealth] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.captured_at_monotonic_ms < 0:
            raise ValueError("captured_at_monotonic_ms must be non-negative")
        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("frame_size must be positive")
        self.captured_at_utc = ensure_utc(self.captured_at_utc)


@dataclass
class SuspiciousEvent:
    session_id: UUID
    event_type: EventType
    severity: Severity
    confidence: float
    started_at: datetime
    track_id: UUID | None = None
    id: UUID = field(default_factory=uuid4)
    status: str = "OPEN"
    risk_delta: float = 0.0
    risk_score_after: float = 0.0
    ended_at: datetime | None = None
    duration_ms: int | None = None
    reason_codes: list[str] = field(default_factory=list)
    explanation: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    policy_version: str = "default"
    config_version: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    rule_key: str = "default"

    def __post_init__(self) -> None:
        self.event_type = EventType(self.event_type)
        self.severity = Severity(self.severity)
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.reason_codes:
            raise ValueError("reason_codes must contain at least one code")
        self.risk_score_after = RiskScore(self.risk_score_after).value
        self.started_at = ensure_utc(self.started_at)
        self.created_at = ensure_utc(self.created_at)
        self.updated_at = ensure_utc(self.updated_at)
        if self.ended_at is not None:
            self.close(self.ended_at)

    def update(self, *, confidence: float, risk_score_after: float, metrics: Mapping[str, Any] | None = None) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence
        self.risk_score_after = RiskScore(risk_score_after).value
        if metrics:
            self.metrics.update(metrics)
        self.updated_at = datetime.now(UTC)

    def close(self, ended_at: datetime) -> None:
        ended = ensure_utc(ended_at)
        if ended < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        self.status = "CLOSED"
        self.ended_at = ended
        self.duration_ms = max(0, round((ended - self.started_at).total_seconds() * 1000))
        self.updated_at = ended
