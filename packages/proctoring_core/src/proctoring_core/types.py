"""Shared domain enums and validated value objects.

This module deliberately has no dependency on FastAPI, SQLAlchemy or an AI framework.  Adapters
convert their output to these bounded values before handing it to business rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Mapping


class SessionStatus(StrEnum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class SessionMode(StrEnum):
    SINGLE_PERSON = "SINGLE_PERSON"
    MULTI_PERSON = "MULTI_PERSON"


class CameraStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"


class ProcessingStatus(StrEnum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskLevel(StrEnum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TrackStatus(StrEnum):
    ACTIVE = "ACTIVE"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    CLOSED = "CLOSED"


class EventType(StrEnum):
    HEAD_TURN_LEFT = "HEAD_TURN_LEFT"
    HEAD_TURN_RIGHT = "HEAD_TURN_RIGHT"
    LOOK_DOWN = "LOOK_DOWN"
    ABNORMAL_HEAD_MOVEMENT = "ABNORMAL_HEAD_MOVEMENT"
    PHONE_DETECTED = "PHONE_DETECTED"
    DOCUMENT_DETECTED = "DOCUMENT_DETECTED"
    MULTIPLE_PERSON_DETECTED = "MULTIPLE_PERSON_DETECTED"
    PERSON_MISSING = "PERSON_MISSING"
    LEAVING_SEAT = "LEAVING_SEAT"
    CAMERA_BLOCKED = "CAMERA_BLOCKED"
    FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"
    SUSPICIOUS_HAND_ACTIVITY = "SUSPICIOUS_HAND_ACTIVITY"
    POSSIBLE_TALKING = "POSSIBLE_TALKING"


class EvidenceKind(StrEnum):
    FRAME = "FRAME"
    CLIP = "CLIP"


class EvidenceStatus(StrEnum):
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    FAILED = "FAILED"
    DELETED = "DELETED"


def _bounded(value: float, *, lower: float, upper: float, name: str) -> float:
    if not isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return float(value)


@dataclass(frozen=True, slots=True)
class Confidence:
    """A detector confidence bounded to the contract range 0..1."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _bounded(self.value, lower=0.0, upper=1.0, name="confidence"))

    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class RiskScore:
    """A risk score bounded to 0..100."""

    value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _bounded(self.value, lower=0.0, upper=100.0, name="risk_score"))

    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized bounding box with coordinates and dimensions in 0..1."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            object.__setattr__(
                self,
                name,
                _bounded(getattr(self, name), lower=0.0, upper=1.0, name=name),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BoundingBox":
        return cls(
            x=float(value["x"]),
            y=float(value["y"]),
            width=float(value["width"]),
            height=float(value["height"]),
        )

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def ensure_utc(value: datetime) -> datetime:
    """Return an aware UTC timestamp and reject ambiguous naive datetimes."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class UtcTimestamp:
    value: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", ensure_utc(self.value))

    @classmethod
    def now(cls) -> "UtcTimestamp":
        return cls(datetime.now(UTC))

    def isoformat(self) -> str:
        return self.value.isoformat()

    def __str__(self) -> str:
        return self.isoformat()

