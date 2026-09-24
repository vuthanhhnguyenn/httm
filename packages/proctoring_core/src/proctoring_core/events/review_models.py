"""Evidence, human review and audit domain records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4


class EvidenceKind(StrEnum):
    FRAME = "FRAME"
    CLIP = "CLIP"


class EvidenceStatus(StrEnum):
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    FAILED = "FAILED"
    DELETED = "DELETED"


class ReviewType(StrEnum):
    VALID_SIGNAL = "VALID_SIGNAL"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


Disposition = ReviewType


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    id: UUID
    event_id: UUID
    session_id: UUID
    kind: EvidenceKind = EvidenceKind.FRAME
    status: EvidenceStatus = EvidenceStatus.PENDING
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    relative_path: str | None = None
    captured_from: datetime | None = None
    captured_to: datetime | None = None
    retention_until: datetime = field(default_factory=lambda: datetime.now(UTC))
    failure_code: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        kind = EvidenceKind(self.kind)
        status = EvidenceStatus(self.status)
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if status is EvidenceStatus.FAILED and not self.failure_code:
            raise ValueError("failure_code is required for failed evidence")
        if status is EvidenceStatus.DELETED and self.deleted_at is None:
            object.__setattr__(self, "deleted_at", datetime.now(UTC))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "created_at", _utc(self.created_at))
        object.__setattr__(self, "retention_until", _utc(self.retention_until))
        for field_name in ("captured_from", "captured_to", "deleted_at"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _utc(value))


@dataclass(frozen=True, slots=True)
class ReviewDisposition:
    id: UUID
    event_id: UUID
    disposition: ReviewType
    reviewed_by: str
    note: str | None = None
    reviewed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.note is not None and len(self.note) > 2000:
            raise ValueError("review note must be at most 2000 characters")
        if not self.reviewed_by or len(self.reviewed_by) > 200:
            raise ValueError("reviewed_by must contain 1..200 characters")
        object.__setattr__(self, "disposition", ReviewType(self.disposition))
        object.__setattr__(self, "reviewed_at", _utc(self.reviewed_at))

    @classmethod
    def create(cls, event_id: UUID, disposition: ReviewType | str, *, reviewed_by: str, note: str | None = None, review_id: UUID | None = None) -> "ReviewDisposition":
        return cls(review_id or uuid4(), event_id, ReviewType(disposition), reviewed_by, note)


@dataclass(frozen=True, slots=True)
class AuditLog:
    id: UUID
    actor_id: str
    actor_role: str
    action: str
    resource_type: str
    resource_id: str
    outcome: str = "ALLOWED"
    session_id: UUID | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.actor_id or not self.action or not self.resource_type or not self.resource_id:
            raise ValueError("audit actor, action, resource type and resource id are required")
        if self.actor_role not in {"operator", "reviewer", "admin"}:
            raise ValueError("invalid audit actor role")
        if self.outcome not in {"ALLOWED", "DENIED", "FAILED"}:
            raise ValueError("invalid audit outcome")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["AuditLog", "Disposition", "EvidenceArtifact", "EvidenceKind", "EvidenceStatus", "ReviewDisposition", "ReviewType"]
