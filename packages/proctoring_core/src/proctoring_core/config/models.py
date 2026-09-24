"""Immutable, versioned configuration and exam-policy domain models.

The core package keeps these models independent from application and storage adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

from .schema import ExamPolicyConfig


class ConfigVersionStatus(StrEnum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


# Backward-compatible descriptive alias used by adapters that name the state after the entity.
SystemConfigStatus = ConfigVersionStatus


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return deterministic JSON used for hashes and optimistic comparisons."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExamPolicy:
    id: UUID
    name: str
    version: int
    allow_book: bool = False
    allow_scratch_paper: bool = True
    allow_phone: bool = False
    allowed_materials: tuple[str, ...] = ()
    evidence_frame_enabled: bool = True
    evidence_clip_enabled: bool = False
    pre_event_seconds: float = 3.0
    post_event_seconds: float = 3.0
    retention_days: int = 30
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        payload = {
            "name": self.name,
            "allow_book": self.allow_book,
            "allow_scratch_paper": self.allow_scratch_paper,
            "allow_phone": self.allow_phone,
            "allowed_materials": list(self.allowed_materials),
            "evidence_frame_enabled": self.evidence_frame_enabled,
            "evidence_clip_enabled": self.evidence_clip_enabled,
            "pre_event_seconds": self.pre_event_seconds,
            "post_event_seconds": self.post_event_seconds,
            "retention_days": self.retention_days,
        }
        validated = ExamPolicyConfig.model_validate(payload)
        if len(validated.allowed_materials) != len(set(validated.allowed_materials)):
            raise ValueError("allowed_materials must be unique")
        if self.version < 1:
            raise ValueError("policy version must be positive")
        if not self.created_by or len(self.created_by) > 200:
            raise ValueError("created_by must contain 1..200 characters")
        object.__setattr__(self, "name", validated.name)
        object.__setattr__(self, "allowed_materials", tuple(validated.allowed_materials))
        object.__setattr__(self, "pre_event_seconds", validated.pre_event_seconds)
        object.__setattr__(self, "post_event_seconds", validated.post_event_seconds)
        object.__setattr__(self, "retention_days", validated.retention_days)
        object.__setattr__(self, "created_at", _utc(self.created_at))
        object.__setattr__(self, "updated_at", _utc(self.updated_at))

    @classmethod
    def create(cls, content: Mapping[str, Any], *, version: int = 1, created_by: str = "system", policy_id: UUID | None = None) -> "ExamPolicy":
        validated = ExamPolicyConfig.model_validate(dict(content))
        now = datetime.now(UTC)
        return cls(
            id=policy_id or uuid4(),
            name=validated.name,
            version=version,
            allow_book=validated.allow_book,
            allow_scratch_paper=validated.allow_scratch_paper,
            allow_phone=validated.allow_phone,
            allowed_materials=tuple(validated.allowed_materials),
            evidence_frame_enabled=validated.evidence_frame_enabled,
            evidence_clip_enabled=validated.evidence_clip_enabled,
            pre_event_seconds=validated.pre_event_seconds,
            post_event_seconds=validated.post_event_seconds,
            retention_days=validated.retention_days,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )

    def content(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allow_book": self.allow_book,
            "allow_scratch_paper": self.allow_scratch_paper,
            "allow_phone": self.allow_phone,
            "allowed_materials": list(self.allowed_materials),
            "evidence_frame_enabled": self.evidence_frame_enabled,
            "evidence_clip_enabled": self.evidence_clip_enabled,
            "pre_event_seconds": self.pre_event_seconds,
            "post_event_seconds": self.post_event_seconds,
            "retention_days": self.retention_days,
        }

    def snapshot(self) -> dict[str, Any]:
        return {"id": str(self.id), "version": self.version, **self.content()}


@dataclass(frozen=True, slots=True)
class SystemConfigVersion:
    id: UUID
    version: int
    status: ConfigVersionStatus
    content: Mapping[str, Any]
    sha256: str
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    activated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("config version must be positive")
        status = ConfigVersionStatus(self.status)
        payload = dict(self.content)
        expected = canonical_sha256(payload)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or self.sha256 != expected:
            raise ValueError("sha256 must match canonical config content")
        if not self.created_by or len(self.created_by) > 200:
            raise ValueError("created_by must contain 1..200 characters")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "content", MappingProxyType(payload))
        object.__setattr__(self, "created_at", _utc(self.created_at))
        if self.activated_at is not None:
            object.__setattr__(self, "activated_at", _utc(self.activated_at))

    @classmethod
    def create(cls, content: Mapping[str, Any], *, version: int = 1, status: ConfigVersionStatus = ConfigVersionStatus.DRAFT, created_by: str = "system", config_id: UUID | None = None) -> "SystemConfigVersion":
        return cls(config_id or uuid4(), version, status, dict(content), canonical_sha256(content), created_by)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "ConfigVersionStatus",
    "ExamPolicy",
    "SystemConfigVersion",
    "SystemConfigStatus",
    "canonical_json",
    "canonical_sha256",
]
