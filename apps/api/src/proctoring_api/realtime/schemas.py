"""Versioned WebSocket event contract."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WebSocketEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0", min_length=1)
    sequence: int = Field(ge=1)
    type: str = Field(min_length=1, max_length=120)
    session_id: UUID
    occurred_at: datetime
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(UTC)


__all__ = ["WebSocketEnvelope"]

