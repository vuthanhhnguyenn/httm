"""Structural interfaces for capture, analysis, storage and realtime delivery."""

from __future__ import annotations

from collections.abc import AsyncContextManager, AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID


class Clock(Protocol):
    def now_utc(self) -> datetime: ...

    def monotonic_ms(self) -> int: ...


class FrameSource(Protocol):
    async def open(self) -> None: ...

    async def read(self) -> Mapping[str, Any] | None: ...

    async def close(self) -> None: ...


class ObjectDetector(Protocol):
    async def detect(self, frame: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...


class LandmarkAnalyzer(Protocol):
    async def analyze(self, frame: Mapping[str, Any]) -> Mapping[str, Any]: ...


class Tracker(Protocol):
    async def update(self, observations: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]: ...


class SessionRepository(Protocol):
    async def get(self, session_id: UUID) -> Mapping[str, Any] | None: ...

    async def list(self, *, status: str | None = None, limit: int = 50) -> Sequence[Mapping[str, Any]]: ...

    async def save(self, session: Mapping[str, Any]) -> Mapping[str, Any]: ...


class EventRepository(Protocol):
    async def get(self, event_id: UUID) -> Mapping[str, Any] | None: ...

    async def list_for_session(self, session_id: UUID, *, limit: int = 50) -> Sequence[Mapping[str, Any]]: ...

    async def save(self, event: Mapping[str, Any]) -> Mapping[str, Any]: ...


class EvidenceStore(Protocol):
    async def put(self, *, event_id: UUID, content: bytes, media_type: str) -> Mapping[str, Any]: ...

    async def open(self, evidence_id: UUID) -> AsyncContextManager[Any]: ...

    async def delete(self, evidence_id: UUID) -> None: ...


class AuditRepository(Protocol):
    async def append(self, entry: Mapping[str, Any]) -> None: ...


class EventPublisher(Protocol):
    async def publish(self, session_id: UUID, event_type: str, data: Mapping[str, Any]) -> Any: ...

    def subscribe(self, session_id: UUID) -> AsyncIterator[Any]: ...
