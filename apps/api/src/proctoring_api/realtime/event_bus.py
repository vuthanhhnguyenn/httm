"""In-memory per-session event bus with monotonic sequence numbers."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from .schemas import WebSocketEnvelope


class SessionEventBus:
    def __init__(self, *, schema_version: str = "1.0", subscriber_queue_size: int = 64) -> None:
        if subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")
        self.schema_version = schema_version
        self._next_sequence: dict[UUID, int] = defaultdict(int)
        self._subscribers: dict[UUID, set[asyncio.Queue[WebSocketEnvelope]]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._queue_size = subscriber_queue_size

    async def publish(
        self,
        session_id: UUID,
        event_type: str,
        data: Mapping[str, Any] | None = None,
        *,
        occurred_at: datetime | None = None,
    ) -> WebSocketEnvelope:
        async with self._lock:
            self._next_sequence[session_id] += 1
            envelope = WebSocketEnvelope(
                schema_version=self.schema_version,
                sequence=self._next_sequence[session_id],
                type=event_type,
                session_id=session_id,
                occurred_at=occurred_at or datetime.now(UTC),
                data=dict(data or {}),
            )
            for queue in tuple(self._subscribers.get(session_id, ())):
                try:
                    queue.put_nowait(envelope)
                except asyncio.QueueFull:
                    # Keep the newest state; the reconnecting client obtains a REST snapshot.
                    queue.get_nowait()
                    queue.put_nowait(envelope)
            return envelope

    def latest_sequence(self, session_id: UUID) -> int:
        return self._next_sequence.get(session_id, 0)

    async def subscribe(self, session_id: UUID) -> AsyncIterator[WebSocketEnvelope]:
        queue: asyncio.Queue[WebSocketEnvelope] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers[session_id].add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(session_id)
                if subscribers is not None:
                    subscribers.discard(queue)
                    if not subscribers:
                        self._subscribers.pop(session_id, None)


__all__ = ["SessionEventBus"]

