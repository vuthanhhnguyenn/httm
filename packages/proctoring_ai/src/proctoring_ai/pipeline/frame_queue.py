from __future__ import annotations

import asyncio
from typing import Any


class LatestFrameQueue:
    """Bounded queue that drops stale frames to protect end-to-end latency."""

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.dropped_frames = 0

    def put_nowait(self, frame: dict[str, Any]) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self.dropped_frames += 1
        self._queue.put_nowait(frame)

    async def put(self, frame: dict[str, Any]) -> None:
        self.put_nowait(frame)

    async def get(self) -> dict[str, Any]:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def __len__(self) -> int:
        return self._queue.qsize()

