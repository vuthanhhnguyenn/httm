"""Decode the next frames while the AI works on the current one.

Wraps any frame source with ``open``/``read``/``close``. A background task keeps up to ``max_queue``
frames ready; ``read`` returns ``None`` at the end and re-raises the source's error at the point the
reader reaches it, so frame order and failures look exactly as without the wrapper.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

_END = object()


class PrefetchFrameSource:
    def __init__(self, source: Any, max_queue: int = 8) -> None:
        self.source = source
        self.max_queue = max(1, max_queue)
        self._queue: asyncio.Queue[Any] | None = None
        self._task: asyncio.Task[None] | None = None
        self._done = False

    def __getattr__(self, name: str) -> Any:  # fps, duration_ms, finished, ... of the wrapped source
        return getattr(self.source, name)

    async def open(self) -> None:
        await self.source.open()
        self._queue = asyncio.Queue(maxsize=self.max_queue)
        self._done = False
        self._task = asyncio.create_task(self._fill())

    async def _fill(self) -> None:
        assert self._queue is not None
        try:
            while (frame := await self.source.read()) is not None:
                await self._queue.put(frame)
            await self._queue.put(_END)
        except Exception as exc:  # handed to the reader in order
            await self._queue.put(exc)

    async def read(self) -> dict[str, Any] | None:
        if self._queue is None:
            raise RuntimeError("frame source is not open")
        if self._done:
            return None
        item = await self._queue.get()
        if item is _END:
            self._done = True
            return None
        if isinstance(item, Exception):
            self._done = True
            raise item
        frame: dict[str, Any] = item
        return frame

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        self._queue = None
        await self.source.close()


__all__ = ["PrefetchFrameSource"]
