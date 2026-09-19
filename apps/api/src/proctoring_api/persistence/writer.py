"""A bounded single-writer queue for SQLite metadata writes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class _WriteJob(Generic[T]):
    operation: Callable[[], Awaitable[T]]
    result: asyncio.Future[T]


class SingleWriterQueue:
    """Serialize write operations and provide backpressure instead of concurrent SQLite writes."""

    def __init__(self, maxsize: int = 128) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: asyncio.Queue[_WriteJob[Any] | None] = asyncio.Queue(maxsize=maxsize)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._worker is None:
            self._closed = False
            self._worker = asyncio.create_task(self._run(), name="sqlite-single-writer")

    async def submit(self, operation: Callable[[], Awaitable[T]]) -> T:
        if self._closed:
            raise RuntimeError("writer queue is closed")
        if self._worker is None:
            await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        await self._queue.put(_WriteJob(operation=operation, result=future))
        return await future

    async def close(self) -> None:
        self._closed = True
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                result = await job.operation()
            except Exception as exc:
                if not job.result.done():
                    job.result.set_exception(exc)
            else:
                if not job.result.done():
                    job.result.set_result(result)
            finally:
                self._queue.task_done()


__all__ = ["SingleWriterQueue"]
