"""Prevent slow native inference from building an unbounded executor backlog."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class InferenceBusyError(RuntimeError):
    """Raised when the previous call for an analyzer is still running."""


async def run_single_flight(lock: Any, callback: Callable[..., T], *args: Any) -> T:
    """Run at most one native call at a time, even if its waiter times out."""

    if not lock.acquire(blocking=False):
        raise InferenceBusyError("previous inference is still running")

    def invoke() -> T:
        try:
            return callback(*args)
        finally:
            lock.release()

    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(None, invoke)
    except BaseException:
        lock.release()
        raise
    return await asyncio.shield(future)


__all__ = ["InferenceBusyError", "run_single_flight"]
