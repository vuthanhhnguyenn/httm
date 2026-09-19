from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class FakeCamera:
    def __init__(self, frames: Iterable[Mapping[str, Any]] = ()) -> None:
        self._frames = list(frames)
        self._index = 0
        self.opened = False

    async def open(self) -> None:
        self.opened = True

    async def read(self) -> Mapping[str, Any] | None:
        if not self.opened:
            raise RuntimeError("fake camera is not open")
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    async def close(self) -> None:
        self.opened = False

