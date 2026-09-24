from __future__ import annotations

from collections import deque
from typing import Any


class PreviewBuffer:
    """Bounded in-memory preview ring; it never persists a full session video."""

    def __init__(self, max_frames: int = 30) -> None:
        self._frames: deque[dict[str, Any]] = deque(maxlen=max_frames)

    def append(self, frame: dict[str, Any]) -> None:
        self._frames.append(frame)

    def latest(self) -> dict[str, Any] | None:
        return self._frames[-1] if self._frames else None

    def recent(self, limit: int = 1) -> list[dict[str, Any]]:
        return list(self._frames)[-max(1, limit) :]

    def clear(self) -> None:
        self._frames.clear()

