"""OpenCV webcam/file frame source with UTC and monotonic timestamps."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

try:  # OpenCV is optional until the capture adapter is selected.
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in environments without CV wheels
    cv2 = None


class OpenCVFrameSource:
    def __init__(
        self,
        source: int | str = 0,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 25.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._capture: Any = None
        self._frame_id = 0
        self._last_monotonic_ms = -1
        self._opened = False

    async def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        if cv2 is None:  # pragma: no cover - guarded by open()
            raise RuntimeError("opencv-python is not installed")
        source: int | str = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._opened = bool(self._capture.isOpened())
        if not self._opened:
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"unable to open camera source {self.source!r}")

    async def read(self) -> dict[str, Any] | None:
        if not self._opened or self._capture is None:
            raise RuntimeError("frame source is not open")
        ok, image = await asyncio.to_thread(self._capture.read)
        if not ok:
            return None
        self._frame_id += 1
        monotonic_ms = max(time.monotonic_ns() // 1_000_000, self._last_monotonic_ms + 1)
        self._last_monotonic_ms = monotonic_ms
        return {
            "frame_id": self._frame_id,
            "image": image,
            "captured_at_utc": datetime.now(UTC),
            "captured_at_monotonic_ms": monotonic_ms,
            "frame_size": (int(image.shape[1]), int(image.shape[0])),
        }

    async def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._opened = False
        self._last_monotonic_ms = -1
        if capture is not None:
            await asyncio.to_thread(capture.release)


__all__ = ["OpenCVFrameSource"]
