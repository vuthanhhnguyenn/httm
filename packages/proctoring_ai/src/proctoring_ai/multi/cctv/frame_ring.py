"""The last few seconds of decoded frames, so the deep tier can look back without reopening the video.

Keeps at most ``fps`` frames a second over ``span_ms`` (20 s at 5 fps is ~100 frames of 1280x720,
~280 MB; ``jpeg_quality`` stores them as JPEG instead, ~20 MB, at the cost of an encode per kept frame).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import cv2
import numpy as np


class FrameRing:
    def __init__(self, span_ms: int = 20000, fps: float = 5.0, jpeg_quality: int | None = None) -> None:
        self.span_ms = span_ms
        self.period_ms = 1000.0 / fps
        self.jpeg_quality = jpeg_quality
        self._frames: deque[tuple[int, Any]] = deque()
        self._lock = threading.Lock()  # written by the fast tier, read by the review thread

    def add(self, offset_ms: int, image: Any) -> None:
        with self._lock:
            if self._frames and offset_ms - self._frames[-1][0] < self.period_ms * 0.98:
                return
            stored = image.copy()
            if self.jpeg_quality is not None:
                ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                stored = buffer if ok else stored
            self._frames.append((offset_ms, stored))
            while self._frames and offset_ms - self._frames[0][0] > self.span_ms:
                self._frames.popleft()

    def oldest_ms(self) -> int | None:
        with self._lock:
            return self._frames[0][0] if self._frames else None

    def offsets(self) -> list[int]:
        with self._lock:
            return [offset for offset, _ in self._frames]

    def get(self, offset_ms: int) -> tuple[int, Any] | None:
        with self._lock:
            if not self._frames:
                return None
            offset, stored = min(self._frames, key=lambda item: abs(item[0] - offset_ms))
        if abs(offset - offset_ms) > self.period_ms:
            return None
        if self.jpeg_quality is not None and isinstance(stored, np.ndarray) and stored.ndim == 1:
            stored = cv2.imdecode(stored, cv2.IMREAD_COLOR)
        return offset, stored


__all__ = ["FrameRing"]
