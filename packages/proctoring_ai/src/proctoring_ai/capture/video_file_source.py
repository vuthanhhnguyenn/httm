"""Deterministic frame source for recorded video replay.

Timestamps come from the video position, not the wall clock, so the same file and
analysis rate always produce the same frames, schedules and rule timings.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

cv2: Any
try:  # OpenCV is optional until a capture adapter is selected.
    import cv2
except ImportError:  # pragma: no cover - exercised in environments without CV wheels
    cv2 = None

REPLAY_EPOCH_UTC = datetime(2026, 1, 1, tzinfo=UTC)


class VideoFileFrameSource:
    """Yield frames sampled at a fixed analysis rate from a video file.

    ``captured_at_monotonic_ms`` equals the video offset of the frame, so rule
    durations are measured in video time. ``read()`` returns ``None`` once the
    file (or ``end_ms``) is exhausted; ``finished`` then becomes true.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        analysis_rate_hz: float = 8.0,
        start_ms: int = 0,
        end_ms: int | None = None,
        epoch_utc: datetime = REPLAY_EPOCH_UTC,
    ) -> None:
        if analysis_rate_hz <= 0:
            raise ValueError("analysis_rate_hz must be positive")
        if start_ms < 0 or (end_ms is not None and end_ms <= start_ms):
            raise ValueError("invalid replay window")
        self.path = Path(path)
        self.analysis_rate_hz = analysis_rate_hz
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.epoch_utc = epoch_utc
        self.fps = 0.0
        self.frame_count = 0
        self.finished = False
        self._capture: Any = None
        self._index = -1
        self._frame_id = 0
        self._next_sample_ms = float(start_ms)

    @property
    def duration_ms(self) -> int:
        if self.fps <= 0:
            return 0
        return round(self.frame_count * 1000 / self.fps)

    async def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"unable to open video {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._capture = capture

    def _offset_ms(self, index: int) -> float:
        return index * 1000.0 / self.fps

    def _read_next_sync(self) -> tuple[int, Any] | None:
        step_ms = 1000.0 / self.analysis_rate_hz
        while True:
            # grab() skips decoding colour conversion for frames that are not analysed.
            if not self._capture.grab():
                return None
            self._index += 1
            offset = self._offset_ms(self._index)
            if self.end_ms is not None and offset >= self.end_ms:
                return None
            if offset + 1e-6 < self._next_sample_ms:
                continue
            ok, image = self._capture.retrieve()
            if not ok:
                return None
            while self._next_sample_ms <= offset + 1e-6:
                self._next_sample_ms += step_ms
            return round(offset), image

    async def read(self) -> dict[str, Any] | None:
        if self._capture is None:
            raise RuntimeError("frame source is not open")
        if self.finished:
            return None
        sampled = await asyncio.to_thread(self._read_next_sync)
        if sampled is None:
            self.finished = True
            return None
        offset_ms, image = sampled
        self._frame_id += 1
        return {
            "frame_id": self._frame_id,
            "image": image,
            "captured_at_utc": self.epoch_utc + timedelta(milliseconds=offset_ms),
            # +1 keeps the first sample strictly positive for monotonic validators.
            "captured_at_monotonic_ms": offset_ms + 1,
            "video_offset_ms": offset_ms,
            "frame_size": (int(image.shape[1]), int(image.shape[0])),
        }

    async def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            await asyncio.to_thread(capture.release)


__all__ = ["REPLAY_EPOCH_UTC", "VideoFileFrameSource"]
