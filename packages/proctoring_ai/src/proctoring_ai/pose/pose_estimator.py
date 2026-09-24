"""MediaPipe Tasks pose adapter; landmarks remain in-process for privacy."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from typing import Any

from ..pipeline.single_flight import InferenceBusyError, run_single_flight

try:
    import mediapipe as mp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    mp = None


class MediaPipePoseEstimator:
    def __init__(self, model_path: str | Path | None = None, *, enabled: bool = True, timeout_ms: int = 1200) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.enabled = enabled
        self.timeout_ms = timeout_ms
        self._landmarker: Any = None
        self._last_timestamp_ms = -1
        self._lock = Lock()
        self._inference_gate = Lock()
        self._reason: str | None = None
        self._initialization_attempted = False

    @property
    def health(self) -> dict[str, str | None]:
        if self._landmarker is not None:
            return {"status": "ready", "reason": None}
        return {"status": "unavailable" if self._reason else "disabled", "reason": self._reason}

    async def initialize(self) -> dict[str, str | None]:
        if self._initialization_attempted:
            return self.health
        self._initialization_attempted = True
        if not self.enabled:
            self._reason = "DISABLED"
        elif mp is None:
            self._reason = "MEDIAPIPE_NOT_INSTALLED"
        elif self.model_path is None or not self.model_path.is_file():
            self._reason = "POSE_MODEL_MISSING"
        else:
            try:
                await asyncio.to_thread(self._load_sync)
            except Exception as exc:
                self._reason = f"MODEL_LOAD_FAILED:{type(exc).__name__}"
        return self.health

    def _load_sync(self) -> None:
        if mp is None or self.model_path is None:
            return
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self._landmarker is None:
            await self.initialize()
        if self._landmarker is None:
            return {"status": "not_run", "reason": self._reason or "POSE_MODEL_MISSING", "poses": []}
        image = frame.get("image")
        if image is None:
            return {"status": "error", "reason": "FRAME_IMAGE_MISSING", "poses": []}
        timestamp = max(int(frame.get("captured_at_monotonic_ms", 0)), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        try:
            result = await asyncio.wait_for(
                run_single_flight(
                    self._inference_gate,
                    self._detect_sync,
                    image,
                    timestamp,
                    frame.get("rgb_image"),
                ),
                timeout=self.timeout_ms / 1000,
            )
        except InferenceBusyError:
            return {"status": "busy", "reason": "INFERENCE_ALREADY_RUNNING", "poses": []}
        except TimeoutError:
            return {"status": "timeout", "reason": "POSE_LANDMARKER_TIMEOUT", "poses": []}
        except Exception as exc:
            return {"status": "error", "reason": type(exc).__name__, "poses": []}
        if not result.pose_landmarks:
            return {"status": "no_detection", "reason": None, "poses": [], "frame_id": frame.get("frame_id")}

        landmarks = []
        for index, point in enumerate(result.pose_landmarks[0]):
            landmarks.append(
                {
                    "index": index,
                    "x": max(0.0, min(1.0, float(point.x))),
                    "y": max(0.0, min(1.0, float(point.y))),
                    "visibility": max(0.0, min(1.0, float(getattr(point, "visibility", 1.0)))),
                }
            )
        visible = [point for point in landmarks if point["visibility"] >= 0.25]
        bbox = _bbox(visible)
        pose = {"bbox": bbox, "landmarks": landmarks, "quality": min(1.0, len(visible) / 20)}
        return {"status": "ready", "reason": None, "poses": [pose], "frame_id": frame.get("frame_id")}

    def _detect_sync(self, image: Any, timestamp_ms: int, rgb_image: Any | None = None) -> Any:
        if mp is None:
            raise RuntimeError("MEDIAPIPE_NOT_INSTALLED")
        rgb = rgb_image if rgb_image is not None else image[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        with self._inference_gate:
            landmarker, self._landmarker = self._landmarker, None
            if landmarker is not None:
                landmarker.close()


def _bbox(points: list[dict[str, float]]) -> dict[str, float]:
    if not points:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    left = min(point["x"] for point in points)
    top = min(point["y"] for point in points)
    right = max(point["x"] for point in points)
    bottom = max(point["y"] for point in points)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


__all__ = ["MediaPipePoseEstimator"]
