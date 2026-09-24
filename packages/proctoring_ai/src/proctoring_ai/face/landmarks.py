"""MediaPipe Tasks face detection and head orientation adapter."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from threading import Lock
from typing import Any

from ..pipeline.single_flight import InferenceBusyError, run_single_flight
from .head_pose import HeadPoseEstimator, pose_from_matrix

try:
    import mediapipe as mp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    mp = None


class MediaPipeFaceLandmarker:
    """Detect one face and return only a box and coarse head pose."""

    def __init__(self, model_path: str | Path | None = None, *, timeout_ms: int = 1200) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.timeout_ms = timeout_ms
        self._landmarker: Any = None
        self._last_timestamp_ms = -1
        self._lock = Lock()
        self._inference_gate = Lock()
        self._reason: str | None = None
        self._initialization_attempted = False
        self._pose_estimator = HeadPoseEstimator()

    def request_calibration(self) -> None:
        self._pose_estimator.begin_calibration()

    @property
    def health(self) -> dict[str, Any]:
        if self._landmarker is not None:
            return {"status": "ready", "reason": None}
        return {"status": "unavailable" if self._reason else "disabled", "reason": self._reason}

    async def initialize(self) -> dict[str, Any]:
        if self._initialization_attempted:
            return self.health
        self._initialization_attempted = True
        if mp is None:
            self._reason = "MEDIAPIPE_NOT_INSTALLED"
        elif self.model_path is None or not self.model_path.is_file():
            self._reason = "FACE_MODEL_MISSING"
        else:
            try:
                await asyncio.to_thread(self._load_sync)
            except Exception as exc:
                self._reason = f"MODEL_LOAD_FAILED:{type(exc).__name__}"
        return self.health

    def _load_sync(self) -> None:
        if mp is None or self.model_path is None:
            return
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self._landmarker is None:
            await self.initialize()
        if self._landmarker is None:
            return {"status": "not_run", "reason": self._reason or "FACE_MODEL_MISSING", "faces": []}

        timestamp = int(frame.get("captured_at_monotonic_ms", 0))
        if timestamp <= self._last_timestamp_ms:
            timestamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp
        image = frame.get("image")
        if image is None:
            return {"status": "error", "reason": "FRAME_IMAGE_MISSING", "faces": []}
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
            return {"status": "busy", "reason": "INFERENCE_ALREADY_RUNNING", "faces": []}
        except TimeoutError:
            return {"status": "timeout", "reason": "FACE_LANDMARKER_TIMEOUT", "faces": []}
        except Exception as exc:
            return {"status": "error", "reason": type(exc).__name__, "faces": []}
        if not result.face_landmarks:
            return {"status": "no_detection", "reason": None, "faces": [], "frame_id": frame.get("frame_id")}

        height, width = image.shape[:2]
        landmarks = result.face_landmarks[0]
        xs = [max(0.0, min(1.0, float(point.x))) for point in landmarks]
        ys = [max(0.0, min(1.0, float(point.y))) for point in landmarks]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        pad_x, pad_y = (right - left) * 0.04, (bottom - top) * 0.04
        bbox = {
            "x": max(0.0, left - pad_x),
            "y": max(0.0, top - pad_y),
            "width": min(1.0, right + pad_x) - max(0.0, left - pad_x),
            "height": min(1.0, bottom + pad_y) - max(0.0, top - pad_y),
        }
        pose = _head_pose(result)
        size_quality = min(1.0, math.sqrt(max(0.0, bbox["width"] * bbox["height"])) * 2.5)
        face = {
            "bbox": bbox,
            "yaw": pose["yaw"] if pose else 0.0,
            "pitch": pose["pitch"] if pose else 0.0,
            "roll": pose["roll"] if pose else 0.0,
            "pose_valid": pose is not None,
            "quality": max(0.0, min(1.0, size_quality)) if pose else 0.0,
            "frame_width": int(width),
            "frame_height": int(height),
        }
        face = self._pose_estimator.update(face, timestamp)
        return {"status": "ready", "reason": None, "faces": [face], "frame_id": frame.get("frame_id")}

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


def _head_pose(result: Any) -> dict[str, float] | None:
    matrices = getattr(result, "facial_transformation_matrixes", None) or []
    if not matrices:
        return None
    return pose_from_matrix(matrices[0])


__all__ = ["MediaPipeFaceLandmarker"]
