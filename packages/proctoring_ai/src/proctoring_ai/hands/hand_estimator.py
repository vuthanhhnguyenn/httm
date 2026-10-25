"""MediaPipe Tasks hand adapter; exposes ephemeral normalized landmarks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from typing import Any

from ..acceleration import create_mediapipe_task
from ..pipeline.single_flight import InferenceBusyError, run_single_flight
from .matching import HandMatcher

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None


WRIST, INDEX_TIP = 0, 8


class MediaPipeHandEstimator:
    def __init__(self, model_path: str | Path | None = None, *, enabled: bool = True, timeout_ms: int = 1200,
                 min_confidence: float = 0.4, movement_threshold: float = 0.025) -> None:
        self.min_confidence = min_confidence
        self.movement_threshold = movement_threshold
        self.model_path = Path(model_path) if model_path else None
        self.enabled = enabled
        self.timeout_ms = timeout_ms
        self._landmarker: Any = None
        self._last_timestamp_ms = -1
        self._matcher = HandMatcher()
        self._lock = Lock()
        self._inference_gate = Lock()
        self._reason: str | None = None
        self.device = "cpu"
        self._initialization_attempted = False

    @property
    def health(self) -> dict[str, str | None]:
        if self._landmarker is not None:
            return {"status": "ready", "reason": None, "device": self.device}
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
            self._reason = "HAND_MODEL_MISSING"
        else:
            try:
                await asyncio.to_thread(self._load_sync)
            except Exception as exc:
                self._reason = f"MODEL_LOAD_FAILED:{type(exc).__name__}"
        return self.health

    def _load_sync(self) -> None:
        if mp is None or self.model_path is None:
            return
        model_path = str(self.model_path)

        def create(delegate: Any) -> Any:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path, delegate=delegate),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.min_confidence,
                min_hand_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            return mp.tasks.vision.HandLandmarker.create_from_options(options)

        self._landmarker, self.device = create_mediapipe_task("hands", mp, create)

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self._landmarker is None:
            await self.initialize()
        if self._landmarker is None:
            return {"status": "not_run", "reason": self._reason or "HAND_MODEL_MISSING", "hands": []}
        image = frame.get("image")
        if image is None:
            return {"status": "error", "reason": "FRAME_IMAGE_MISSING", "hands": []}
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
            return {"status": "busy", "reason": "INFERENCE_ALREADY_RUNNING", "hands": []}
        except TimeoutError:
            return {"status": "timeout", "reason": "HAND_LANDMARKER_TIMEOUT", "hands": []}
        except Exception as exc:
            return {"status": "error", "reason": type(exc).__name__, "hands": []}

        hands: list[dict[str, Any]] = []
        handedness = getattr(result, "handedness", [])
        for index, raw_landmarks in enumerate(result.hand_landmarks):
            landmarks = [
                {
                    "index": point_index,
                    "x": max(0.0, min(1.0, float(point.x))),
                    "y": max(0.0, min(1.0, float(point.y))),
                    "z": float(getattr(point, "z", 0.0)),
                }
                for point_index, point in enumerate(raw_landmarks)
            ]
            xs = [point["x"] for point in landmarks]
            ys = [point["y"] for point in landmarks]
            bbox = {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}
            categories = handedness[index] if index < len(handedness) else []
            category = categories[0] if categories else None
            hands.append(
                {
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "wrist": (landmarks[WRIST]["x"], landmarks[WRIST]["y"]),
                    "index_tip": (landmarks[INDEX_TIP]["x"], landmarks[INDEX_TIP]["y"]),
                    "handedness": getattr(category, "category_name", "unknown"),
                    "confidence": max(0.0, min(1.0, float(getattr(category, "score", 0.0)))),
                }
            )
        matches = self._matcher.update([(hand["handedness"], hand["wrist"]) for hand in hands], timestamp)
        for hand, (hand_id, displacement, dt_ms) in zip(hands, matches, strict=True):
            hand["hand_id"] = hand_id
            hand["displacement"] = displacement
            hand["dt_ms"] = dt_ms
            hand["activity"] = displacement >= self.movement_threshold
        return {
            "status": "ready" if hands else "no_detection",
            "reason": None,
            "hands": hands,
            "hand_activity": any(hand["activity"] for hand in hands),
            "frame_id": frame.get("frame_id"),
        }

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


__all__ = ["MediaPipeHandEstimator"]
