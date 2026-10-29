"""CCTV-mode frame pipeline: YOLO-pose for everyone, phone detection every few frames.

Shaped like ``MultiSubjectPipeline`` (``detect`` then ``analyze_subjects``) so the same replay
loop, writer and timing recorder drive it; results carry keypoints in ``poses`` instead of faces.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any
from uuid import UUID

from proctoring_core.events.models import ObservationHealth

from ..pipeline import DetectionStage, FrameTiming, MultiFrameResult, _snapshot
from ..tracker import MultiSubjectTracker
from .rules import keypoints_for


class CctvPipeline:
    def __init__(self, *, pose_detector: Any, tracker: MultiSubjectTracker | Any, object_detector: Any | None = None,
                 phone_every: int = 3, timeout_ms: int = 1200) -> None:
        self.pose_detector = pose_detector
        self.tracker = tracker
        self.object_detector = object_detector
        self.phone_every = max(1, phone_every)
        self.timeout_ms = timeout_ms
        self._frames_seen = 0
        self._objects: list[dict[str, Any]] = []

    async def _call(self, detector: Any, frame: dict[str, Any]) -> tuple[list[dict[str, Any]], ObservationHealth]:
        started = perf_counter()
        try:
            result = await asyncio.wait_for(detector.detect(frame), timeout=self.timeout_ms / 1000)
            return list(result), ObservationHealth("ready", (perf_counter() - started) * 1000)
        except TimeoutError:
            return [], ObservationHealth("timeout", (perf_counter() - started) * 1000, "DETECTOR_TIMEOUT")
        except Exception as exc:
            return [], ObservationHealth("error", (perf_counter() - started) * 1000, type(exc).__name__)

    async def detect(self, frame: dict[str, Any]) -> DetectionStage:
        index = self._frames_seen
        self._frames_seen += 1
        people, health = await self._call(self.pose_detector, frame)
        reused = True
        if self.object_detector is not None and index % self.phone_every == 0:
            found, object_health = await self._call(self.object_detector, frame)
            if object_health.status == "ready":
                # Phones persist for seconds; detections from a few frames back are still current.
                self._objects = [item for item in found if str(item.get("class", "")).casefold() != "person"]
            health = ObservationHealth(health.status, health.latency_ms + object_health.latency_ms, health.reason)
            reused = False
        started = perf_counter()
        update = _snapshot(self.tracker.update(people, int(frame.get("captured_at_monotonic_ms", 0))))
        return DetectionStage(frame, people + list(self._objects), update, health, health.latency_ms,
                              (perf_counter() - started) * 1000, reused)

    async def analyze_subjects(self, stage: DetectionStage, *, session_id: UUID) -> MultiFrameResult:
        frame = stage.frame
        people = [item for item in stage.detections if "keypoints" in item]
        objects = [item for item in stage.detections if "keypoints" not in item]
        poses = keypoints_for(stage.tracker.analyzed, people)
        total = stage.detect_ms + stage.track_ms
        timing = FrameTiming(stage.detect_ms, stage.track_ms, 0.0, {}, 0.0, total, 0, 0, stage.detector_reused)
        return MultiFrameResult(int(frame.get("frame_id", 0)), int(frame.get("captured_at_monotonic_ms", 0)),
                                stage.tracker, {}, {}, {}, objects, timing, poses)

    async def analyze(self, frame: dict[str, Any], *, session_id: UUID) -> MultiFrameResult:
        return await self.analyze_subjects(await self.detect(frame), session_id=session_id)

    def mark_focus(self, seat_id: UUID, monotonic_ms: int) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = ["CctvPipeline"]
