"""Compose capture and optional analyzers into a normalized ObservationBundle."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from time import monotonic
from typing import Any
from uuid import UUID

from proctoring_core.events.models import ObservationBundle, ObservationHealth

from .single_flight import InferenceBusyError


class ObservationPipeline:
    def __init__(
        self,
        *,
        detector: Any,
        tracker: Any,
        face_analyzer: Any | None = None,
        pose_analyzer: Any | None = None,
        hand_analyzer: Any | None = None,
        image_quality: Any | None = None,
        timeout_ms: int = 1200,
        object_tracker: Any | None = None,
        person_confidence_threshold: float = 0.6,
        pose_interval_ms: int = 250,
        hands_interval_ms: int = 200,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.face_analyzer = face_analyzer
        self.pose_analyzer = pose_analyzer
        self.hand_analyzer = hand_analyzer
        self.image_quality = image_quality
        self.timeout_ms = timeout_ms
        self.object_tracker = object_tracker
        self.person_confidence_threshold = person_confidence_threshold
        self._intervals = {"pose": max(0, pose_interval_ms), "hands": max(0, hands_interval_ms)}
        self._last_run: dict[str, int] = {}
        self._last_health: dict[str, ObservationHealth] = {}
        self._completed: dict[str, deque[float]] = {}
        self._last_frame_id = -1
        self._last_monotonic_ms = -1

    async def _scheduled(self, name: str, analyzer: Any, frame: dict[str, Any]) -> tuple[Any, ObservationHealth]:
        timestamp = int(frame["captured_at_monotonic_ms"])
        previous = self._last_health.get(name)
        if (analyzer is not None and previous is not None
                and previous.status in {"ready", "no_detection"}
                and timestamp - self._last_run[name] < self._intervals.get(name, 0)):
            # No cached observations: a skipped inference is not fresh rule evidence.
            return {}, ObservationHealth("skipped", reason="SCHEDULED_INTERVAL")
        self._last_run[name] = timestamp
        return await self._run(name, analyzer, frame)

    def performance(self) -> dict[str, dict[str, Any]]:
        now = monotonic()
        result = {}
        for name, health in self._last_health.items():
            times = self._completed.get(name, deque())
            recent = [value for value in times if now - value <= 2.0]
            elapsed = now - recent[0] if len(recent) >= 2 else 0
            result[name] = {
                "fps": (len(recent) - 1) / elapsed if elapsed > 0 else 0.0,
                "latency_ms": health.latency_ms,
                "observed_at_ms": self._last_run.get(name),
                "status": health.status,
                "reason": health.reason or "",
            }
        return result

    async def _run(self, name: str, analyzer: Any | None, frame: dict[str, Any]) -> tuple[Any, ObservationHealth]:
        if analyzer is None:
            return {}, ObservationHealth("not_run", reason="DISABLED")
        started = monotonic()
        try:
            result = await asyncio.wait_for(analyzer.analyze(frame), timeout=self.timeout_ms / 1000)
            status = str(result.get("status", "ready")) if isinstance(result, dict) else "ready"
            return result, ObservationHealth(status, (monotonic() - started) * 1000, result.get("reason") if isinstance(result, dict) else None)
        except InferenceBusyError:
            return {"status": "busy", "reason": "INFERENCE_ALREADY_RUNNING"}, ObservationHealth(
                "busy", (monotonic() - started) * 1000, "INFERENCE_ALREADY_RUNNING"
            )
        except TimeoutError:
            return {}, ObservationHealth("timeout", (monotonic() - started) * 1000, f"{name.upper()}_TIMEOUT")
        except Exception as exc:
            return {}, ObservationHealth("error", (monotonic() - started) * 1000, type(exc).__name__)

    async def analyze(self, frame: dict[str, Any], *, session_id: UUID) -> ObservationBundle:
        frame_id = int(frame.get("frame_id", 0))
        monotonic_ms = int(frame.get("captured_at_monotonic_ms", 0))
        if frame_id <= self._last_frame_id or monotonic_ms <= self._last_monotonic_ms:
            raise ValueError("observation frame id and monotonic timestamp must increase")
        self._last_frame_id = frame_id
        self._last_monotonic_ms = monotonic_ms
        health: dict[str, ObservationHealth] = {}

        image = frame.get("image")

        async def detect() -> tuple[list[dict[str, Any]], ObservationHealth]:
            detector_started = monotonic()
            try:
                result = await asyncio.wait_for(self.detector.detect(frame), timeout=self.timeout_ms / 1000)
                return list(result), ObservationHealth("ready", (monotonic() - detector_started) * 1000)
            except InferenceBusyError:
                return [], ObservationHealth("busy", (monotonic() - detector_started) * 1000, "INFERENCE_ALREADY_RUNNING")
            except TimeoutError:
                return [], ObservationHealth("timeout", (monotonic() - detector_started) * 1000, "DETECTOR_TIMEOUT")
            except Exception as exc:
                return [], ObservationHealth("error", (monotonic() - detector_started) * 1000, type(exc).__name__)

        async def landmarks() -> Any:
            analyzer_frame = frame
            if image is not None and any((self.face_analyzer, self.pose_analyzer, self.hand_analyzer)):
                rgb_image = await asyncio.to_thread(_bgr_to_rgb, image)
                analyzer_frame = {**frame, "rgb_image": rgb_image}
            return await asyncio.gather(
                self._scheduled("face", self.face_analyzer, analyzer_frame),
                self._scheduled("pose", self.pose_analyzer, analyzer_frame),
                self._scheduled("hands", self.hand_analyzer, analyzer_frame),
            )

        async def image_quality() -> dict[str, Any]:
            if self.image_quality is not None and image is not None:
                return await asyncio.to_thread(self.image_quality.extract, image)
            return {"status": "not_run", "reason": "NO_IMAGE"}

        # Independent work on the SAME frame; no cached or mixed-frame evidence.
        # TaskGroup also cancels sibling awaiters when the session is stopped.
        async with asyncio.TaskGroup() as tasks:
            detections_task = tasks.create_task(detect())
            landmarks_task = tasks.create_task(landmarks())
            quality_task = tasks.create_task(image_quality())
        detections, health["object_detector"] = detections_task.result()
        (face, health["face"]), (pose, health["pose"]), (hand, health["hands"]) = landmarks_task.result()
        quality = quality_task.result()
        self._last_run["object_detector"] = monotonic_ms
        for name, detail in health.items():
            if detail.status == "skipped":
                continue
            self._last_health[name] = detail
            if detail.status in {"ready", "no_detection"}:
                self._completed.setdefault(name, deque(maxlen=120)).append(monotonic())
        allowed_classes = getattr(self.detector, "allowed_classes", None)
        if allowed_classes:
            allowed = {str(name).casefold() for name in allowed_classes}
            detections = [
                item for item in detections if str(item.get("class", "")).casefold() in allowed
            ]
        detections = [item for item in detections if str(item.get("class", "")).casefold() != "person"
                      or float(item.get("confidence", 0)) >= self.person_confidence_threshold]
        if self.object_tracker is not None:
            detections = self.object_tracker.update(detections, monotonic_ms)
        try:
            people = await self.tracker.update(list(detections), session_id=session_id, occurred_at=frame.get("captured_at_utc"))
        except TypeError:
            people = await self.tracker.update(list(detections))
        except Exception:
            people = []
            health["tracker"] = ObservationHealth("error", reason="TRACKER_ERROR")
        raw_people = [item for item in detections if str(item.get("class", "")).lower() == "person"]
        if len(raw_people) > 1:
            people = [{**item, "runtime_track_id": index + 1, "track_id": None} for index, item in enumerate(raw_people)]
        return ObservationBundle(
            frame_id=frame_id,
            session_id=session_id,
            captured_at_utc=frame.get("captured_at_utc", datetime.now(UTC)),
            captured_at_monotonic_ms=monotonic_ms,
            frame_size=tuple(frame.get("frame_size", (1, 1))),
            people=list(people or []),
            objects=list(detections),
            faces=list(face.get("faces", [])) if isinstance(face, dict) else [],
            poses=list(pose.get("poses", [])) if isinstance(pose, dict) else [],
            hands=list(hand.get("hands", [])) if isinstance(hand, dict) else [],
            image_quality=dict(quality),
            analyzer_health=health,
        )


def _bgr_to_rgb(image: Any) -> Any:
    """Make one contiguous RGB buffer shared by the MediaPipe analyzers."""

    try:
        import cv2
    except ImportError:
        return image[:, :, ::-1].copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
