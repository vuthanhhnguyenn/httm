"""Compose detection, seat tracking and per-seat analysis for one frame.

The work is split in two stages so a caller can overlap them: ``detect`` (YOLO on
the GPU, then the tracker) for frame t+1 can run while ``analyze_subjects``
(MediaPipe on the CPU) is still busy with frame t.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any
from uuid import UUID

from proctoring_core.events.models import ObservationBundle, ObservationHealth

from ..pipeline.single_flight import InferenceBusyError
from .bundles import assign_objects, build_subject_bundle
from .crop import Crop, SubjectCropper
from .face_pool import FaceResult, PerSubjectFaceAnalyzer
from .focus import FocusScheduler
from .focus_analyzers import FocusAnalyzers, FocusResult
from .tracker import MultiSubjectTracker, Seat, TrackerUpdate


@dataclass(frozen=True, slots=True)
class FrameTiming:
    detect_ms: float
    track_ms: float
    face_ms: float
    face_each_ms: dict[UUID, float]
    focus_ms: float
    total_ms: float
    fresh_faces: int
    reused_faces: int
    detector_reused: bool


@dataclass(frozen=True, slots=True)
class DetectionStage:
    frame: dict[str, Any]
    detections: list[dict[str, Any]]
    tracker: TrackerUpdate
    detector_health: ObservationHealth
    detect_ms: float
    track_ms: float
    detector_reused: bool


@dataclass(frozen=True, slots=True)
class MultiFrameResult:
    frame_id: int
    monotonic_ms: int
    tracker: TrackerUpdate
    bundles: dict[UUID, ObservationBundle]
    faces: dict[UUID, FaceResult]
    focus: dict[UUID, FocusResult]
    objects: list[dict[str, Any]]
    timing: FrameTiming


def _snapshot(update: TrackerUpdate) -> TrackerUpdate:
    """Copy seats so detecting frame t+1 cannot move the boxes frame t is analysed with."""

    def copy(seats: list[Seat]) -> list[Seat]:
        return [replace(seat) for seat in seats]

    return replace(update, analyzed=copy(update.analyzed), over_limit=copy(update.over_limit),
                   missing=copy(update.missing), expired=copy(update.expired))


class MultiSubjectPipeline:
    def __init__(self, *, detector: Any, tracker: MultiSubjectTracker, cropper: SubjectCropper,
                 faces: PerSubjectFaceAnalyzer, focus_analyzers: FocusAnalyzers, scheduler: FocusScheduler,
                 image_quality: Any | None = None, object_tracker: Any | None = None,
                 person_confidence_threshold: float = 0.6, yolo_every: int = 1, timeout_ms: int = 1200) -> None:
        self.detector = detector
        self.tracker = tracker
        self.cropper = cropper
        self.faces = faces
        self.focus_analyzers = focus_analyzers
        self.scheduler = scheduler
        self.image_quality = image_quality
        self.object_tracker = object_tracker
        self.person_confidence_threshold = person_confidence_threshold
        self.yolo_every = max(1, yolo_every)
        self.timeout_ms = timeout_ms
        self._frames_seen = 0
        self._last_detections: list[dict[str, Any]] | None = None
        self._last_total_ms: float | None = None

    async def _run_detector(self, frame: dict[str, Any]) -> tuple[list[dict[str, Any]], ObservationHealth]:
        started = perf_counter()
        try:
            result = await asyncio.wait_for(self.detector.detect(frame), timeout=self.timeout_ms / 1000)
            return list(result), ObservationHealth("ready", (perf_counter() - started) * 1000)
        except InferenceBusyError:
            return [], ObservationHealth("busy", (perf_counter() - started) * 1000, "INFERENCE_ALREADY_RUNNING")
        except TimeoutError:
            return [], ObservationHealth("timeout", (perf_counter() - started) * 1000, "DETECTOR_TIMEOUT")
        except Exception as exc:
            return [], ObservationHealth("error", (perf_counter() - started) * 1000, type(exc).__name__)

    async def detect(self, frame: dict[str, Any]) -> DetectionStage:
        index = self._frames_seen
        self._frames_seen += 1
        reuse = self.yolo_every > 1 and index % self.yolo_every != 0 and self._last_detections is not None
        if reuse and self._last_detections is not None:
            detections = list(self._last_detections)
            health = ObservationHealth("ready", 0.0, "REUSED_DETECTIONS")
        else:
            detections, health = await self._run_detector(frame)
            self._last_detections = detections if health.status == "ready" else None
        monotonic_ms = int(frame.get("captured_at_monotonic_ms", 0))
        detections = [item for item in detections if str(item.get("class", "")).casefold() != "person"
                      or float(item.get("confidence", 0.0)) >= self.person_confidence_threshold]
        if self.object_tracker is not None:
            detections = self.object_tracker.update(detections, monotonic_ms)
        started = perf_counter()
        people = [item for item in detections if str(item.get("class", "")).casefold() == "person"]
        update = _snapshot(self.tracker.update(people, monotonic_ms))
        return DetectionStage(frame, detections, update, health, health.latency_ms,
                              (perf_counter() - started) * 1000, reuse)

    def mark_focus(self, seat_id: UUID, monotonic_ms: int) -> None:
        self.scheduler.mark(seat_id, monotonic_ms)

    async def _quality(self, image: Any) -> dict[str, Any]:
        if self.image_quality is None or image is None:
            return {"status": "not_run", "reason": "NO_IMAGE"}
        return dict(await asyncio.to_thread(self.image_quality.extract, image))

    async def analyze_subjects(self, stage: DetectionStage, *, session_id: UUID) -> MultiFrameResult:
        started = perf_counter()
        frame, update = stage.frame, stage.tracker
        frame_id = int(frame.get("frame_id", 0))
        monotonic_ms = int(frame.get("captured_at_monotonic_ms", 0))
        image = frame.get("image")
        for seat in update.expired:
            self.faces.release(seat.seat_id)
            self.focus_analyzers.release(seat.seat_id)
            self.scheduler.forget(seat.seat_id)
        seats = {seat.seat_id: seat for seat in update.analyzed}
        plan = self.scheduler.plan(list(seats), monotonic_ms, last_frame_ms=self._last_total_ms)
        # Seat order (not set order) keeps instance creation and outputs deterministic.
        face_seats = [seat_id for seat_id in seats if seat_id in plan.face_seats]
        head_crops: dict[UUID, Crop] = {}
        if image is not None:
            for seat_id in face_seats:
                crop = self.cropper.crop_head(image, seats[seat_id].bbox)
                if crop is not None:
                    head_crops[seat_id] = crop
        body_crops: dict[UUID, Crop] = {}
        if plan.run_focus and image is not None:
            for seat_id in plan.focus_seats:
                crop = self.cropper.crop_body(image, seats[seat_id].bbox)
                if crop is not None:
                    body_crops[seat_id] = crop

        async def run_faces() -> tuple[dict[UUID, FaceResult], float]:
            began = perf_counter()
            boxes = {seat_id: seats[seat_id].bbox for seat_id in head_crops}
            result = await self.faces.analyze(head_crops, boxes, frame_id=frame_id, monotonic_ms=monotonic_ms,
                                              force=plan.force_face)
            return result, (perf_counter() - began) * 1000

        async def run_focus() -> tuple[dict[UUID, FocusResult], float]:
            began = perf_counter()
            if not body_crops:
                return {}, 0.0
            result = await self.focus_analyzers.analyze(body_crops, frame_id=frame_id, monotonic_ms=monotonic_ms)
            return result, (perf_counter() - began) * 1000

        (faces, face_ms), (focus, focus_ms), quality = await asyncio.gather(
            run_faces(), run_focus(), self._quality(image))
        for seat_id in face_seats:
            if seat_id not in faces:  # degenerate crop: analysed, nothing to see
                faces[seat_id] = FaceResult("no_detection", None, False, 0.0, "uncalibrated")
            self.scheduler.record_face(seat_id, monotonic_ms)
        assigned = assign_objects(stage.detections, list(seats.values()))
        bundles: dict[UUID, ObservationBundle] = {}
        for seat_id in face_seats:
            bundles[seat_id] = build_subject_bundle(
                frame=frame, session_id=session_id, seat=seats[seat_id], visible=True, face=faces[seat_id],
                focus=focus.get(seat_id), objects=assigned.get(seat_id, []), image_quality=quality,
                detector_health=stage.detector_health)
        for seat in update.missing:
            bundles[seat.seat_id] = build_subject_bundle(
                frame=frame, session_id=session_id, seat=seat, visible=False, face=None, focus=None,
                objects=[], image_quality=quality, detector_health=stage.detector_health)
        total_ms = stage.detect_ms + stage.track_ms + (perf_counter() - started) * 1000
        self._last_total_ms = total_ms
        fresh = {seat_id: result.latency_ms for seat_id, result in faces.items()
                 if not result.reused and result.status not in {"not_run"}}
        timing = FrameTiming(stage.detect_ms, stage.track_ms, face_ms, fresh, focus_ms, total_ms,
                             len(fresh), sum(1 for result in faces.values() if result.reused),
                             stage.detector_reused)
        return MultiFrameResult(frame_id, monotonic_ms, update, bundles, faces, focus,
                                list(stage.detections), timing)

    async def analyze(self, frame: dict[str, Any], *, session_id: UUID) -> MultiFrameResult:
        return await self.analyze_subjects(await self.detect(frame), session_id=session_id)

    def close(self) -> None:
        self.faces.close()
        self.focus_analyzers.close()


__all__ = ["DetectionStage", "FrameTiming", "MultiFrameResult", "MultiSubjectPipeline"]
