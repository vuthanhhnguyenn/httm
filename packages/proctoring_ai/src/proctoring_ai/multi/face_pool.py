"""One face landmarker per seat, run concurrently on per-seat head crops.

MediaPipe VIDEO mode keeps tracking state, and the head-pose estimator keeps the
seat's neutral baseline, so each seat owns its instance for as long as it exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Protocol
from uuid import UUID

from proctoring_core.observations import Box

from .crop import Crop
from .motion import MotionGate


class FaceAnalyzerLike(Protocol):
    async def initialize(self) -> dict[str, Any]: ...

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]: ...

    def auto_calibrate(self) -> str: ...

    def reject_last_face(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FaceResult:
    status: str
    face: dict[str, Any] | None
    reused: bool
    latency_ms: float
    calibration_status: str


_NOT_RUN = FaceResult("not_run", None, False, 0.0, "uncalibrated")


def _inside(point: tuple[float, float], box: Box) -> bool:
    return box.x <= point[0] <= box.x + box.width and box.y <= point[1] <= box.y + box.height


class PerSubjectFaceAnalyzer:
    def __init__(self, factory: Callable[[], FaceAnalyzerLike], *, max_instances: int = 6,
                 motion_gate: MotionGate | None = None) -> None:
        self.factory = factory
        self.max_instances = max_instances
        self.motion_gate = motion_gate
        self._instances: dict[UUID, FaceAnalyzerLike] = {}
        self._last: dict[UUID, FaceResult] = {}

    async def _instance(self, seat_id: UUID) -> FaceAnalyzerLike | None:
        instance = self._instances.get(seat_id)
        if instance is None and len(self._instances) < self.max_instances:
            instance = self.factory()
            await instance.initialize()
            self._instances[seat_id] = instance
        return instance

    async def _run(self, seat_id: UUID, crop: Crop, seat_box: Box | None, frame_id: int,
                   monotonic_ms: int) -> FaceResult:
        instance = await self._instance(seat_id)
        if instance is None:
            return _NOT_RUN
        started = perf_counter()
        frame = {"image": crop.image, "frame_id": frame_id, "captured_at_monotonic_ms": monotonic_ms,
                 "frame_size": (crop.image.shape[1], crop.image.shape[0])}
        try:
            raw = await instance.analyze(frame)
        except Exception:
            raw = {"status": "error", "faces": []}
        latency = (perf_counter() - started) * 1000
        if self.motion_gate is not None:
            self.motion_gate.record(seat_id, crop.image, monotonic_ms)
        status = str(raw.get("status", "error"))
        faces = raw.get("faces") or []
        if status == "ready" and faces:
            face = crop.transform.map_face(faces[0])
            bbox = face.get("bbox")
            if seat_box is not None and isinstance(bbox, Mapping):
                centre = (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
                if not _inside(centre, seat_box):
                    # The candidate seated behind: never attribute it, nor let it set this seat's baseline.
                    instance.reject_last_face()
                    previous = self._last.get(seat_id)
                    return FaceResult("mismatch", None, False, latency,
                                      previous.calibration_status if previous else "uncalibrated")
        calibration = instance.auto_calibrate()
        if status != "ready" or not faces:
            return FaceResult(status if status != "ready" else "no_detection", None, False, latency, calibration)
        return FaceResult("ready", face, False, latency, calibration)

    async def analyze(self, crops: Mapping[UUID, Crop], seat_boxes: Mapping[UUID, Box], *, frame_id: int,
                      monotonic_ms: int, force: Set[UUID] = frozenset()) -> dict[UUID, FaceResult]:
        results: dict[UUID, FaceResult] = {}
        pending: dict[UUID, Any] = {}
        for seat_id, crop in crops.items():
            previous = self._last.get(seat_id)
            if (previous is not None and self.motion_gate is not None and seat_id not in force
                    and not self.motion_gate.should_run(seat_id, crop.image, monotonic_ms)):
                results[seat_id] = replace(previous, reused=True, latency_ms=0.0)
                continue
            pending[seat_id] = self._run(seat_id, crop, seat_boxes.get(seat_id), frame_id, monotonic_ms)
        for seat_id, result in zip(pending, await asyncio.gather(*pending.values()), strict=True):
            results[seat_id] = result
            if result is not _NOT_RUN:
                self._last[seat_id] = result
        return {seat_id: results[seat_id] for seat_id in crops}

    def release(self, seat_id: UUID) -> None:
        instance = self._instances.pop(seat_id, None)
        self._last.pop(seat_id, None)
        if self.motion_gate is not None:
            self.motion_gate.forget(seat_id)
        if instance is not None:
            instance.close()

    def close(self) -> None:
        for seat_id in list(self._instances):
            self.release(seat_id)


__all__ = ["FaceAnalyzerLike", "FaceResult", "PerSubjectFaceAnalyzer"]
