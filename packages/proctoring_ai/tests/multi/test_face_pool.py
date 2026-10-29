from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID, uuid4

import pytest
from proctoring_core.observations import Box

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from proctoring_ai.multi.crop import Crop, CropTransform  # noqa: E402
from proctoring_ai.multi.face_pool import PerSubjectFaceAnalyzer  # noqa: E402
from proctoring_ai.multi.motion import MotionGate  # noqa: E402

CENTRE_FACE = {"bbox": {"x": 0.4, "y": 0.4, "width": 0.2, "height": 0.2}, "yaw": 5.0, "pitch": 1.0,
               "quality": 0.9, "pose_valid": True, "face_width": 0.2, "keypoints": {}}


class FakeFace:
    def __init__(self, *, face: dict[str, Any] | None = None, delay: float = 0.0,
                 error: bool = False, calibration: str = "uncalibrated") -> None:
        self.face = face if face is not None else CENTRE_FACE
        self.delay, self.error, self.calibration = delay, error, calibration
        self.calls = 0
        self.rejected = 0
        self.calibrate_calls = 0
        self.closed = False
        self.frames: list[dict[str, Any]] = []

    async def initialize(self) -> dict[str, Any]:
        return {"status": "ready", "reason": None}

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.frames.append(frame)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise RuntimeError("boom")
        return {"status": "ready", "reason": None, "faces": [dict(self.face)]}

    def auto_calibrate(self) -> str:
        self.calibrate_calls += 1
        return self.calibration

    def reject_last_face(self) -> None:
        self.rejected += 1

    def close(self) -> None:
        self.closed = True


def crop_at(x0: int) -> Crop:
    return Crop(np.full((256, 256, 3), 90, np.uint8),
                CropTransform(x0=x0, y0=40, width=160, height=120, frame_width=640, frame_height=480))


def seat_box(crop: Crop) -> Box:
    t = crop.transform
    return Box(t.x0 / 640, t.y0 / 480, t.width / 640, 0.6)


def pool_with(fakes: list[FakeFace], **kwargs: Any) -> PerSubjectFaceAnalyzer:
    queue = list(fakes)
    return PerSubjectFaceAnalyzer(lambda: queue.pop(0), **kwargs)


async def run(pool: PerSubjectFaceAnalyzer, crops: dict[UUID, Crop], ms: int = 0,
              force: set[UUID] | None = None) -> Any:
    boxes = {seat: seat_box(crop) for seat, crop in crops.items()}
    return await pool.analyze(crops, boxes, frame_id=ms // 100 + 1, monotonic_ms=ms, force=force or set())


async def test_each_seat_gets_its_own_instance_and_mapped_face() -> None:
    fakes = [FakeFace(), FakeFace()]
    pool = pool_with(fakes)
    a, b = uuid4(), uuid4()
    crop_a, crop_b = crop_at(0), crop_at(320)
    results = await run(pool, {a: crop_a, b: crop_b})
    assert fakes[0].calls == 1 and fakes[1].calls == 1
    assert results[a].status == "ready" and not results[a].reused
    assert results[a].face["bbox"]["x"] == pytest.approx((0 + 0.4 * 160) / 640)
    assert results[b].face["bbox"]["x"] == pytest.approx((320 + 0.4 * 160) / 640)
    assert fakes[0].frames[0]["image"] is crop_a.image
    assert fakes[0].frames[0]["captured_at_monotonic_ms"] == 0


async def test_seats_run_concurrently() -> None:
    pool = pool_with([FakeFace(delay=0.05) for _ in range(3)])
    crops = {uuid4(): crop_at(index * 200) for index in range(3)}
    started = time.perf_counter()
    await run(pool, crops)
    assert time.perf_counter() - started < 0.12


async def test_still_seat_reuses_previous_result_unless_forced() -> None:
    fake = FakeFace(calibration="calibrated")
    pool = pool_with([fake], motion_gate=MotionGate())
    a = uuid4()
    crop = crop_at(0)
    await run(pool, {a: crop}, 0)
    second = await run(pool, {a: crop}, 100)
    assert second[a].reused and fake.calls == 1
    assert second[a].face == (await run(pool, {a: crop}, 150))[a].face
    third = await run(pool, {a: crop}, 200, force={a})
    assert not third[a].reused and fake.calls == 2


async def test_still_seat_keeps_running_until_calibrated() -> None:
    # A still head is exactly what the neutral-pose suggestion needs; reusing would starve it.
    fake = FakeFace()
    pool = pool_with([fake], motion_gate=MotionGate())
    a = uuid4()
    crop = crop_at(0)
    for ms in (0, 100, 200):
        assert not (await run(pool, {a: crop}, ms))[a].reused
    assert fake.calls == 3
    fake.calibration = "calibrated"
    await run(pool, {a: crop}, 300)
    assert (await run(pool, {a: crop}, 400))[a].reused and fake.calls == 4


async def test_face_centred_outside_seat_box_is_mismatch() -> None:
    behind = {**CENTRE_FACE, "bbox": {"x": 0.9, "y": 0.0, "width": 0.1, "height": 0.1}}
    pool = pool_with([FakeFace(face=behind)])
    a = uuid4()
    crop = crop_at(0)
    narrow = Box(0.0, 40 / 480, 100 / 640, 0.6)  # seat box ends before x0 + 0.95 × width
    results = await pool.analyze({a: crop}, {a: narrow}, frame_id=1, monotonic_ms=0)
    assert results[a].status == "mismatch" and results[a].face is None


async def test_seventh_seat_is_not_run() -> None:
    pool = pool_with([FakeFace() for _ in range(7)], max_instances=6)
    crops = {uuid4(): crop_at(index * 80) for index in range(7)}
    results = await run(pool, crops)
    statuses = [results[seat].status for seat in crops]
    assert statuses[:6] == ["ready"] * 6 and statuses[6] == "not_run"


async def test_one_seat_raising_does_not_affect_others() -> None:
    pool = pool_with([FakeFace(error=True), FakeFace()])
    a, b = uuid4(), uuid4()
    results = await run(pool, {a: crop_at(0), b: crop_at(320)})
    assert results[a].status == "error" and results[b].status == "ready"


async def test_calibration_status_comes_from_auto_calibrate() -> None:
    pool = pool_with([FakeFace(calibration="calibrated")])
    a = uuid4()
    assert (await run(pool, {a: crop_at(0)}))[a].calibration_status == "calibrated"


async def test_release_closes_instance_and_frees_slot() -> None:
    fakes = [FakeFace(), FakeFace()]
    pool = pool_with(fakes, max_instances=1)
    a, b = uuid4(), uuid4()
    await run(pool, {a: crop_at(0)})
    pool.release(a)
    assert fakes[0].closed
    assert (await run(pool, {b: crop_at(0)}))[b].status == "ready"


async def test_mismatch_is_rejected_before_calibration() -> None:
    behind = {**CENTRE_FACE, "bbox": {"x": 0.9, "y": 0.0, "width": 0.1, "height": 0.1}}
    fake = FakeFace(face=behind, calibration="calibrated")
    pool = pool_with([fake])
    a = uuid4()
    narrow = Box(0.0, 40 / 480, 100 / 640, 0.6)
    result = (await pool.analyze({a: crop_at(0)}, {a: narrow}, frame_id=1, monotonic_ms=0))[a]
    assert fake.rejected == 1 and fake.calibrate_calls == 0
    assert result.status == "mismatch"
