from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from proctoring_core.types import EventType

np = pytest.importorskip("numpy")

from proctoring_ai.multi.crop import Crop, CropTransform  # noqa: E402
from proctoring_ai.multi.focus import FOCUS_EVENT_TYPES, FocusScheduler  # noqa: E402
from proctoring_ai.multi.focus_analyzers import FocusAnalyzers  # noqa: E402


def test_focus_event_types() -> None:
    assert {EventType.HEAD_TURN_LEFT, EventType.HEAD_TURN_RIGHT, EventType.LOOK_DOWN,
                                 EventType.PHONE_DETECTED, EventType.LOOK_AT_NEIGHBOR} == FOCUS_EVENT_TYPES


def test_marked_seats_are_focus_for_5_s_max_two_newest_first() -> None:
    s = FocusScheduler()
    a, b, c = uuid4(), uuid4(), uuid4()
    s.mark(a, 0)
    s.mark(b, 100)
    s.mark(c, 200)
    assert s.plan([a, b, c], 300, last_frame_ms=50).focus_seats == (c, b)
    assert s.plan([a, b, c], 5_300, last_frame_ms=50).focus_seats == ()


def test_focus_only_for_present_seats() -> None:
    s = FocusScheduler()
    a, b = uuid4(), uuid4()
    s.mark(a, 0)
    assert s.plan([b], 100, last_frame_ms=None).focus_seats == ()


def test_within_budget_runs_every_face_and_focus() -> None:
    s = FocusScheduler()
    a, b = uuid4(), uuid4()
    s.record_face(a, 0)
    s.record_face(b, 0)
    plan = s.plan([a, b], 100, last_frame_ms=110.0)
    assert plan.run_focus and plan.face_seats == {a, b}


def test_over_budget_skips_focus_and_rotates_stable_seats_at_4_hz() -> None:
    s = FocusScheduler()
    a, b = uuid4(), uuid4()
    s.record_face(a, 0)
    s.record_face(b, 100)
    plan = s.plan([a, b], 260, last_frame_ms=140.0)
    assert not plan.run_focus and plan.face_seats == {a}


def test_new_and_focus_seats_are_forced_even_over_budget() -> None:
    s = FocusScheduler()
    old, new, focused = uuid4(), uuid4(), uuid4()
    s.record_face(old, 200)
    s.record_face(focused, 200)
    s.mark(focused, 200)
    plan = s.plan([old, new, focused], 300, last_frame_ms=500.0)
    assert plan.face_seats == {new, focused}
    assert plan.force_face == {new, focused}


def test_forget_drops_focus_and_face_history() -> None:
    s = FocusScheduler()
    a = uuid4()
    s.record_face(a, 0)
    s.mark(a, 0)
    s.forget(a)
    plan = s.plan([a], 100, last_frame_ms=None)
    assert plan.focus_seats == () and a in plan.force_face


class FakeLandmarker:
    instances: list[FakeLandmarker] = []

    def __init__(self, key: str) -> None:
        self.key, self.calls, self.closed = key, 0, False
        FakeLandmarker.instances.append(self)

    async def initialize(self) -> dict[str, Any]:
        return {"status": "ready"}

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        item = {"bbox": {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1},
                "landmarks": [{"x": 0.5, "y": 0.5}]}
        if self.key == "hands":
            return {"status": "ready", "hands": [{**item, "wrist": (0.5, 0.5), "index_tip": (0.5, 0.5)}]}
        return {"status": "ready", "poses": [item]}

    def close(self) -> None:
        self.closed = True


def body(x0: int = 0) -> Crop:
    return Crop(np.zeros((64, 64, 3), np.uint8), CropTransform(x0, 0, 320, 480, 640, 480))


def analyzers(**kwargs: Any) -> FocusAnalyzers:
    FakeLandmarker.instances = []
    return FocusAnalyzers(lambda: FakeLandmarker("hands"), lambda: FakeLandmarker("pose"), **kwargs)


async def test_focus_results_are_mapped_to_frame() -> None:
    focus = analyzers()
    a = uuid4()
    result = (await focus.analyze({a: body(320)}, frame_id=1, monotonic_ms=0))[a]
    assert result.hands_status == "ready" and result.pose_status == "ready"
    assert result.hands[0]["wrist"] == pytest.approx(((320 + 160) / 640, 0.5))
    assert result.poses[0]["landmarks"][0]["x"] == pytest.approx(0.75)


async def test_focus_slot_rebinds_and_closes_previous_instance() -> None:
    focus = analyzers(slots=1)
    a, b = uuid4(), uuid4()
    await focus.analyze({a: body()}, frame_id=1, monotonic_ms=0)
    first = list(FakeLandmarker.instances)
    await focus.analyze({b: body()}, frame_id=2, monotonic_ms=1000)
    assert all(item.closed for item in first)
    assert len(FakeLandmarker.instances) == 4


async def test_only_slots_count_seats_are_analysed() -> None:
    focus = analyzers(slots=2)
    crops = {uuid4(): body() for _ in range(3)}
    results = await focus.analyze(crops, frame_id=1, monotonic_ms=0)
    assert len(results) == 2


async def test_hands_respect_interval() -> None:
    focus = analyzers(hands_interval_ms=200, pose_interval_ms=250)
    a = uuid4()
    await focus.analyze({a: body()}, frame_id=1, monotonic_ms=0)
    second = (await focus.analyze({a: body()}, frame_id=2, monotonic_ms=100))[a]
    assert second.hands_status == "skipped" and second.pose_status == "skipped"
    third = (await focus.analyze({a: body()}, frame_id=3, monotonic_ms=200))[a]
    assert third.hands_status == "ready" and third.pose_status == "skipped"


async def test_missing_factory_reports_not_run() -> None:
    focus = FocusAnalyzers(None, None)
    a = uuid4()
    result = (await focus.analyze({a: body()}, frame_id=1, monotonic_ms=0))[a]
    assert (result.hands_status, result.pose_status) == ("not_run", "not_run")
