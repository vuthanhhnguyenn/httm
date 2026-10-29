from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from proctoring_ai.multi.bundles import assign_objects, build_subject_bundle
from proctoring_ai.multi.face_pool import FaceResult
from proctoring_ai.multi.focus_analyzers import FocusResult
from proctoring_ai.multi.tracker import Seat
from proctoring_core.events.models import ObservationHealth
from proctoring_core.observations import Box, Observation

FRAME = {"frame_id": 3, "captured_at_monotonic_ms": 300, "captured_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
         "frame_size": (640, 480)}
FACE = {"bbox": {"x": 0.12, "y": 0.32, "width": 0.05, "height": 0.08}, "yaw": 3.0, "pitch": 1.0,
        "quality": 0.9, "pose_valid": True, "calibration_status": "calibrated"}


def seat(x: float, runtime_id: int = 1) -> Seat:
    box = Box(x, 0.3, 0.15, 0.4)
    return Seat(uuid4(), runtime_id, box, box.center, 0.9, 0, 300, True)


def phone(x: float, y: float = 0.5) -> dict[str, Any]:
    return {"class": "cell phone", "confidence": 0.8, "bbox": {"x": x, "y": y, "width": 0.04, "height": 0.06}}


def build(s: Seat, **kwargs: Any) -> Any:
    options: dict[str, Any] = {"frame": FRAME, "session_id": uuid4(), "seat": s, "visible": True,
                               "face": FaceResult("ready", FACE, False, 5.0, "calibrated"), "focus": None,
                               "objects": [], "image_quality": {"status": "ready"},
                               "detector_health": ObservationHealth("ready", 20.0)}
    options.update(kwargs)
    return build_subject_bundle(**options)


def test_phone_goes_to_seat_with_largest_overlap() -> None:
    left, right = seat(0.10), seat(0.30, 2)
    assigned = assign_objects([phone(0.24), {"class": "person", "bbox": {"x": 0.1, "y": 0.3,
                                                                          "width": 0.1, "height": 0.1}}],
                              [left, right])
    assert [item["class"] for item in assigned[left.seat_id]] == ["cell phone"]
    assert assigned[right.seat_id] == []


def test_phone_just_outside_box_is_kept_by_20_percent_expansion() -> None:
    s = seat(0.10)  # box ends at x = 0.25; expanded box ends at 0.28
    assert len(assign_objects([phone(0.255)], [s])[s.seat_id]) == 1


def test_object_far_from_every_seat_is_dropped() -> None:
    s = seat(0.10)
    assert assign_objects([phone(0.8)], [s]) == {s.seat_id: []}


def test_visible_seat_bundle_contains_only_its_person_and_face() -> None:
    s = seat(0.10)
    bundle = build(s, objects=[phone(0.2)])
    assert [p["track_id"] for p in bundle.people] == [str(s.seat_id)]
    assert bundle.people[0]["runtime_track_id"] == 1
    assert bundle.faces == [FACE] and bundle.objects == [phone(0.2)]
    assert bundle.analyzer_health["face"].status == "ready"
    assert bundle.analyzer_health["hands"].reason == "NOT_IN_FOCUS"
    assert (bundle.frame_id, bundle.captured_at_monotonic_ms, bundle.frame_size) == (3, 300, (640, 480))
    observation = Observation.from_bundle(bundle)
    assert observation.people[0].track_id == str(s.seat_id)
    assert observation.analyzer_status("object_detector") == "ready"


def test_missing_seat_bundle_has_no_person_and_face_not_run() -> None:
    bundle = build(seat(0.10), visible=False, face=None)
    assert bundle.people == [] and bundle.faces == []
    assert bundle.analyzer_health["face"].status == "not_run"
    assert bundle.analyzer_health["face"].reason == "SUBJECT_NOT_VISIBLE"


def test_mismatch_face_is_reported_as_no_detection() -> None:
    bundle = build(seat(0.10), face=FaceResult("mismatch", None, False, 5.0, "calibrated"))
    assert bundle.faces == [] and bundle.analyzer_health["face"].status == "no_detection"


def test_focus_hands_and_pose_are_attached() -> None:
    hand = {"bbox": FACE["bbox"], "wrist": (0.1, 0.5), "index_tip": (0.1, 0.5), "landmarks": []}
    focus = FocusResult([hand], [], "ready", "skipped", 8.0)
    bundle = build(seat(0.10), focus=focus)
    assert bundle.hands == [hand] and bundle.poses == []
    assert bundle.analyzer_health["hands"].status == "ready"
    assert bundle.analyzer_health["pose"].status == "skipped"
