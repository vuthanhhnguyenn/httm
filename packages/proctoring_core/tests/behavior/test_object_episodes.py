from __future__ import annotations

from typing import Any
from uuid import uuid4

from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.types import EventType

BOX = {"x": 0.5, "y": 0.6, "width": 0.08, "height": 0.12}


def phone(track: int, confidence: float = 0.7) -> dict[str, Any]:
    return {"class": "cell phone", "confidence": confidence, "bbox": dict(BOX), "object_track_id": track,
            "track_confirmed": True, "model_name": "yolo11n", "model_version": "8.3.0"}


def run(timeline: list[tuple[int, list[dict[str, Any]]]], **kwargs: Any) -> list[dict[str, Any]]:
    rules = ObjectRuleEvaluator(phone_min_duration_ms=600, phone_episode_gap_ms=1500, max_gap_ms=600, **kwargs)
    session = uuid4()
    output = []
    for frame_id, (timestamp, objects) in enumerate(timeline, start=1):
        for item in rules.evaluate(session_id=session, track_id=None, objects=objects, monotonic_ms=timestamp,
                                   frame_id=frame_id):
            if item["event_type"] is EventType.PHONE_DETECTED:
                output.append({**item, "t": timestamp})
    return output


def transitions(results: list[dict[str, Any]]) -> list[tuple[int, bool]]:
    changes, previous = [], False
    for item in results:
        if item["active"] != previous:
            changes.append((item["t"], item["active"]))
            previous = item["active"]
    return changes


def test_dropouts_and_track_switch_stay_one_episode() -> None:
    timeline = [(t, [phone(1)]) for t in range(0, 3000, 125)]
    timeline += [(t, []) for t in range(3000, 3500, 125)]          # 0.5 s dropout
    timeline += [(t, [phone(2)]) for t in range(3500, 6000, 125)]  # new object track id
    timeline += [(t, []) for t in range(6000, 8000, 125)]

    changes = transitions(run(timeline))

    # Last sighting 5875 ms; the first sample more than 1.5 s later closes the episode.
    assert changes == [(625, True), (7500, False)]


def test_long_absence_closes_episode_and_needs_fresh_confirmation() -> None:
    timeline = [(t, [phone(1)]) for t in range(0, 2000, 125)]
    timeline += [(t, []) for t in range(2000, 5000, 125)]
    timeline += [(t, [phone(3)]) for t in range(5000, 7000, 125)]

    changes = transitions(run(timeline))

    assert [active for _, active in changes] == [True, False, True]
    assert changes[1][0] == 3500  # first sample more than 1.5 s after the last sighting at 1875 ms
    assert changes[2][0] >= 5600


def test_episode_carries_provenance_of_the_opening_detection() -> None:
    results = run([(t, [phone(7, 0.6 + t / 100_000)]) for t in range(0, 2000, 125)])
    opened = next(item for item in results if item["active"])

    trace = opened["object_trace"]
    assert trace["frame_id"] == 6 and trace["monotonic_ms"] == 625
    assert trace["object_track_id"] == 7 and trace["bbox"] == BOX
    assert trace["model_name"] == "yolo11n" and trace["model_version"] == "8.3.0"
    later = [item for item in results if item["active"]][-1]
    assert later["object_trace"] == trace


def test_policy_allowing_phone_never_opens_or_holds_an_episode() -> None:
    rules = ObjectRuleEvaluator({"allow_phone": True}, phone_min_duration_ms=0, phone_episode_gap_ms=1500)
    session = uuid4()
    for timestamp in range(0, 2000, 125):
        result = rules.evaluate(session_id=session, track_id=None, objects=[phone(1)], monotonic_ms=timestamp)
        assert not any(item["active"] for item in result)
