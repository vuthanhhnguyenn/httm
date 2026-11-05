from __future__ import annotations

import asyncio
from uuid import uuid4

from cctv_kps import person
from proctoring_ai.multi.cctv.rules import CctvSeatRules
from proctoring_ai.multi.cctv.suspects import CctvSuspectConfig, SuspectTracker
from proctoring_ai.multi.pipeline import FrameTiming, MultiFrameResult
from proctoring_ai.multi.tracker import Seat, TrackerUpdate
from proctoring_core.observations import Box

TIMING = FrameTiming(0.0, 0.0, 0.0, {}, 0.0, 0.0, 0, 0, False)


def seat(cx: float, runtime_id: int, *, top: float = 0.3) -> Seat:
    box = Box(cx - 0.05, top, 0.1, 0.2)
    return Seat(uuid4(), runtime_id, box, box.center, 0.9, 0, 0, True)


def with_wrists(kps, left=None, right=None):
    if left is not None:
        kps[9] = [left[0], left[1], 0.9]
    if right is not None:
        kps[10] = [right[0], right[1], 0.9]
    return kps


def feed(tracker: SuspectTracker, seats, pose_at, start: int, end: int, step: int = 100,
         objects=(), rule_events=None) -> None:
    for ms in range(start, end, step):
        poses = {s.seat_id: pose_at(s, ms) for s in seats}
        fired = rule_events(ms) if rule_events else ()
        tracker.observe({s.seat_id: s for s in seats}, poses, list(objects), ms, ms, 1.0, fired)


def calm(s: Seat, _ms: int):
    return person(cx=s.bbox.x + 0.05, top=s.bbox.y)


def test_head_away_without_lift_opens_window() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    feed(tracker, [a], lambda s, ms: person(cx=0.3, top=0.3, bent=True,
                                            nose_dx=0.3 if ms >= 1500 else 0.0), 0, 3000)
    windows = tracker.finish(None)
    assert len(windows) == 1 and windows[0].seat_id == a.seat_id
    assert "HEAD_AWAY" in windows[0].reasons


def test_condition_shorter_than_hold_is_ignored() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    feed(tracker, [a], lambda s, ms: person(cx=0.3, top=0.3, bent=True,
                                            nose_dx=0.3 if 1500 <= ms < 2300 else 0.0), 0, 3500)
    assert tracker.finish(None) == []


def test_hands_meet_across_rows() -> None:
    tracker, back, front = SuspectTracker(), seat(0.5, 1, top=0.3), seat(0.5, 2, top=0.5)

    def pose(s: Seat, _ms: int):
        wrist = (0.55, 0.5) if s is back else (0.55, 0.52)
        return with_wrists(person(cx=0.5, top=s.bbox.y), left=wrist)

    feed(tracker, [back, front], pose, 0, 1500)
    windows = {w.seat_id: w for w in tracker.finish(None)}
    assert set(windows) == {back.seat_id, front.seat_id}
    assert all("HANDS_MEET" in w.reasons for w in windows.values())
    assert front.seat_id in windows[back.seat_id].neighbor_ids
    assert back.seat_id in windows[front.seat_id].neighbor_ids


def test_head_down_hands_low_uses_desk_line() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)

    def pose(s: Seat, ms: int):
        y = 0.41 if ms < 5000 else 0.55
        return with_wrists(person(cx=0.3, top=0.3, bent=True), left=(0.28, y), right=(0.32, y))

    feed(tracker, [a], pose, 0, 6500)
    windows = tracker.finish(None)
    assert [w.reasons for w in windows] == [("HEAD_DOWN_HANDS_LOW",)]
    assert windows[0].start_ms >= 5000 - CctvSuspectConfig().pad_ms


def test_object_hint_low_confidence_phone_in_seat() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    phone = {"class": "cell phone", "confidence": 0.2, "bbox": {"x": 0.29, "y": 0.4, "width": 0.02, "height": 0.02}}
    feed(tracker, [a], calm, 0, 1500, objects=[phone])
    assert [w.reasons for w in tracker.finish(None)] == [("OBJECT_HINT",)]


def test_windows_padded_clamped_and_merged() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    feed(tracker, [a], calm, 0, 5000, rule_events=lambda ms: {a.seat_id} if ms in (500, 4000) else ())
    windows = tracker.finish(5000)
    assert len(windows) == 1
    assert windows[0].start_ms == 0 and windows[0].end_ms == 5000
    assert windows[0].reasons == ("RULE_EVENT",)
    assert windows[0].to_json()["bbox"] == {"x": a.bbox.x, "y": a.bbox.y, "width": a.bbox.width,
                                            "height": a.bbox.height}


def _frame(ms: int, seats, poses) -> MultiFrameResult:
    update = TrackerUpdate(list(seats), [], [], [], 0, 0)
    return MultiFrameResult(ms // 100, ms, update, {}, {}, {}, [], TIMING,
                            {s.seat_id: kps for s, kps in zip(seats, poses, strict=True)})


def _play(rules: CctvSeatRules, seats, pose_at, start: int, end: int) -> list[dict]:
    events = []
    for ms in range(start, end, 100):
        frame = _frame(ms, seats, [pose_at(s, ms) for s in seats])
        events += [item for _, item in asyncio.run(rules.step(frame, {}, ms))]
    return events


def test_rule_events_unchanged_and_fed_to_tracker() -> None:
    def turned(s: Seat, ms: int):
        return person(cx=s.bbox.x + 0.05, top=s.bbox.y, nose_dx=0.6 if ms >= 10_000 and s.runtime_track_id == 1 else 0.0)

    a, b = seat(0.3, 1), seat(0.45, 2)
    plain = _play(CctvSeatRules(), [a, b], turned, 0, 12_000)
    tracker = SuspectTracker()
    tracked = _play(CctvSeatRules(suspects=tracker), [a, b], turned, 0, 12_000)
    strip = lambda events: [(e["action"], e["type"], e["seat_id"], e["video_offset_ms"]) for e in events]  # noqa: E731
    assert strip(plain) == strip(tracked) and strip(plain)
    windows = [w for w in tracker.finish(12_000) if w.seat_id == a.seat_id]
    assert windows and "RULE_EVENT" in windows[0].reasons


def test_peak_is_moment_with_most_reasons() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    # HEAD_AWAY is marked from 2.5 s (held 1 s since 1.5 s); a rule event joins it at 2.6-2.8 s.
    feed(tracker, [a], _away(1500, 3500), 0, 4000, rule_events=lambda ms: [a.seat_id] if 2600 <= ms <= 2800 else [])
    (window,) = tracker.finish(None)
    assert {"HEAD_AWAY", "RULE_EVENT"} <= set(window.reasons)
    assert window.peak_ms == 2600


def _away(start, end):
    return lambda s, ms: person(cx=0.3, top=0.3, bent=True, nose_dx=0.3 if start <= ms < end else 0.0)


def _always(a):
    return lambda ms: [a.seat_id]


def test_poll_returns_closed_window_once() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    feed(tracker, [a], calm, 0, 4000, rule_events=lambda ms: [a.seat_id] if 1500 <= ms < 3000 else [])
    assert tracker.poll(4000) == []  # could still grow
    feed(tracker, [a], calm, 4000, 12000)
    (window,) = tracker.poll(12000)
    assert window.start_ms <= 1500 and window.end_ms >= 3000
    assert tracker.poll(13000) == [] and tracker.finish(None) == []


def test_poll_splits_long_open_span_at_10s() -> None:
    tracker, a = SuspectTracker(), seat(0.3, 1)
    feed(tracker, [a], calm, 0, 14000, rule_events=lambda ms: [a.seat_id] if ms >= 1500 else [])
    (first,) = tracker.poll(14000, max_open_ms=10000)
    assert first.end_ms - first.start_ms == 10000
    feed(tracker, [a], calm, 14000, 30000, rule_events=lambda ms: [a.seat_id] if ms < 16000 else [])
    rest = tracker.poll(30000) + tracker.finish(None)
    assert len(rest) == 1 and rest[0].start_ms == first.end_ms
