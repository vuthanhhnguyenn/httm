"""Deep tier of phase 2e: decide what happened in one suspect window.

Input is every analysed frame of the window (plus a warm-up before it, so baselines are ready),
with keypoints from the big pose model and objects from all review detectors, already in full-frame
coordinates. Output uses the ``events.jsonl`` record shape of the fast tier with ``stage: review``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID, uuid4

from proctoring_core.observations import Box
from proctoring_core.types import EventType

from ..pipeline import FrameTiming, MultiFrameResult
from ..tracker import Seat, TrackerUpdate
from .head import Keypoints, distance, head_lift, head_point, point, shoulders
from .pose_detector import LEFT_WRIST, RIGHT_WRIST
from .review_detectors import ReviewObject
from .roles import CANDIDATE
from .rules import CctvRuleConfig, CctvSeatRules
from .suspects import CctvSuspectConfig, DeskLine, SuspectWindow

_TIMING = FrameTiming(0.0, 0.0, 0.0, {}, 0.0, 0.0, 0, 0, False)


@dataclass(frozen=True, slots=True)
class ReviewRuleConfig:
    phone_min_conf: float = 0.35
    phone_wrist_widths: float = 0.3
    phone_frames: int = 3
    # A calculator lies on the desk by the writing hand; a phone in use is held up off the desk.
    phone_raise_heights: float = 0.1
    phone_span_ms: int = 1000
    passing_wrist_widths: float = 0.3
    passing_hold_ms: int = 300
    passing_transfer_ms: int = 2000
    document_hold_ms: int = 1000
    document_face_widths: float = 0.3
    lap_hold_ms: int = 3000
    min_head_lift: float = 0.1
    gap_ms: int = 300
    home_min_ms: int = 5000  # the warm-up before a window is shorter than the fast tier's 10 s


@dataclass(frozen=True, slots=True)
class ReviewFrame:
    offset_ms: int
    seats: Mapping[UUID, Box]
    poses: Mapping[UUID, Keypoints]
    objects: Sequence[ReviewObject]
    frame_aspect: float = 16 / 9


def _person_head(item: Mapping[str, Any]) -> tuple[float, float] | None:
    """The person's own head place: face points, else the shoulder midpoint, else the top of their box.
    Never the seat's box: a faceless person would otherwise look like they sit in every seat."""

    kps = item["keypoints"]
    face = [p for index in (0, 1, 2, 3, 4) if (p := point(kps, index))]
    if face:
        return sum(p[0] for p in face) / len(face), sum(p[1] for p in face) / len(face)
    if (pair := shoulders(kps)) is not None:
        return (pair[0][0] + pair[1][0]) / 2, (pair[0][1] + pair[1][1]) / 2
    if (box := Box.parse(item.get("bbox"))) is not None:
        return box.x + box.width / 2, box.y + 0.12 * box.height
    return None


def assign_poses(seats: Mapping[UUID, Box], people: Sequence[Mapping[str, Any]]) -> dict[UUID, Keypoints]:
    """Each seat takes the person whose head is inside its box and nearest its usual head place."""

    out: dict[UUID, Keypoints] = {}
    taken: set[int] = set()
    for seat_id, box in seats.items():
        target = (box.x + box.width / 2, box.y + 0.25 * box.height)
        best, best_d = None, float("inf")
        for index, item in enumerate(people):
            if index in taken:
                continue
            head = _person_head(item)
            if head is None:
                continue
            inside = box.x <= head[0] <= box.x + box.width and box.y <= head[1] <= box.y + box.height
            if inside and (d := distance(head, target)) < best_d:
                best, best_d = index, d
        if best is not None:
            taken.add(best)
            out[seat_id] = people[best]["keypoints"]
    return out


def _inside(p: tuple[float, float], box: Box) -> bool:
    return box.x <= p[0] <= box.x + box.width and box.y <= p[1] <= box.y + box.height


def _wrists(kps: Keypoints | None) -> list[tuple[float, float]]:
    return [p for index in (LEFT_WRIST, RIGHT_WRIST) if (p := point(kps, index, 0.4))]


def _runs(samples: Sequence[tuple[int, bool]], hold_ms: int, gap_ms: int) -> list[tuple[int, int, int]]:
    """(since, opened at, last true) for every stretch that stays true for ``hold_ms``."""

    runs: list[tuple[int, int, int]] = []
    since = last = None
    for t, ok in samples:
        if ok:
            if since is None or (last is not None and t - last > gap_ms):
                if since is not None and last is not None and last - since >= hold_ms:
                    runs.append((since, since + hold_ms, last))
                since = t
            last = t
    if since is not None and last is not None and last - since >= hold_ms:
        runs.append((since, since + hold_ms, last))
    return runs


class _AllCandidates:
    """Seat roles inside a window: every seat the fast tier kept is a seated candidate."""

    def update(self, *_args: Any) -> str:
        return CANDIDATE

    def left_seat(self, *_args: Any) -> bool:
        return False


def review_window(window: SuspectWindow, frames: Sequence[ReviewFrame],
                  config: ReviewRuleConfig | None = None) -> list[dict[str, Any]]:
    cfg = config or ReviewRuleConfig()
    frames = sorted(frames, key=lambda item: item.offset_ms)
    main, seats = window.seat_id, dict(window.seats) or {window.seat_id: window.bbox}
    records: list[dict[str, Any]] = []
    heads: dict[UUID, list[tuple[int, tuple[float, float]]]] = {seat_id: [] for seat_id in seats}

    def head_at(seat_id: UUID, t: int) -> tuple[float, float] | None:
        track = heads.get(seat_id) or []
        return min(track, key=lambda item: abs(item[0] - t))[1] if track else None

    def emit(kind: EventType, seat_id: UUID, opened: int, closed: int, **details: Any) -> None:
        if closed < window.start_ms or opened > window.end_ms:
            return
        opened = max(opened, window.start_ms)  # began during warm-up, still going when the window starts
        event_id = str(uuid4())
        base = {"id": event_id, "type": kind.value, "seat_id": str(seat_id), "scene": "cctv", "stage": "review",
                "runtime_track_id": window.runtime_track_id if seat_id == main else None,
                "reasons": list(window.reasons), **details}
        head = head_at(seat_id, opened)
        records.append({"action": "opened", "video_offset_ms": opened,
                        **({"head": [round(head[0], 3), round(head[1], 3)]} if head else {}), **base})
        records.append({"action": "closed", "video_offset_ms": max(opened, closed),
                        "duration_ms": max(0, closed - opened), **base})

    def flag(kind: EventType, seat_id: UUID, samples: Sequence[tuple[int, bool]], hold: int, **details: Any) -> None:
        for _since, opened, last in _runs(samples, hold, cfg.gap_ms):
            emit(kind, seat_id, opened, last, **details)

    # Desk line per seat, learned from the frames where that seat writes.
    desk = {seat_id: DeskLine(CctvSuspectConfig()) for seat_id in seats}
    lines: dict[UUID, list[float]] = {seat_id: [] for seat_id in seats}
    down: dict[UUID, list[bool]] = {seat_id: [] for seat_id in seats}
    for frame in frames:
        for seat_id, box in seats.items():
            kps = frame.poses.get(seat_id)
            lift = head_lift(kps, frame.frame_aspect)
            is_down = lift is not None and lift < cfg.min_head_lift
            line = desk[seat_id].value(box, frame.offset_ms)
            if is_down:
                for p in _wrists(kps):
                    if _inside(p, box) and p[1] <= line:
                        desk[seat_id].add(p[1], frame.offset_ms)
            lines[seat_id].append(line)
            down[seat_id].append(is_down)
            if kps is not None:
                heads[seat_id].append((frame.offset_ms, head_point(kps, box)))

    box = seats[main]
    width = box.width

    # (a) phone close to a wrist or below the desk, in enough frames within a second.
    hits = []
    for i, frame in enumerate(frames):
        wrists = _wrists(frame.poses.get(main))
        # The desk top is where the exam paper lies. A high camera sees the desk at the bottom of the seat
        # box, so "below the desk" inside the box is still the desk; a phone in the lap is LOOK_AT_LAP's job.
        papers = [obj.bbox.y for obj in frame.objects if obj.label in ("book", "paper") and _inside(obj.bbox.center, box)]
        desk_top = min([lines[main][i], *papers])
        ok = any(obj.label == "phone" and obj.confidence >= cfg.phone_min_conf and _inside(obj.bbox.center, box)
                 and any(distance(obj.bbox.center, p) < cfg.phone_wrist_widths * width for p in wrists)
                 and obj.bbox.center[1] < desk_top - cfg.phone_raise_heights * box.height for obj in frame.objects)
        hits.append((frame.offset_ms, ok))
    dense = [(t, sum(1 for s, ok in hits if ok and t - cfg.phone_span_ms < s <= t) >= cfg.phone_frames)
             for t, _ in hits]
    flag(EventType.PHONE_DETECTED, main, dense, 0)

    # (b) passing: wrists of two seats meet, or an object moves from one seat to the other.
    for other in seats:
        if other == main:
            continue
        pair_width = (width + seats[other].width) / 2
        meet = [(f.offset_ms, any(distance(p, q) < cfg.passing_wrist_widths * pair_width
                                  for p in _wrists(f.poses.get(main)) for q in _wrists(f.poses.get(other))))
                for f in frames]
        runs = _runs(meet, cfg.passing_hold_ms, cfg.gap_ms)
        if not runs:
            runs = _transfers(frames, main, other, seats, cfg, window.start_ms)
        for _since, opened, last in runs:
            emit(EventType.PASSING_OBJECT, main, opened, last, neighbor_seat_id=str(other))
            emit(EventType.PASSING_OBJECT, other, opened, last, neighbor_seat_id=str(main))

    # (d) paper or book below the desk or held up at the face.
    document = []
    for i, frame in enumerate(frames):
        head = head_point(frame.poses.get(main), box) if frame.poses.get(main) is not None else None
        ok = any(obj.label in ("paper", "book") and _inside(obj.bbox.center, box)
                 and (obj.bbox.center[1] > lines[main][i]
                      or (head is not None and distance(obj.bbox.center, head) < cfg.document_face_widths * width))
                 for obj in frame.objects)
        document.append((frame.offset_ms, ok))
    flag(EventType.DOCUMENT_DETECTED, main, document, cfg.document_hold_ms)

    # (d) head down with the visible hands below the desk for a long time.
    lap = []
    for i, frame in enumerate(frames):
        wrists = _wrists(frame.poses.get(main))
        lap.append((frame.offset_ms, down[main][i] and bool(wrists) and all(p[1] > lines[main][i] for p in wrists)))
    flag(EventType.LOOK_AT_LAP, main, lap, cfg.lap_hold_ms)

    # (c) head turns: the fast tier's own rules on the big model's keypoints.
    records += _head_rules(window, frames, seats, cfg)

    if not any(item["action"] == "opened" for item in records):
        # Two different fast-tier reasons, or a fast-tier rule event the big model did not repeat: keep it
        # for the invigilator rather than lose an event phase 2d already had.
        if len(set(window.reasons)) >= 2 or "RULE_EVENT" in window.reasons:
            emit(EventType.UNCLASSIFIED_SUSPECT, main, window.start_ms, window.end_ms)
        else:
            records.append({"action": "dismissed", "seat_id": str(main), "start_ms": window.start_ms,
                            "end_ms": window.end_ms, "reasons": list(window.reasons), "stage": "review"})
    return records


def _transfers(frames: Sequence[ReviewFrame], a: UUID, b: UUID, seats: Mapping[UUID, Box],
               cfg: ReviewRuleConfig, window_start: int = 0) -> list[tuple[int, int, int]]:
    """An object seen only in seat ``a`` and, within ``passing_transfer_ms``, only in seat ``b`` (or back)."""

    reach = cfg.passing_wrist_widths * (seats[a].width + seats[b].width) / 2

    def where(frame: ReviewFrame, label: str) -> set[UUID]:
        return {key for key in (a, b) for obj in frame.objects if obj.label == label
                and _inside(obj.bbox.center, seats[key])}

    def held(frame: ReviewFrame, label: str, key: UUID) -> bool:
        """Somebody's hand is on the object: papers lying on two desks flicker without anyone moving."""

        wrists = _wrists(frame.poses.get(a)) + _wrists(frame.poses.get(b))
        return any(obj.label == label and _inside(obj.bbox.center, seats[key])
                   and any(distance(obj.bbox.center, p) < reach for p in wrists) for obj in frame.objects)

    out: list[tuple[int, int, int]] = []
    for label in ("paper", "book", "phone"):
        last_only: dict[UUID, int] = {}
        for frame in frames:
            here = where(frame, label)
            if len(here) == 1 and held(frame, label, next(iter(here))):
                (holder,) = here
                giver = b if holder == a else a
                if giver in last_only and frame.offset_ms - last_only[giver] <= cfg.passing_transfer_ms:
                    out.append((last_only[giver], frame.offset_ms, frame.offset_ms))
                    last_only.clear()
                last_only[holder] = frame.offset_ms
    inside = [item for item in out if item[1] >= window_start]
    return inside[:1]


def _head_rules(window: SuspectWindow, frames: Sequence[ReviewFrame], seats: Mapping[UUID, Box],
                cfg: ReviewRuleConfig) -> list[dict[str, Any]]:
    rules = CctvSeatRules(config=replace(CctvRuleConfig(), home_min_ms=cfg.home_min_ms), roles=_AllCandidates())
    keep = {EventType.LOOK_AT_NEIGHBOR.value, EventType.HEAD_TURN_LEFT.value, EventType.HEAD_TURN_RIGHT.value,
            EventType.TURN_AROUND.value, EventType.HEADS_TOGETHER.value}
    opened: dict[str, dict[str, Any]] = {}
    warm: dict[str, dict[str, Any]] = {}  # latched before the window began
    out: list[dict[str, Any]] = []
    seat_objs = [Seat(seat_id, (window.runtime_track_id or 0) if seat_id == window.seat_id else 0, box, box.center, 1.0,
                      0, 0, True) for seat_id, box in seats.items()]
    run: Callable[[Any], Any] = asyncio.run
    for frame in frames:
        update = TrackerUpdate(seat_objs, [], [], [], 0, 0)
        result = MultiFrameResult(frame.offset_ms // 100, frame.offset_ms, update, {}, {}, {}, [], _TIMING,
                                  {key: list(map(list, kps)) for key, kps in frame.poses.items()})
        size = (round(1000 * frame.frame_aspect), 1000)
        if frame.offset_ms >= window.start_ms and warm:
            for key, item in warm.items():  # still open when the window starts
                item.update(stage="review", reasons=list(window.reasons), video_offset_ms=frame.offset_ms)
                opened[key] = item
                out.append(item)
            warm.clear()
        for _seat, item in run(rules.step(result, {"frame_size": size}, frame.offset_ms)):
            if item["type"] not in keep or item["seat_id"] != str(window.seat_id):
                continue
            if item["action"] == "opened" and item["video_offset_ms"] < window.start_ms:
                warm[str(item["id"])] = item
            elif item["action"] == "closed" and str(item["id"]) in warm:
                warm.pop(str(item["id"]))
            elif item["action"] == "opened" and window.start_ms <= item["video_offset_ms"] <= window.end_ms:
                item.update(stage="review", reasons=list(window.reasons))
                opened[str(item["id"])] = item
                out.append(item)
            elif item["action"] == "closed" and str(item["id"]) in opened:
                item.update(stage="review", reasons=list(window.reasons))
                out.append(item)
    return out


__all__ = ["ReviewFrame", "ReviewRuleConfig", "assign_poses", "review_window"]
