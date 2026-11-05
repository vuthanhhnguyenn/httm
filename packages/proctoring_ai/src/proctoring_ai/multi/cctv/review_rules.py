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
from .review_detectors import ReviewObject, _iou
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
    # Below the desk only a document in the hand counts: a wrist this close (seat heights) in the same frame.
    # The exam paper under a writing hand sits a third of a seat height below the wrist (clip 1, 10.0/32.0 s).
    document_wrist_heights: float = 0.15
    lap_hold_ms: int = 3000
    min_head_lift: float = 0.1
    gap_ms: int = 300
    home_min_ms: int = 5000  # the warm-up before a window is shorter than the fast tier's 10 s
    # Phase 2f: a phone or paper only counts while somebody handles it. Things lying still on the desk all
    # window long (calculator, exam paper) are desk items, not cheating.
    object_min_frames: int = 2  # one detection alone is noise
    object_track_iou: float = 0.3  # same object in consecutive sightings
    object_wrist_widths: float = 0.6  # "in the hand": centre this close to a wrist, in seat widths
    object_move_widths: float = 0.5  # or it travelled this far, in its own widths
    calculator_on_paper: bool = False  # a "phone" on the seat's own paper is a calculator (set if 31.6 stays)
    # Hands meeting is judged against the people's size, not the frame: far rows are tiny and their
    # keypoints jitter by more than a hand's width.
    passing_wrist_heights: float = 0.15  # wrists closer than this x the smaller seat box height
    passing_wrist_conf: float = 0.5
    far_row_height: float = 0.12  # seat boxes lower than this (frame heights) are far rows
    far_row_inside_frames: int = 2  # far rows also need a wrist inside the other seat this often
    # Two people's wrists on the very same spot is one arm the pose model gave to both, not two hands.
    same_arm_heights: float = 0.03
    # A handover takes about a second; hands side by side for longer, with nothing between them, are two
    # neighbours resting on a shared desk edge.
    passing_max_ms: int = 3000
    # LOOK_AT_LAP: no hand of the candidate on the desk. A writer always has one hand on the paper; a hand
    # the camera cannot see is under the desk, so hidden hands count as low (phase 1 lap rule needed one seen).
    lap_needs_both_wrists: bool = True


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


def _wrists(kps: Keypoints | None, min_conf: float = 0.4) -> list[tuple[float, float]]:
    return [p for index in (LEFT_WRIST, RIGHT_WRIST) if (p := point(kps, index, min_conf))]


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
    # The live deep tier sees ~3 frames a second: a gap between two frames is not a break in what they show.
    spacing = sorted(b.offset_ms - a.offset_ms for a, b in zip(frames, frames[1:], strict=False))
    if spacing and (gap := int(1.5 * spacing[len(spacing) // 2])) > cfg.gap_ms:
        cfg = replace(cfg, gap_ms=gap)
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
    alive = _alive(frames, main, box, cfg)

    # (a) phone close to a wrist or below the desk, in enough frames within a second.
    hits = []
    for i, frame in enumerate(frames):
        wrists = _wrists(frame.poses.get(main))
        # The desk top is where the exam paper lies. A high camera sees the desk at the bottom of the seat
        # box, so "below the desk" inside the box is still the desk; a phone in the lap is LOOK_AT_LAP's job.
        papers = [obj.bbox.y for obj in frame.objects if obj.label in ("book", "paper") and _inside(obj.bbox.center, box)]
        desk_top = min([lines[main][i], *papers])
        # Above the candidate's own head is the desk of the row in front, seen past their head.
        own_head = head_point(frame.poses[main], box) if frame.poses.get(main) is not None else None
        ok = any(obj.label == "phone" and id(obj) in alive and obj.confidence >= cfg.phone_min_conf
                 and _inside(obj.bbox.center, box)
                 and not (cfg.calculator_on_paper and _on_paper(obj, frame, box))
                 and (own_head is None or obj.bbox.center[1] > own_head[1])
                 and any(distance(obj.bbox.center, p) < cfg.phone_wrist_widths * width for p in wrists)
                 and obj.bbox.center[1] < desk_top - cfg.phone_raise_heights * box.height for obj in frame.objects)
        hits.append((frame.offset_ms, ok))
    def needed(t: int) -> int:  # with sparse frames a second may hold fewer than phone_frames frames
        return max(1, min(cfg.phone_frames, sum(1 for s, _ in hits if t - cfg.phone_span_ms < s <= t)))

    dense = [(t, sum(1 for s, ok in hits if ok and t - cfg.phone_span_ms < s <= t) >= needed(t)) for t, _ in hits]
    flag(EventType.PHONE_DETECTED, main, dense, 0)

    # (b) passing: wrists of two seats meet, or an object moves from one seat to the other.
    for other in seats:
        if other == main:
            continue
        meet = _hands_meet(frames, main, other, seats, cfg)
        runs = [run for run in _runs(meet, cfg.passing_hold_ms, cfg.gap_ms)
                if run[2] - run[0] <= cfg.passing_max_ms or _object_between(frames, main, other, run, cfg)]
        if not runs:
            runs = _transfers(frames, main, other, seats, cfg, window.start_ms)
        for _since, opened, last in runs:
            # Only the candidate who reaches over is flagged (Nguyên, phase 2d), not the neighbour.
            reacher = _reacher(frames, main, other, seats, opened - cfg.passing_hold_ms, last)
            neighbour = other if reacher == main else main
            emit(EventType.PASSING_OBJECT, reacher, opened, last, neighbor_seat_id=str(neighbour))

    # (d) paper or book below the desk or held up at the face.
    moving = _alive(frames, main, box, cfg, by_hand=False)
    document = []
    for i, frame in enumerate(frames):
        head = head_point(frame.poses.get(main), box) if frame.poses.get(main) is not None else None
        hands = _wrists(frame.poses.get(main))
        held = cfg.document_wrist_heights * box.height
        ok = any(obj.label in ("paper", "book") and id(obj) in alive and _inside(obj.bbox.center, box)
                 and ((obj.bbox.center[1] > lines[main][i]
                       and (id(obj) in moving or any(distance(obj.bbox.center, p) < held for p in hands)))
                      or (head is not None and distance(obj.bbox.center, head) < cfg.document_face_widths * width))
                 for obj in frame.objects)
        document.append((frame.offset_ms, ok))
    flag(EventType.DOCUMENT_DETECTED, main, document, cfg.document_hold_ms)

    # (d) head down with the visible hands below the desk for a long time.
    lap = []
    for i, frame in enumerate(frames):
        wrists = _wrists(frame.poses.get(main))
        low = all(p[1] > lines[main][i] for p in wrists)
        seen = bool(wrists) or cfg.lap_needs_both_wrists
        lap.append((frame.offset_ms, down[main][i] and seen and low))
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


def _hands_meet(frames: Sequence[ReviewFrame], a: UUID, b: UUID, seats: Mapping[UUID, Box],
                cfg: ReviewRuleConfig) -> list[tuple[int, bool]]:
    height = min(seats[a].height, seats[b].height)
    reach = cfg.passing_wrist_heights * height
    far = height < cfg.far_row_height
    inside = sum(1 for f in frames
                 if any(_inside(p, seats[b]) for p in _wrists(f.poses.get(a), cfg.passing_wrist_conf))
                 or any(_inside(q, seats[a]) for q in _wrists(f.poses.get(b), cfg.passing_wrist_conf)))
    out: list[tuple[int, bool]] = []
    same_arm = cfg.same_arm_heights * height
    for f in frames:
        pairs = [(p, q) for p in _wrists(f.poses.get(a), cfg.passing_wrist_conf)
                 for q in _wrists(f.poses.get(b), cfg.passing_wrist_conf) if same_arm <= distance(p, q) < reach]
        ok = bool(pairs)
        if ok and far:
            between = any(distance(obj.bbox.center, ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)) < reach
                          for p, q in pairs for obj in f.objects)
            ok = between or inside >= cfg.far_row_inside_frames
        out.append((f.offset_ms, ok))
    return out


def _object_between(frames: Sequence[ReviewFrame], a: UUID, b: UUID, run: tuple[int, int, int],
                    cfg: ReviewRuleConfig) -> bool:
    for f in frames:
        if not run[0] <= f.offset_ms <= run[2]:
            continue
        for p in _wrists(f.poses.get(a), cfg.passing_wrist_conf):
            for q in _wrists(f.poses.get(b), cfg.passing_wrist_conf):
                middle = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                if any(distance(obj.bbox.center, middle) < distance(p, q) for obj in f.objects):
                    return True
    return False


def _alive(frames: Sequence[ReviewFrame], main: UUID, box: Box, cfg: ReviewRuleConfig, *,
           by_hand: bool = True) -> set[int]:
    """``id()`` of the objects in the seat that somebody handles: the same object (linked frame to frame by
    IoU) seen in ``object_min_frames`` frames, and near the candidate's wrist once or moved meanwhile.
    ``by_hand=False`` keeps only the ones that moved."""

    tracks: list[list[tuple[ReviewObject, list[tuple[float, float]]]]] = []
    for frame in frames:
        wrists = _wrists(frame.poses.get(main))
        for obj in frame.objects:
            if not _inside(obj.bbox.center, box):
                continue
            match = next((track for track in tracks if track[-1][0].label == obj.label
                          and _iou(track[-1][0].bbox, obj.bbox) > cfg.object_track_iou), None)
            if match is None:
                tracks.append([(obj, wrists)])
            else:
                match.append((obj, wrists))
    out: set[int] = set()
    for track in tracks:
        if len(track) < cfg.object_min_frames:
            continue
        held = any(distance(obj.bbox.center, p) <= cfg.object_wrist_widths * box.width
                   for obj, wrists in track for p in wrists)
        first, last = track[0][0].bbox, track[-1][0].bbox
        moved = distance(first.center, last.center) > cfg.object_move_widths * max(first.width, 1e-6)
        if (held and by_hand) or moved:
            out |= {id(obj) for obj, _ in track}
    return out


def _on_paper(obj: ReviewObject, frame: ReviewFrame, box: Box) -> bool:
    for paper in frame.objects:
        if paper.label not in ("book", "paper") or not _inside(paper.bbox.center, box):
            continue
        b = paper.bbox
        grown = Box(b.x - 0.1 * b.width, b.y - 0.1 * b.height, 1.2 * b.width, 1.2 * b.height)
        if _inside(obj.bbox.center, grown):
            return True
    return False


def _reacher(frames: Sequence[ReviewFrame], a: UUID, b: UUID, seats: Mapping[UUID, Box], start: int,
             end: int) -> UUID:
    """The seat whose wrist got farthest past the midline between the two seats during [start, end];
    ``a`` (the window's seat) when nobody crossed it."""

    ax, bx = seats[a].center[0], seats[b].center[0]
    mid = (ax + bx) / 2
    toward = 1.0 if bx >= ax else -1.0  # direction from a to b along x

    def reach(seat_id: UUID, sign: float) -> float:
        return max([sign * (p[0] - mid) for f in frames if start <= f.offset_ms <= end
                    for p in _wrists(f.poses.get(seat_id))], default=0.0)

    reach_a, reach_b = reach(a, toward), reach(b, -toward)
    return b if reach_b > max(0.0, reach_a) else a


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
