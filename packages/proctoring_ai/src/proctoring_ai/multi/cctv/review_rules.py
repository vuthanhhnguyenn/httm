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
from .pose_detector import LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST
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
    # Evidence only (never gates an event): a phone this far (seat heights) above the desk edge is held up,
    # not just lifted off the desk. Confidence levels are derived from it downstream.
    phone_raised_heights: float = 0.2
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
    # A "phone" lying on the seat's own paper in at least this share of the frames it is seen in is a
    # calculator, not a phone. Above 1 switches the rule off.
    phone_paper_fraction: float = 0.6
    # Hands meeting is judged against the people's size, not the frame: far rows are tiny and their
    # keypoints jitter by more than a hand's width.
    passing_wrist_heights: float = 0.2  # wrists closer than this x the smaller seat box height (25.5 s: 0.13-0.2)
    passing_wrist_conf: float = 0.5
    far_row_height: float = 0.12  # seat boxes lower than this (frame heights) are far rows
    far_row_inside_frames: int = 2  # far rows also need a wrist inside the other seat this often
    # Two people's wrists on the very same spot is one arm the pose model gave to both, not two hands.
    same_arm_heights: float = 0.03
    # A handover takes about a second; hands side by side for longer, with nothing between them, are two
    # neighbours resting on a shared desk edge.
    passing_max_ms: int = 3000
    # LOOK_AT_LAP: no hand of the candidate on the desk. A writer always has one hand on the paper, but a
    # writer's hand is often hidden by the arm or the desk too, so a hidden wrist counts as under the desk
    # only when the same-side elbow is seen below the desk line.
    lap_hidden_needs_low_elbow: bool = True
    # A run is kept only if every visible wrist moved at most this many seat widths (a hand sliding along the
    # desk edge is writing, not resting in the lap).
    # None: off (pose jitter of a real lap is as large as a writer's slide).
    lap_still_widths: float | None = None
    # A wrist inside paper or a book lying on the desk (box centre above the desk line) is writing, even when
    # the wrist sits below the line; paper in the lap does not block.
    lap_paper_blocks: bool = True
    # A forearm seen whole (wrist and elbow) must hang into the lap: the wrist at least this many shoulder
    # widths below its elbow. A writer's forearm lies on the desk, about level (clip 1: 0.05-0.23 against
    # 0.36-0.46 for real laps). The desk line alone cannot tell: the seat box ends at the desk edge, so a
    # writer's hands sit below its default line and never teach it. None: off.
    lap_forearm_drop: float | None = 0.25


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


def _hands_under_desk(kps: Keypoints, line: float, frame: ReviewFrame, box: Box, cfg: ReviewRuleConfig) -> bool:
    """No wrist above the desk line or on the seat's paper, no forearm level on the desk; a wrist the camera lost
    needs its elbow below the line."""

    for wrist_index, elbow_index in ((LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)):
        wrist = point(kps, wrist_index, 0.4)
        if wrist is not None:
            if wrist[1] <= line or (cfg.lap_paper_blocks and _point_on_paper(wrist, frame, box, line)):
                return False
            if cfg.lap_forearm_drop is not None and (pair := shoulders(kps)) is not None \
                    and (elbow := point(kps, elbow_index, 0.4)) is not None:
                width = abs(pair[0][0] - pair[1][0]) * frame.frame_aspect
                if wrist[1] - elbow[1] < cfg.lap_forearm_drop * width:
                    return False
        elif cfg.lap_hidden_needs_low_elbow:
            elbow = point(kps, elbow_index, 0.4)
            if elbow is None or elbow[1] <= line:
                return False
    return True


def _lap_still(frames: Sequence[ReviewFrame], seat: UUID, since: int, last: int, limit: float) -> bool:
    """Every wrist seen in [since, last] stayed within ``limit`` of its own other positions."""

    for index in (LEFT_WRIST, RIGHT_WRIST):
        seen = [p for frame in frames if since <= frame.offset_ms <= last
                and (p := point(frame.poses.get(seat), index, 0.4)) is not None]
        if any(distance(a, b) > limit for n, a in enumerate(seen) for b in seen[n + 1:]):
            return False
    return True


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
    alive = _alive(frames, main, box, cfg) - _calculators(frames, box, cfg)

    # (a) phone close to a wrist or below the desk, in enough frames within a second.
    hits = []
    phone_facts: list[tuple[int, bool, float, float]] = []  # per qualifying frame: (offset, raised, conf, aspect)
    for i, frame in enumerate(frames):
        wrists = _wrists(frame.poses.get(main))
        # The desk top is where the exam paper lies. A high camera sees the desk at the bottom of the seat
        # box, so "below the desk" inside the box is still the desk; a phone in the lap is LOOK_AT_LAP's job.
        papers = [obj.bbox.y for obj in frame.objects if obj.label in ("book", "paper") and _inside(obj.bbox.center, box)]
        desk_top = min([lines[main][i], *papers])
        # Above the candidate's own head is the desk of the row in front, seen past their head.
        own_head = head_point(frame.poses[main], box) if frame.poses.get(main) is not None else None
        found = [obj for obj in frame.objects
                 if obj.label == "phone" and id(obj) in alive and obj.confidence >= cfg.phone_min_conf
                 and _inside(obj.bbox.center, box)
                 and (own_head is None or obj.bbox.center[1] > own_head[1])
                 and any(distance(obj.bbox.center, p) < cfg.phone_wrist_widths * width for p in wrists)
                 and obj.bbox.center[1] < desk_top - cfg.phone_raise_heights * box.height]
        hits.append((frame.offset_ms, bool(found)))
        if found:
            raised = any(obj.bbox.center[1] < desk_top - cfg.phone_raised_heights * box.height for obj in found)
            phone_facts.append((frame.offset_ms, raised, max(obj.confidence for obj in found),
                                max(obj.bbox.height / max(obj.bbox.width, 1e-6) for obj in found)))

    def needed(t: int) -> int:  # with sparse frames a second may hold fewer than phone_frames frames
        return max(1, min(cfg.phone_frames, sum(1 for s, _ in hits if t - cfg.phone_span_ms < s <= t)))

    dense = [(t, sum(1 for s, ok in hits if ok and t - cfg.phone_span_ms < s <= t) >= needed(t)) for t, _ in hits]
    for since, opened, last in _runs(dense, 0, cfg.gap_ms):
        facts = [item for item in phone_facts if since - cfg.phone_span_ms < item[0] <= last]
        emit(EventType.PHONE_DETECTED, main, opened, last, raised=any(item[1] for item in facts),
             confidence=max((item[2] for item in facts), default=0.0),
             aspect=round(max((item[3] for item in facts), default=0.0), 2))

    # (b) passing: wrists of two seats meet, or an object moves from one seat to the other.
    for other in seats:
        if other == main:
            continue
        meet = _hands_meet(frames, main, other, seats, cfg)
        meets = [(run, _object_between(frames, main, other, run, cfg))
                 for run in _runs(meet, cfg.passing_hold_ms, cfg.gap_ms)]
        # (run, evidence): an object seen between the hands, hands alone, or an object changing seats.
        runs = [(run, "object_between" if between else "hands_only") for run, between in meets
                if run[2] - run[0] <= cfg.passing_max_ms or between]
        if not runs:
            runs = [(run, "transfer") for run in _transfers(frames, main, other, seats, cfg, window.start_ms)]
        for (_since, opened, last), evidence in runs:
            # Only the candidate who reaches over is flagged (Nguyên, phase 2d), not the neighbour.
            reacher = _reacher(frames, main, other, seats, opened - cfg.passing_hold_ms, last)
            neighbour = other if reacher == main else main
            emit(EventType.PASSING_OBJECT, reacher, opened, last, neighbor_seat_id=str(neighbour),
                 evidence_kind=evidence)

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

    # (d) head down with the hands under the desk for a long time.
    lap = []
    for i, frame in enumerate(frames):
        kps = frame.poses.get(main)
        lap.append((frame.offset_ms, down[main][i] and kps is not None and _hands_under_desk(
            kps, lines[main][i], frame, box, cfg)))
    for _since, opened, last in _runs(lap, cfg.lap_hold_ms, cfg.gap_ms):
        if cfg.lap_still_widths is None or _lap_still(frames, main, _since, last, width * cfg.lap_still_widths):
            emit(EventType.LOOK_AT_LAP, main, opened, last)

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


def _calculators(frames: Sequence[ReviewFrame], box: Box, cfg: ReviewRuleConfig) -> set[int]:
    """``id()`` of the "phone" objects that lie on the seat's paper in ``phone_paper_fraction`` of the frames
    they are seen in (linked frame to frame by IoU, as in ``_alive``): a calculator, not a phone."""

    tracks: list[list[tuple[ReviewObject, bool]]] = []
    for frame in frames:
        for obj in frame.objects:
            if obj.label != "phone" or not _inside(obj.bbox.center, box):
                continue
            item = (obj, _on_paper(obj, frame, box))
            match = next((t for t in tracks if _iou(t[-1][0].bbox, obj.bbox) > cfg.object_track_iou), None)
            if match is None:
                tracks.append([item])
            else:
                match.append(item)
    return {id(obj) for track in tracks if sum(on for _, on in track) / len(track) >= cfg.phone_paper_fraction
            for obj, _ in track}


def _on_paper(obj: ReviewObject, frame: ReviewFrame, box: Box) -> bool:
    return _point_on_paper(obj.bbox.center, frame, box)


def _point_on_paper(p: tuple[float, float], frame: ReviewFrame, box: Box, line: float | None = None) -> bool:
    for paper in frame.objects:
        if paper.label not in ("book", "paper") or not _inside(paper.bbox.center, box):
            continue
        if line is not None and paper.bbox.center[1] > line:
            continue
        b = paper.bbox
        grown = Box(b.x - 0.1 * b.width, b.y - 0.1 * b.height, 1.2 * b.width, 1.2 * b.height)
        if _inside(p, grown):
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
    seat_objs = [Seat(seat_id, (window.runtime_track_id or 0) if seat_id == window.seat_id else 0, box, box.center, 1.0,
                      0, 0, True) for seat_id, box in seats.items()]
    # One event loop for the whole window: a loop per frame meant thousands of loops per video, and on Windows
    # every new loop opens a loopback socketpair (CI hung waiting on one).
    with asyncio.Runner() as runner:
        return _head_rules_loop(window, frames, rules, keep, seat_objs, runner.run)


def _head_rules_loop(window: SuspectWindow, frames: Sequence[ReviewFrame], rules: CctvSeatRules, keep: set[str],
                     seat_objs: list[Seat], run: Callable[[Any], Any]) -> list[dict[str, Any]]:
    opened: dict[str, dict[str, Any]] = {}
    warm: dict[str, dict[str, Any]] = {}  # latched before the window began
    out: list[dict[str, Any]] = []
    for frame in frames:
        update = TrackerUpdate(seat_objs, [], [], [], 0, 0)
        result = MultiFrameResult(frame.offset_ms // 100, frame.offset_ms, update, {}, {}, {}, [], _TIMING,
                                  {key: list(map(list, kps)) for key, kps in frame.poses.items() if kps is not None})
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
