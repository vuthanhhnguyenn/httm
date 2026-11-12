"""Link the people the pose model found across frames, so a seat's owner does not change from frame to frame.

Seated people barely move, so a greedy match of each person to the track seen last (box overlap, or shoulders
close compared with their width) is enough; a track survives ``max_gap_ms`` without a detection.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import hypot
from typing import Any
from uuid import UUID

from proctoring_core.observations import Box

from .head import Keypoints, shoulders
from .review_rules import ReviewFrame, assign_poses

MIN_IOU = 0.3
SHOULDER_REACH = 0.6  # shoulder midpoints closer than this many shoulder widths still match


@dataclass(slots=True)
class _Track:
    ident: int
    box: Box | None
    mid: tuple[float, float] | None
    width: float
    last_ms: int


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _shoulder_mid(person: Mapping[str, Any]) -> tuple[tuple[float, float], float] | None:
    pair = shoulders(person.get("keypoints"))
    if pair is None:
        return None
    (lx, ly), (rx, ry) = pair
    return ((lx + rx) / 2, (ly + ry) / 2), abs(lx - rx)


def _affinity(track: _Track, box: Box | None, shoulder: tuple[tuple[float, float], float] | None) -> float:
    best = 0.0
    if track.box is not None and box is not None and (value := _iou(track.box, box)) >= MIN_IOU:
        best = value
    if track.mid is not None and shoulder is not None:
        mid, width = shoulder
        gap = hypot(mid[0] - track.mid[0], mid[1] - track.mid[1])
        reach = SHOULDER_REACH * max(track.width, width)
        if reach > 0 and gap < reach:
            best = max(best, 1.0 - gap / reach)
    return best


def link_people(frames_people: Sequence[Sequence[Mapping[str, Any]]], offsets_ms: Sequence[int], *,
                max_gap_ms: int = 1000) -> list[list[int]]:
    """A track id for every person of every frame (same shape as ``frames_people``), in time order."""

    tracks: list[_Track] = []
    out: list[list[int]] = []
    for people, ms in zip(frames_people, offsets_ms, strict=True):
        live = [track for track in tracks if ms - track.last_ms <= max_gap_ms]
        details = [(Box.parse(person.get("bbox")), _shoulder_mid(person)) for person in people]
        pairs = sorted(((score, t_index, p_index) for t_index, track in enumerate(live)
                        for p_index, (box, shoulder) in enumerate(details)
                        if (score := _affinity(track, box, shoulder)) > 0), key=lambda item: -item[0])
        ids: list[int | None] = [None] * len(people)
        taken: set[int] = set()
        for _, t_index, p_index in pairs:
            if ids[p_index] is None and t_index not in taken:
                ids[p_index] = live[t_index].ident
                taken.add(t_index)
        for p_index, (box, shoulder) in enumerate(details):
            if ids[p_index] is None:
                track = _Track(len(tracks), None, None, 0.0, ms)
                tracks.append(track)
                live.append(track)
                ids[p_index] = track.ident
            track = tracks[ids[p_index]]  # type: ignore[index]
            track.box, track.last_ms = box or track.box, ms
            if shoulder is not None:
                track.mid, track.width = shoulder
        out.append([int(ident) for ident in ids if ident is not None])
    return out


def distinct_people(people: Sequence[Mapping[str, Any]], *, max_iou: float = 0.6) -> list[Mapping[str, Any]]:
    """The same person found twice (a duplicate detection) once: of two boxes overlapping more than ``max_iou`` the
    first stays. Duplicates would otherwise become two tracks and split a seat's votes."""

    kept: list[Mapping[str, Any]] = []
    boxes: list[Box | None] = []
    for person in people:
        box = Box.parse(person.get("bbox"))
        if box is not None and any(other is not None and _iou(box, other) > max_iou for other in boxes):
            continue
        kept.append(person)
        boxes.append(box)
    return kept


def assign_tracked(frames: Sequence[ReviewFrame], *, window_ms: int = 20000) -> list[ReviewFrame]:
    """Seat owners from tracks. Each frame's nearest-head choice (the old rule) is a vote for a track; a seat's owner
    at time t is the track with most votes within ``window_ms`` around t. The seat's pose in a frame is the owner's
    when the owner was detected there and none otherwise, so a head that only drifts into the box for a moment, or
    a neighbour taking an empty seat, no longer replaces the real occupant."""

    if not any(frame.people for frame in frames):
        return list(frames)
    ordered = [replace(frame, people=distinct_people(frame.people)) for frame in sorted(frames, key=lambda f: f.offset_ms)]
    ids = link_people([frame.people for frame in ordered], [frame.offset_ms for frame in ordered])
    votes: list[dict[UUID, int]] = []
    for frame, track_ids in zip(ordered, ids, strict=True):
        chosen = assign_poses(frame.seats, frame.people, by_body=False)
        by_object = {id(person["keypoints"]): ident for person, ident in zip(frame.people, track_ids, strict=True)}
        votes.append({seat: by_object[id(kps)] for seat, kps in chosen.items() if id(kps) in by_object})
    out: list[ReviewFrame] = []
    for frame, track_ids in zip(ordered, ids, strict=True):
        tally: dict[UUID, Counter[int]] = {}
        for other, mine in zip(ordered, votes, strict=True):
            if abs(other.offset_ms - frame.offset_ms) <= window_ms / 2:
                for seat, ident in mine.items():
                    tally.setdefault(seat, Counter())[ident] += 1
        present = {ident: person for person, ident in zip(frame.people, track_ids, strict=True)}
        poses: dict[UUID, Keypoints] = {}
        for seat in frame.seats:
            owner = tally[seat].most_common(1)[0][0] if seat in tally else None
            if owner is not None and owner in present:
                poses[seat] = present[owner]["keypoints"]
        out.append(replace(frame, poses=poses))
    return out
