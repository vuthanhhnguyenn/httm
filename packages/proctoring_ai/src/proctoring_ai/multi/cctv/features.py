"""Phase 3 tier 2: what the learned core sees of one seat in one 3 s window.

The seat's frames come from its whole-video window (``timeline.py``). Each window of ``WIN_MS`` (every
``STEP_MS``) becomes one ``FeatureRow`` of ``FEATURE_NAMES``, all finite numbers. Distances are in seat
sizes, head angles against that seat's own median over the whole video (offline runs can afford it). The old
deep-tier rules run once per seat and only become ``rule_*`` flags: they no longer decide anything.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median, pstdev
from typing import Any
from uuid import UUID

from proctoring_core.observations import Box

from .head import distance, head_lift, head_point, head_yaw_index, point
from .pose_detector import LEFT_WRIST, RIGHT_WRIST
from .review_cache import CacheKey, window_key
from .review_rules import ReviewFrame, review_window
from .suspects import CctvSuspectConfig, DeskLine, SuspectWindow

WIN_MS = 3000
STEP_MS = 1000
RULE_TYPES = ("LOOK_AT_NEIGHBOR", "HEADS_TOGETHER", "PASSING_OBJECT", "PHONE_DETECTED", "DOCUMENT_DETECTED",
              "LOOK_AT_LAP", "TURN_AROUND")
_RULE_FLAGS = (*RULE_TYPES, "HEAD_TURN")  # HEAD_TURN_LEFT / _RIGHT of the old rules, one flag
TURN_YAW = 0.35
HEAD_DOWN_LIFT = 0.1
MIN_BASELINE_MS = 10_000
OBJECT_WRIST_WIDTHS = 0.6
RAISED_HEIGHTS = 0.2
MAX_SPEED_GAP_MS = 1000
FAR = 5.0  # pair distances when there is no neighbour to measure against

FEATURE_NAMES: tuple[str, ...] = (
    "yaw_dev_mean", "yaw_dev_max", "yaw_dev_std", "yaw_turn_share", "lift_dev_mean", "lift_min", "head_down_share",
    "wrist_below_desk_share", "wrist_speed_max", "reach_out_max",
    "wrist_gap_min", "head_gap_ratio_min", "lean_toward",
    "phone_conf_max", "phone_near_wrist_share", "phone_raised", "paper_conf_max", "paper_near_wrist_share",
    "object_moved",
    *(f"rule_{name}" for name in _RULE_FLAGS),
    "kp_conf_mean", "seat_height", "frames_seen_share",
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """``partner``: the neighbour the pair features were measured against (None without one); ``reacher``:
    which of the two reached over to the other (the seat itself or ``partner``)."""

    video: str
    seat_id: UUID
    start_ms: int
    end_ms: int
    seat_box: Box
    values: Mapping[str, float]
    partner: str | None = None
    reacher: str | None = None


def _wrists(kps: Any) -> list[tuple[int, tuple[float, float]]]:
    return [(index, p) for index in (LEFT_WRIST, RIGHT_WRIST) if (p := point(kps, index, 0.4)) is not None]


def _outside(p: tuple[float, float], box: Box) -> float:
    dx = max(0.0, box.x - p[0], p[0] - (box.x + box.width))
    dy = max(0.0, box.y - p[1], p[1] - (box.y + box.height))
    return math.hypot(dx, dy) / max(box.width, 1e-6)


def _toward(p: tuple[float, float], own: Box, other: Box) -> float:
    """How far (own widths) ``p`` lies past ``own``'s edge on ``other``'s side."""

    if other.center[0] >= own.center[0]:
        return max(0.0, p[0] - (own.x + own.width)) / max(own.width, 1e-6)
    return max(0.0, own.x - p[0]) / max(own.width, 1e-6)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else default


@dataclass(slots=True)
class _Seat:
    seat: UUID
    box: Box
    frames: list[ReviewFrame]
    desk: list[float]  # desk line per frame
    yaw_base: float = 0.0
    lift_base: float = 0.0
    head_x_base: float = 0.0


def _prepare(window: SuspectWindow, frames: Sequence[ReviewFrame]) -> tuple[_Seat, list[float], list[float], int]:
    seat, box = window.seat_id, window.bbox
    frames = sorted(frames, key=lambda f: f.offset_ms)
    desk_line = DeskLine(CctvSuspectConfig())
    desk, yaws, lifts, seen = [], [], [], []
    for frame in frames:
        kps = frame.poses.get(seat)
        line = desk_line.value(box, frame.offset_ms)
        lift = head_lift(kps, frame.frame_aspect)
        if lift is not None and lift < HEAD_DOWN_LIFT:
            for _i, p in _wrists(kps):
                if box.x <= p[0] <= box.x + box.width and box.y <= p[1] <= line:
                    desk_line.add(p[1], frame.offset_ms)
        desk.append(line)
        if kps is not None:
            seen.append(frame.offset_ms)
            if (yaw := head_yaw_index(kps)) is not None:
                yaws.append(yaw)
            if lift is not None:
                lifts.append(lift)
    span = seen[-1] - seen[0] if seen else 0
    heads = [head_point(f.poses[seat], box)[0] for f in frames if f.poses.get(seat) is not None]
    state = _Seat(seat, box, frames, desk, head_x_base=median(heads) if heads else box.center[0])
    return state, yaws, lifts, span


def _baselines(states: dict[UUID, _Seat], raw: dict[UUID, tuple[list[float], list[float], int]]) -> None:
    own = {seat: (median(yaws), median(lifts)) for seat, (yaws, lifts, span) in raw.items()
           if span >= MIN_BASELINE_MS and yaws and lifts}
    for seat, state in states.items():
        if seat in own:
            state.yaw_base, state.lift_base = own[seat]
            continue
        row = [own[other] for other in own if other != seat
               and abs(states[other].box.center[1] - state.box.center[1]) < 0.5 * state.box.height]
        state.yaw_base = median(item[0] for item in row) if row else 0.0
        state.lift_base = median(item[1] for item in row) if row else 0.0


def _rule_spans(window: SuspectWindow, frames: Sequence[ReviewFrame]) -> dict[str, list[tuple[int, int]]]:
    records = review_window(window, list(frames))
    opened = {item["id"]: item for item in records if item.get("action") == "opened"
              and item.get("seat_id") == str(window.seat_id)}
    spans: dict[str, list[tuple[int, int]]] = {}
    for item in records:
        if item.get("action") == "closed" and item.get("id") in opened:
            kind = str(item["type"])
            kind = "HEAD_TURN" if kind.startswith("HEAD_TURN") else kind
            spans.setdefault(kind, []).append((int(opened[item["id"]]["video_offset_ms"]), int(item["video_offset_ms"])))
    return spans


def _row_values(state: _Seat, others: Mapping[UUID, _Seat], neighbours: Sequence[UUID], start: int, end: int,
                spans: Mapping[str, list[tuple[int, int]]], pair_median: Callable[[UUID], float]
                ) -> tuple[dict[str, float], str | None, str | None] | None:
    idx = [i for i, f in enumerate(state.frames) if start <= f.offset_ms < end]
    if not idx:
        return None
    seat, box = state.seat, state.box
    seen = [i for i in idx if state.frames[i].poses.get(seat) is not None]
    if not seen:
        return None
    v: dict[str, float] = {}
    devs, lifts = [], []
    down = 0
    for i in seen:
        frame = state.frames[i]
        kps = frame.poses[seat]
        if (yaw := head_yaw_index(kps)) is not None:
            devs.append(yaw - state.yaw_base)
        lift = head_lift(kps, frame.frame_aspect)
        if lift is not None:
            lifts.append(lift)
            down += lift < HEAD_DOWN_LIFT
    absd = [abs(d) for d in devs]
    v["yaw_dev_mean"] = _mean(absd)
    v["yaw_dev_max"] = max(absd, default=0.0)
    v["yaw_dev_std"] = pstdev(devs) if len(devs) > 1 else 0.0
    v["yaw_turn_share"] = _mean([float(d > TURN_YAW) for d in absd])
    v["lift_dev_mean"] = _mean([lift - state.lift_base for lift in lifts])
    v["lift_min"] = min(lifts, default=0.0)
    v["head_down_share"] = down / len(seen)

    below, speed, reach = 0, 0.0, 0.0
    last: dict[int, tuple[int, tuple[float, float]]] = {}
    for i in seen:
        frame = state.frames[i]
        wrists = _wrists(frame.poses[seat])
        below += all(p[1] > state.desk[i] for _j, p in wrists)
        for j, p in wrists:
            reach = max(reach, _outside(p, box))
            if j in last and 0 < frame.offset_ms - last[j][0] <= MAX_SPEED_GAP_MS:
                dt = (frame.offset_ms - last[j][0]) / 1000
                speed = max(speed, distance(p, last[j][1]) / max(box.width, 1e-6) / dt)
            last[j] = (frame.offset_ms, p)
    v["wrist_below_desk_share"] = below / len(seen)
    v["wrist_speed_max"] = speed
    v["reach_out_max"] = reach

    # Each pair feature is its strongest value over all neighbours; the partner is the neighbour with the
    # closest wrists, else the closest head, else the strongest lean toward it.
    pairs: list[tuple[float, float, float, UUID, str]] = []
    for other in neighbours:
        if other not in others:
            continue
        nb = others[other].box
        gap, head_ratio, lean, mine, theirs = FAR, FAR, [], 0.0, 0.0
        ref = pair_median(other)
        for i in seen:
            frame = state.frames[i]
            okps = frame.poses.get(other)
            if okps is None:
                continue
            kps = frame.poses[seat]
            size = max(min(box.height, nb.height), 1e-6)
            for _a, p in _wrists(kps):
                mine = max(mine, _toward(p, box, nb))
                for _b, q in _wrists(okps):
                    gap = min(gap, distance(p, q) / size)
            for _b, q in _wrists(okps):
                theirs = max(theirs, _toward(q, nb, box))
            head = head_point(kps, box)
            if ref > 0:
                head_ratio = min(head_ratio, distance(head, head_point(okps, nb)) / ref)
            side = 1.0 if nb.center[0] >= box.center[0] else -1.0
            lean.append(side * (head[0] - state.head_x_base) / max(box.width, 1e-6))
        # A tie (neither reached) goes to the smaller seat id, so both seats' rows name the same reacher.
        if mine != theirs:
            reacher_id = str(seat) if mine > theirs else str(other)
        else:
            reacher_id = min(str(seat), str(other))
        pairs.append((gap, head_ratio, _mean(lean), other, reacher_id))
    if not pairs:
        v["wrist_gap_min"], v["head_gap_ratio_min"], v["lean_toward"] = FAR, FAR, 0.0
        partner = reacher = None
    else:
        v["wrist_gap_min"] = min(item[0] for item in pairs)
        v["head_gap_ratio_min"] = min(item[1] for item in pairs)
        v["lean_toward"] = max(item[2] for item in pairs)
        if v["wrist_gap_min"] < FAR:
            best = min(pairs, key=lambda item: item[0])
        elif v["head_gap_ratio_min"] < FAR:
            best = min(pairs, key=lambda item: item[1])
        else:
            best = max(pairs, key=lambda item: item[2])
        partner, reacher = str(best[3]), best[4]

    def objects(labels: tuple[str, ...]) -> tuple[float, float, float, float]:
        conf, near, raised = 0.0, 0, 0.0
        centres: list[tuple[float, float]] = []
        for i in seen:
            frame = state.frames[i]
            hands = [p for _j, p in _wrists(frame.poses[seat])]
            hit = False
            for obj in frame.objects:
                c = obj.bbox.center
                if obj.label not in labels or not (box.x <= c[0] <= box.x + box.width and box.y <= c[1] <= box.y + box.height):
                    continue
                conf = max(conf, obj.confidence)
                centres.append(c)
                if any(distance(c, p) < OBJECT_WRIST_WIDTHS * box.width for p in hands):
                    hit = True
                    if c[1] < state.desk[i] - RAISED_HEIGHTS * box.height:
                        raised = 1.0
            near += hit
        moved = max((distance(a, b) for a in centres for b in centres), default=0.0) / max(box.width, 1e-6)
        return conf, near / len(seen), raised, moved

    v["phone_conf_max"], v["phone_near_wrist_share"], v["phone_raised"], phone_moved = objects(("phone",))
    v["paper_conf_max"], v["paper_near_wrist_share"], _paper_raised, paper_moved = objects(("paper", "book"))
    v["object_moved"] = max(phone_moved, paper_moved)
    for name in _RULE_FLAGS:
        v[f"rule_{name}"] = float(any(a < end and b >= start for a, b in spans.get(name, ())))
    confs = [p[2] for i in seen for p in state.frames[i].poses[seat] if len(p) > 2]
    v["kp_conf_mean"] = _mean(confs)
    v["seat_height"] = box.height
    v["frames_seen_share"] = len(seen) / len(idx)
    values = {name: float(v[name]) if math.isfinite(v[name]) else 0.0 for name in FEATURE_NAMES}
    return values, partner, reacher


def video_features(video: str, windows: Sequence[SuspectWindow],
                   frames: Mapping[CacheKey, list[ReviewFrame]]) -> list[FeatureRow]:
    """Rows for every seat window of ``windows`` (one per seat, from ``all_seat_windows``) whose frames
    are in ``frames``; a 3 s window where the seat is never seen gives no row."""

    states: dict[UUID, _Seat] = {}
    raw: dict[UUID, tuple[list[float], list[float], int]] = {}
    by_seat: dict[UUID, SuspectWindow] = {}
    for window in windows:
        found = frames.get(window_key(window))
        if not found:
            continue
        state, yaws, lifts, span = _prepare(window, found)
        states[window.seat_id], raw[window.seat_id], by_seat[window.seat_id] = state, (yaws, lifts, span), window
    _baselines(states, raw)

    rows: list[FeatureRow] = []
    for seat, window in by_seat.items():
        state = states[seat]
        spans = _rule_spans(window, state.frames)
        medians: dict[UUID, float] = {}

        def pair_median(other: UUID, state: _Seat = state, medians: dict[UUID, float] = medians) -> float:
            if other not in medians:
                gaps = [distance(head_point(f.poses[state.seat], state.box), head_point(f.poses[other], states[other].box))
                        for f in state.frames if f.poses.get(state.seat) is not None and f.poses.get(other) is not None]
                medians[other] = median(gaps) if gaps else 0.0
            return medians[other]

        neighbours = [other for other in window.neighbor_ids if other in states]
        for start in range(window.start_ms, window.end_ms - WIN_MS + 1, STEP_MS):
            out = _row_values(state, states, neighbours, start, start + WIN_MS, spans, pair_median)
            if out is not None:
                values, partner, reacher = out
                rows.append(FeatureRow(video, seat, start, start + WIN_MS, state.box, values, partner, reacher))
    return rows


_HEAD = ("video", "seat_id", "start_ms", "end_ms", "box_x", "box_y", "box_w", "box_h", "partner", "reacher")


def write_csv(path: Path, rows: Sequence[FeatureRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((*_HEAD, *FEATURE_NAMES))
        for r in rows:
            b = r.seat_box
            writer.writerow((r.video, str(r.seat_id), r.start_ms, r.end_ms, *(round(x, 6) for x in (b.x, b.y, b.width, b.height)),
                             r.partner or "", r.reacher or "", *(round(r.values[name], 6) for name in FEATURE_NAMES)))


def read_csv(path: Path) -> list[FeatureRow]:
    """Rows written by ``write_csv``; a file whose columns are not ``FEATURE_NAMES`` raises ``ValueError``."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in FEATURE_NAMES if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: thiếu cột đặc trưng {', '.join(missing)}")
        return [FeatureRow(item["video"], UUID(item["seat_id"]), int(item["start_ms"]), int(item["end_ms"]),
                           Box(float(item["box_x"]), float(item["box_y"]), float(item["box_w"]), float(item["box_h"])),
                           {name: float(item[name]) for name in FEATURE_NAMES},
                           item["partner"] or None, item["reacher"] or None) for item in reader]


__all__ = ["FEATURE_NAMES", "RULE_TYPES", "STEP_MS", "WIN_MS", "FeatureRow", "read_csv", "video_features", "write_csv"]
