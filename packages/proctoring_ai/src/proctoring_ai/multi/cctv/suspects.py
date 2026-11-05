"""Fast tier of phase 2e: mark *suspect windows* per seat, erring towards too many.

Each reason is a cheap test on the keypoints the CCTV pipeline already has. A reason that holds for
``hold_ms`` marks the span; spans are padded, clamped to the clip and merged per seat in ``finish``.
The deep tier (``scripts/review_suspects.py``) then looks at each window with bigger models.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from proctoring_core.observations import Box

from ..tracker import Seat
from .head import Keypoints, RollingBaseline, distance, head_lift, head_point, head_yaw_index, point
from .pose_detector import LEFT_WRIST, RIGHT_WRIST

REASONS = ("HEAD_AWAY", "LEAN", "HAND_OUT", "HANDS_MEET", "HEAD_DOWN_HANDS_LOW", "OBJECT_HINT", "RULE_EVENT")


@dataclass(frozen=True, slots=True)
class CctvSuspectConfig:
    yaw_factor: float = 0.8
    yaw_threshold: float = 0.35  # CctvRuleConfig.yaw_threshold
    yaw_spread_factor: float = 2.5  # CctvRuleConfig.yaw_spread_factor
    lean_widths: float = 0.35
    hand_out_widths: float = 0.15
    hands_meet_widths: float = 0.5
    pair_gap_widths: float = 2.5
    object_min_conf: float = 0.15
    min_head_lift: float = 0.1
    hold_ms: int = 1000
    gap_ms: int = 500
    pad_ms: int = 2000
    desk_default: float = 0.8
    desk_margin: float = 0.2
    desk_min_samples: int = 5
    baseline_min_span_ms: int = 1000


def _box_json(box: Box) -> dict[str, float]:
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _union(a: Box, b: Box) -> Box:
    if a == b:
        return a
    x0, y0 = min(a.x, b.x), min(a.y, b.y)
    x1, y1 = max(a.x + a.width, b.x + b.width), max(a.y + a.height, b.y + b.height)
    return Box(x0, y0, x1 - x0, y1 - y0)


@dataclass(frozen=True, slots=True)
class SuspectWindow:
    seat_id: UUID
    runtime_track_id: int | None
    neighbor_ids: tuple[UUID, ...]
    start_ms: int
    end_ms: int
    reasons: tuple[str, ...]
    bbox: Box
    # Every seat the deep tier must look at (this one and its neighbours), with its box.
    seats: Mapping[UUID, Box] = field(default_factory=dict)
    # When the most fast-tier reasons held at once (earliest on a tie): the deep tier always looks there.
    peak_ms: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"seat_id": str(self.seat_id), "runtime_track_id": self.runtime_track_id,
                "neighbor_ids": [str(item) for item in self.neighbor_ids], "start_ms": self.start_ms,
                "end_ms": self.end_ms, "reasons": list(self.reasons), "bbox": _box_json(self.bbox),
                "seats": {str(key): _box_json(box) for key, box in self.seats.items()}, "peak_ms": self.peak_ms}

    @classmethod
    def from_json(cls, item: Mapping[str, Any]) -> SuspectWindow:
        def box(value: Mapping[str, Any]) -> Box:
            return Box(float(value["x"]), float(value["y"]), float(value["width"]), float(value["height"]))

        return cls(UUID(item["seat_id"]), item.get("runtime_track_id"),
                   tuple(UUID(value) for value in item.get("neighbor_ids", ())), int(item["start_ms"]),
                   int(item["end_ms"]), tuple(item.get("reasons", ())), box(item["bbox"]),
                   {UUID(key): box(value) for key, value in item.get("seats", {}).items()},
                   None if item.get("peak_ms") is None else int(item["peak_ms"]))


class DeskLine:
    """Where this seat's desk is: median wrist height while writing (head down, wrist in the box)."""

    def __init__(self, config: CctvSuspectConfig | None = None) -> None:
        self.config = config or CctvSuspectConfig()
        self._samples = RollingBaseline(window_ms=60_000, min_span_ms=0, min_samples=self.config.desk_min_samples)

    def add(self, wrist_y: float, ms: int) -> None:
        self._samples.add(wrist_y, ms)

    def value(self, box: Box, ms: int) -> float:
        median = self._samples.value(ms)
        if median is None:
            return box.y + self.config.desk_default * box.height
        return median + self.config.desk_margin * box.height


@dataclass(slots=True)
class _Span:
    start: int
    end: int
    reasons: set[str]
    neighbors: set[UUID]
    seats: dict[UUID, Box]
    marks: dict[int, int] = field(default_factory=dict)  # video ms -> reasons marked at that frame

    def peak(self) -> int | None:
        return min(self.marks, key=lambda t: (-self.marks[t], t)) if self.marks else None


class SuspectTracker:
    def __init__(self, config: CctvSuspectConfig | None = None) -> None:
        self.config = config or CctvSuspectConfig()
        self._yaw: dict[UUID, RollingBaseline] = {}
        self._home: dict[UUID, tuple[RollingBaseline, RollingBaseline]] = {}
        self._desk: dict[UUID, DeskLine] = {}
        self._hold: dict[tuple[UUID, str], tuple[int, int]] = {}  # (since, last true) in video ms
        self._spans: dict[UUID, list[_Span]] = {}
        self._runtime_ids: dict[UUID, int] = {}
        self._emitted: dict[UUID, int] = {}  # per seat: windows up to this video ms were handed out by poll

    def _baseline(self) -> RollingBaseline:
        return RollingBaseline(min_span_ms=self.config.baseline_min_span_ms, min_samples=5)

    def observe(self, candidates: Mapping[UUID, Seat], poses: Mapping[UUID, Keypoints],
                objects: Sequence[Mapping[str, Any]], offset_ms: int, ms: int, frame_aspect: float,
                rule_event_seats: Iterable[UUID]) -> None:
        cfg = self.config
        heads = {seat_id: head_point(poses.get(seat_id), seat.bbox) for seat_id, seat in candidates.items()}
        found: dict[UUID, dict[str, set[UUID]]] = {seat_id: {} for seat_id in candidates}
        near: dict[UUID, set[UUID]] = {seat_id: set() for seat_id in candidates}
        for first, a in candidates.items():
            for second, b in candidates.items():
                width = (a.bbox.width + b.bbox.width) / 2
                if first != second and distance(heads[first], heads[second]) <= cfg.pair_gap_widths * width:
                    near[first].add(second)

        for seat_id, seat in candidates.items():
            self._runtime_ids[seat_id] = seat.runtime_track_id
            kps, box = poses.get(seat_id), seat.bbox
            reasons = found[seat_id]
            lift = head_lift(kps, frame_aspect)
            head_down = lift is not None and lift < cfg.min_head_lift
            yaw = head_yaw_index(kps)
            baseline = self._yaw.setdefault(seat_id, self._baseline())
            reference = baseline.value(ms)
            threshold = cfg.yaw_factor * max(cfg.yaw_threshold, cfg.yaw_spread_factor * baseline.spread(ms))
            if yaw is not None:
                if reference is not None and abs(yaw - reference) > threshold:
                    reasons["HEAD_AWAY"] = set()
                baseline.add(yaw, ms)
            home_x, home_y = self._home.setdefault(seat_id, (self._baseline(), self._baseline()))
            hx, hy = home_x.value(ms), home_y.value(ms)
            if hx is not None and hy is not None and distance(heads[seat_id], (hx, hy)) > cfg.lean_widths * box.width:
                reasons["LEAN"] = set()
            home_x.add(heads[seat_id][0], ms)
            home_y.add(heads[seat_id][1], ms)
            wrists = [p for index in (LEFT_WRIST, RIGHT_WRIST) if (p := point(kps, index, 0.4))]
            margin = cfg.hand_out_widths * box.width
            if any(p[0] < box.x - margin or p[0] > box.x + box.width + margin for p in wrists):
                reasons["HAND_OUT"] = set()
            desk = self._desk.setdefault(seat_id, DeskLine(cfg))
            line = desk.value(box, ms)
            if head_down and wrists and all(p[1] > line for p in wrists):
                reasons["HEAD_DOWN_HANDS_LOW"] = set()
            elif head_down:
                for p in wrists:
                    if box.y <= p[1] <= box.y + box.height and box.x <= p[0] <= box.x + box.width:
                        desk.add(p[1], ms)
            for other_id in near[seat_id]:
                other = candidates[other_id]
                theirs = [p for index in (LEFT_WRIST, RIGHT_WRIST) if (p := point(poses.get(other_id), index, 0.4))]
                width = (box.width + other.bbox.width) / 2
                if any(distance(p, q) < cfg.hands_meet_widths * width for p in wrists for q in theirs):
                    reasons.setdefault("HANDS_MEET", set()).add(other_id)
            for item in objects:
                parsed = Box.parse(item.get("bbox"))
                if (str(item.get("class", "")).casefold() == "cell phone" and parsed is not None
                        and float(item.get("confidence", 0.0)) >= cfg.object_min_conf):
                    cx, cy = parsed.center
                    if box.x <= cx <= box.x + box.width and box.y <= cy <= box.y + box.height:
                        reasons["OBJECT_HINT"] = set()
        for seat_id in rule_event_seats:
            if seat_id in found:
                found[seat_id]["RULE_EVENT"] = set()

        for seat_id, seat in candidates.items():
            for reason in REASONS:
                key = (seat_id, reason)
                if reason not in found[seat_id]:
                    held = self._hold.get(key)
                    if held is not None and offset_ms - held[1] > cfg.gap_ms:
                        del self._hold[key]
                    continue
                since, last = self._hold.get(key, (offset_ms, offset_ms))
                if offset_ms - last > cfg.gap_ms:
                    since = offset_ms
                self._hold[key] = (since, offset_ms)
                if reason == "RULE_EVENT" or offset_ms - since >= cfg.hold_ms:
                    involved = near[seat_id] | found[seat_id][reason]
                    self._mark(seat_id, reason, since, offset_ms, involved,
                               {seat_id: seat.bbox, **{other: candidates[other].bbox for other in involved}})

    def _mark(self, seat_id: UUID, reason: str, start: int, end: int, neighbors: set[UUID],
              seats: dict[UUID, Box]) -> None:
        spans = self._spans.setdefault(seat_id, [])
        if spans and start <= spans[-1].end + self.config.gap_ms:
            span = spans[-1]
            span.start, span.end = min(span.start, start), max(span.end, end)
        else:
            span = _Span(start, end, set(), set(), {})
            spans.append(span)
        span.reasons.add(reason)
        span.marks[end] = span.marks.get(end, 0) + 1
        span.neighbors |= neighbors
        for key, box in seats.items():
            span.seats[key] = _union(span.seats[key], box) if key in span.seats else box

    def _merged(self, seat_id: UUID, duration_ms: int | None) -> list[_Span]:
        """The seat's spans padded by ``pad_ms`` each side and merged where they overlap."""

        pad = self.config.pad_ms
        merged: list[_Span] = []
        for span in sorted(self._spans.get(seat_id, []), key=lambda item: item.start):
            start = max(0, span.start - pad)
            end = span.end + pad if duration_ms is None else min(duration_ms, span.end + pad)
            if merged and start <= merged[-1].end:
                last = merged[-1]
                last.end = max(last.end, end)
                last.reasons |= span.reasons
                last.neighbors |= span.neighbors
                for key, box in span.seats.items():
                    last.seats[key] = _union(last.seats[key], box) if key in last.seats else box
                for t, count in span.marks.items():
                    last.marks[t] = last.marks.get(t, 0) + count
            else:
                merged.append(_Span(start, end, set(span.reasons), set(span.neighbors), dict(span.seats),
                                    dict(span.marks)))
        return merged

    def _window(self, seat_id: UUID, span: _Span, start: int, end: int) -> SuspectWindow:
        marks = {t: c for t, c in span.marks.items() if start <= t <= end}
        peak = min(marks, key=lambda t: (-marks[t], t)) if marks else None
        return SuspectWindow(seat_id, self._runtime_ids.get(seat_id), tuple(sorted(span.neighbors, key=str)),
                             start, end, tuple(reason for reason in REASONS if reason in span.reasons),
                             span.seats[seat_id], span.seats, peak)

    def poll(self, offset_ms: int, *, max_open_ms: int = 10000) -> list[SuspectWindow]:
        """Windows that can no longer grow, each handed out once; a window open longer than ``max_open_ms``
        is handed out in ``max_open_ms`` pieces so the live deep tier does not wait for its end."""

        cfg = self.config
        # A new mark merges into a span up to gap_ms after it, and padded spans merge up to 2 x pad apart;
        # a mark can also start hold_ms before the frame that confirms it.
        settle = 2 * cfg.pad_ms + cfg.gap_ms + cfg.hold_ms
        out: list[SuspectWindow] = []
        for seat_id in list(self._spans):
            done = self._emitted.get(seat_id, -1)
            for span in self._merged(seat_id, None):
                start = max(span.start, done)
                if span.end <= done:
                    continue
                if offset_ms - (span.end - cfg.pad_ms) > settle:
                    # Settled: the rest in max_open_ms pieces, each handed out once.
                    while start < span.end:
                        end = span.end if span.end - start <= max_open_ms else start + max_open_ms
                        out.append(self._window(seat_id, span, start, end))
                        start = done = end
                    continue
                while offset_ms - start >= max_open_ms and start + max_open_ms < span.end:
                    out.append(self._window(seat_id, span, start, start + max_open_ms))
                    start = done = start + max_open_ms
            self._emitted[seat_id] = done
        return sorted(out, key=lambda item: (item.start_ms, str(item.seat_id)))

    def finish(self, duration_ms: int | None) -> list[SuspectWindow]:
        """Every window not yet handed out by ``poll`` (all of them when ``poll`` was never used)."""

        out: list[SuspectWindow] = []
        for seat_id in self._spans:
            done = self._emitted.get(seat_id, -1)
            for span in self._merged(seat_id, duration_ms):
                if span.end <= done:
                    continue
                out.append(self._window(seat_id, span, max(span.start, done), span.end))
            self._emitted[seat_id] = max([done] + [span.end for span in self._merged(seat_id, duration_ms)])
        return sorted(out, key=lambda item: (item.start_ms, str(item.seat_id)))


__all__ = ["CctvSuspectConfig", "DeskLine", "REASONS", "SuspectTracker", "SuspectWindow"]
