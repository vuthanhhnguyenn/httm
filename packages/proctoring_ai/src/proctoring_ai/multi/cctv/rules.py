"""Event rules for CCTV mode, evaluated on candidates only.

Each rule is a condition per frame; a latch opens the event once the condition has held for
``hold_ms`` (gaps shorter than ``gap_ms`` tolerated) and closes it after ``release_ms`` without it.
Records use the ``events.jsonl`` shape of webcam mode (``opened``/``closed``, ``id``, ``type``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from proctoring_core.observations import Box
from proctoring_core.types import EventType

from ..pipeline import MultiFrameResult
from ..tracker import Seat
from .head import RollingBaseline, distance, facing, head_point, head_yaw_index, point
from .pose_detector import LEFT_WRIST, RIGHT_WRIST
from .roles import CANDIDATE, SeatRoleClassifier


@dataclass(frozen=True, slots=True)
class CctvRuleConfig:
    yaw_threshold: float = 0.35
    turn_hold_ms: int = 1_500
    together_ratio: float = 0.6
    together_hold_ms: int = 1_500
    hand_hold_ms: int = 1_000
    phone_hold_ms: int = 1_000
    phone_min_confidence: float = 0.5
    release_ms: int = 1_000
    gap_ms: int = 400
    neighbor_gap_widths: float = 2.5
    pair_gap_widths: float = 1.6


@dataclass(slots=True)
class _Latch:
    hold_ms: int
    since: int | None = None
    last_true: int | None = None
    event_id: str | None = None
    opened_ms: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def _inside(p: tuple[float, float], box: Box) -> bool:
    return box.x <= p[0] <= box.x + box.width and box.y <= p[1] <= box.y + box.height


class CctvSeatRules:
    """Same interface as the webcam ``_SeatRules`` so ``run_multi_replay`` can drive either."""

    def __init__(self, config: CctvRuleConfig | None = None, roles: SeatRoleClassifier | None = None) -> None:
        self.config = config or CctvRuleConfig()
        self.roles = roles or SeatRoleClassifier()
        self._latches: dict[tuple[str, ...], _Latch] = {}
        self._yaw: dict[UUID, RollingBaseline] = {}
        self._pair_distance: dict[tuple[UUID, UUID], RollingBaseline] = {}
        self._seen_face: set[UUID] = set()
        self._runtime_ids: dict[UUID, int] = {}
        self._open: dict[UUID, dict[str, str]] = {}
        self._role_of: dict[UUID, str] = {}
        self.opened_by_type: dict[str, int] = {}

    # -- latch bookkeeping -------------------------------------------------------------
    def _record(self, action: str, seat_id: UUID, event_type: str, latch: _Latch, offset: int,
                monotonic_ms: int, **extra: Any) -> tuple[UUID, dict[str, Any]]:
        item = {"action": action, "id": latch.event_id, "type": event_type, "seat_id": str(seat_id),
                "runtime_track_id": self._runtime_ids.get(seat_id), "video_offset_ms": offset,
                "monotonic_ms": monotonic_ms, "scene": "cctv", **latch.details, **extra}
        labels = self._open.setdefault(seat_id, {})
        if action == "opened":
            labels[str(latch.event_id)] = event_type
            self.opened_by_type[event_type] = self.opened_by_type.get(event_type, 0) + 1
        else:
            labels.pop(str(latch.event_id), None)
        return seat_id, item

    def _close(self, key: tuple[str, ...], latch: _Latch, offset: int, ms: int,
               **extra: Any) -> tuple[UUID, dict[str, Any]]:
        record = self._record("closed", UUID(key[1]), key[0], latch, offset, ms,
                              duration_ms=ms - latch.opened_ms, **extra)
        latch.event_id, latch.since, latch.last_true, latch.details = None, None, None, {}
        return record

    def _feed(self, key: tuple[str, ...], hold_ms: int, active: bool, ms: int, offset: int,
              details: Mapping[str, Any] | None = None) -> list[tuple[UUID, dict[str, Any]]]:
        """``key`` = (event type, seat id[, partner]): one latch per seat and rule."""

        cfg = self.config
        latch = self._latches.setdefault(key, _Latch(hold_ms))
        out: list[tuple[UUID, dict[str, Any]]] = []
        if active:
            if latch.since is None or (latch.last_true is not None and ms - latch.last_true > cfg.gap_ms):
                latch.since = ms if latch.event_id is None else latch.since
            latch.last_true = ms
            if latch.event_id is None and latch.since is not None and ms - latch.since >= latch.hold_ms:
                latch.event_id, latch.opened_ms, latch.details = str(uuid4()), ms, dict(details or {})
                out.append(self._record("opened", UUID(key[1]), key[0], latch, offset, ms))
        elif latch.event_id is not None:
            if latch.last_true is None or ms - latch.last_true >= cfg.release_ms:
                out.append(self._close(key, latch, offset, ms))
        elif latch.last_true is None or ms - latch.last_true > cfg.gap_ms:
            latch.since = None
        return out

    # -- per-frame evaluation -------------------------------------------------------------
    def _neighbor_on_side(self, seat: Seat, head: tuple[float, float], side: int,
                          heads: Mapping[UUID, tuple[float, float]], seats: Mapping[UUID, Seat]) -> UUID | None:
        best, best_dx = None, 0.0
        for other_id, other_head in heads.items():
            if other_id == seat.seat_id:
                continue
            dx = (other_head[0] - head[0]) * side
            width = (seat.bbox.width + seats[other_id].bbox.width) / 2
            close = 0 < dx <= self.config.neighbor_gap_widths * width
            level = abs(other_head[1] - head[1]) <= seat.bbox.height / 2
            if close and level and (best is None or dx < best_dx):
                best, best_dx = other_id, dx
        return best

    async def step(self, result: MultiFrameResult, frame: Mapping[str, Any],
                   offset: int) -> list[tuple[UUID, dict[str, Any]]]:
        cfg, ms = self.config, result.monotonic_ms
        out: list[tuple[UUID, dict[str, Any]]] = []
        seats = {seat.seat_id: seat for seat in result.tracker.analyzed}
        for seat in (*result.tracker.analyzed, *result.tracker.missing):
            self._runtime_ids[seat.seat_id] = seat.runtime_track_id
        for seat_id, seat in seats.items():
            self._role_of[seat_id] = self.roles.update(seat_id, seat.bbox, ms, result.poses.get(seat_id))
        candidates = {seat_id: seat for seat_id, seat in seats.items() if self._role_of[seat_id] == CANDIDATE}
        heads = {seat_id: head_point(result.poses.get(seat_id), seat.bbox) for seat_id, seat in candidates.items()}
        phones = [box for item in result.objects if str(item.get("class", "")).casefold() == "cell phone"
                  and float(item.get("confidence", 0.0)) >= cfg.phone_min_confidence
                  and (box := Box.parse(item.get("bbox"))) is not None]
        active: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = {}

        for seat_id, seat in candidates.items():
            kps = result.poses.get(seat_id)
            seat_key = str(seat_id)
            view = facing(kps)
            if view == "front":
                self._seen_face.add(seat_id)
            yaw = head_yaw_index(kps)
            baseline = self._yaw.setdefault(seat_id, RollingBaseline())
            reference = baseline.value(ms)
            if yaw is not None:
                baseline.add(yaw, ms)
            if yaw is not None and reference is not None and abs(yaw - reference) > cfg.yaw_threshold:
                side = 1 if yaw > reference else -1
                neighbor = self._neighbor_on_side(seat, heads[seat_id], side, heads, candidates)
                details: dict[str, Any] = {"yaw_index": round(yaw - reference, 2)}
                if neighbor is not None:
                    details["neighbor_seat_id"] = str(neighbor)
                    active[(EventType.LOOK_AT_NEIGHBOR.value, seat_key)] = (cfg.turn_hold_ms, details)
                else:
                    # Candidates face the camera: nose towards image right is their left.
                    turn = EventType.HEAD_TURN_LEFT if side > 0 else EventType.HEAD_TURN_RIGHT
                    active[(turn.value, seat_key)] = (cfg.turn_hold_ms, details)
            if view == "back" and seat_id in self._seen_face:
                active[(EventType.TURN_AROUND.value, seat_key)] = (cfg.turn_hold_ms, {})
            if self.roles.left_seat(seat_id, ms):
                active[(EventType.LEAVING_SEAT.value, seat_key)] = (0, {})
            wrists = [p for index in (LEFT_WRIST, RIGHT_WRIST) if (p := point(kps, index, 0.4))]
            for other_id, other in candidates.items():
                if other_id != seat_id and any(_inside(p, other.bbox) and not _inside(p, seat.bbox) for p in wrists):
                    active[(EventType.SUSPICIOUS_HAND_ACTIVITY.value, seat_key)] = (
                        cfg.hand_hold_ms, {"neighbor_seat_id": str(other_id)})
            for phone in phones:
                centre = phone.center
                if _inside(centre, seat.bbox):
                    owner = min((s for s in candidates.values() if _inside(centre, s.bbox)),
                                key=lambda s: distance(centre, s.bbox.center))
                    if owner.seat_id == seat_id:
                        active[(EventType.PHONE_DETECTED.value, seat_key)] = (cfg.phone_hold_ms, {})

        ordered = sorted(candidates, key=str)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                a, b = candidates[first], candidates[second]
                width = (a.bbox.width + b.bbox.width) / 2
                gap = distance(heads[first], heads[second])
                if gap > cfg.pair_gap_widths * width * 2:
                    continue
                baseline = self._pair_distance.setdefault((first, second), RollingBaseline())
                reference = baseline.value(ms)
                baseline.add(gap, ms)
                if reference is not None and distance(a.bbox.center, b.bbox.center) <= cfg.pair_gap_widths * width \
                        and gap < cfg.together_ratio * reference:
                    for seat_id, partner in ((first, second), (second, first)):
                        active[(EventType.HEADS_TOGETHER.value, str(seat_id), str(partner))] = (
                            cfg.together_hold_ms, {"neighbor_seat_id": str(partner)})

        for key in sorted(set(self._latches) | set(active)):
            hold, details = active.get(key, (0, {}))
            if key in active or self._latches[key].event_id is not None or self._latches[key].last_true is not None:
                out += self._feed(key, hold, key in active, ms, offset, details)
        for seat in result.tracker.expired:
            out += self._drop(seat.seat_id, offset, ms, reason="SEAT_EXPIRED")
        return out

    def _drop(self, seat_id: UUID, offset: int, ms: int, *, reason: str) -> list[tuple[UUID, dict[str, Any]]]:
        out = []
        for key in [key for key in self._latches if key[1] == str(seat_id)]:
            latch = self._latches.pop(key)
            if latch.event_id is not None:
                out.append(self._close(key, latch, offset, ms, close_reason=reason))
        self.roles.forget(seat_id)
        self._yaw.pop(seat_id, None)
        self._seen_face.discard(seat_id)
        self._role_of.pop(seat_id, None)
        for pair in [pair for pair in self._pair_distance if seat_id in pair]:
            self._pair_distance.pop(pair)
        return out

    def focus_seats(self) -> list[UUID]:
        return []

    def finish(self, frame: Mapping[str, Any], offset: int) -> list[tuple[UUID, dict[str, Any]]]:
        ms = int(frame.get("captured_at_monotonic_ms", 0))
        out = []
        for key, latch in sorted(self._latches.items()):
            if latch.event_id is not None:
                out.append(self._close(key, latch, offset, ms, close_reason="END_OF_VIDEO"))
        return out

    def labels(self) -> dict[UUID, list[str]]:
        return {seat_id: list(labels.values()) for seat_id, labels in self._open.items()}

    def seat_roles(self) -> dict[UUID, str]:
        return dict(self._role_of)

    def summary(self) -> dict[str, Any]:
        return {"roles": self.roles.counts(), "events_opened": dict(sorted(self.opened_by_type.items()))}


def keypoints_for(seats: Sequence[Seat], people: Sequence[Mapping[str, Any]]) -> dict[UUID, list[list[float]]]:
    """The tracker copies detection boxes into seats, so the matching detection is the best-IoU one."""

    from ..tracker import iou

    boxes = [(Box.parse(person.get("bbox")), person.get("keypoints")) for person in people]
    result: dict[UUID, list[list[float]]] = {}
    for seat in seats:
        best = max(((iou(seat.bbox, box), kps) for box, kps in boxes if box is not None and kps),
                   key=lambda item: item[0], default=None)
        if best is not None and best[0] > 0.5:
            result[seat.seat_id] = best[1]
    return result


__all__ = ["CctvRuleConfig", "CctvSeatRules", "keypoints_for"]
