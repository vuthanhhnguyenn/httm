"""Hands and pose for the few seats currently in focus.

Each slot is bound to one seat because MediaPipe VIDEO mode tracks over time; a
seat taking over a freed slot gets fresh instances instead of another seat's state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import UUID

from .crop import Crop


@dataclass(frozen=True, slots=True)
class FocusResult:
    hands: list[dict[str, Any]]
    poses: list[dict[str, Any]]
    hands_status: str
    pose_status: str
    latency_ms: float


@dataclass(slots=True)
class _Slot:
    hand: Any | None
    pose: Any | None
    last_run: dict[str, int] = field(default_factory=dict)


class FocusAnalyzers:
    def __init__(self, hand_factory: Callable[[], Any] | None, pose_factory: Callable[[], Any] | None, *,
                 slots: int = 2, hands_interval_ms: int = 200, pose_interval_ms: int = 250) -> None:
        self.hand_factory = hand_factory
        self.pose_factory = pose_factory
        self.slots = slots
        self.intervals = {"hands": hands_interval_ms, "pose": pose_interval_ms}
        self._bound: dict[UUID, _Slot] = {}

    @staticmethod
    def _close(slot: _Slot) -> None:
        for analyzer in (slot.hand, slot.pose):
            if analyzer is not None:
                analyzer.close()

    async def _bind(self, seat_id: UUID) -> _Slot:
        slot = _Slot(self.hand_factory() if self.hand_factory else None,
                     self.pose_factory() if self.pose_factory else None)
        for analyzer in (slot.hand, slot.pose):
            if analyzer is not None:
                await analyzer.initialize()
        self._bound[seat_id] = slot
        return slot

    async def _one(self, name: str, slot: _Slot, analyzer: Any | None, crop: Crop, frame_id: int,
                   monotonic_ms: int) -> tuple[str, list[dict[str, Any]]]:
        if analyzer is None:
            return "not_run", []
        last = slot.last_run.get(name)
        if last is not None and monotonic_ms - last < self.intervals[name]:
            return "skipped", []
        slot.last_run[name] = monotonic_ms
        frame = {"image": crop.image, "frame_id": frame_id, "captured_at_monotonic_ms": monotonic_ms,
                 "frame_size": (crop.image.shape[1], crop.image.shape[0])}
        try:
            raw = await analyzer.analyze(frame)
        except Exception:
            return "error", []
        key, mapper = (("hands", crop.transform.map_hand) if name == "hands"
                       else ("poses", crop.transform.map_pose))
        return str(raw.get("status", "ready")), [mapper(item) for item in raw.get(key) or []]

    async def _analyze_seat(self, slot: _Slot, crop: Crop, frame_id: int, monotonic_ms: int) -> FocusResult:
        started = perf_counter()
        (hands_status, hands), (pose_status, poses) = await asyncio.gather(
            self._one("hands", slot, slot.hand, crop, frame_id, monotonic_ms),
            self._one("pose", slot, slot.pose, crop, frame_id, monotonic_ms),
        )
        return FocusResult(hands, poses, hands_status, pose_status, (perf_counter() - started) * 1000)

    async def analyze(self, crops: Mapping[UUID, Crop], *, frame_id: int, monotonic_ms: int) -> dict[UUID, FocusResult]:
        bound = [seat for seat in crops if seat in self._bound]
        chosen = (bound + [seat for seat in crops if seat not in self._bound])[:self.slots]
        for seat_id in [seat for seat in self._bound if seat not in chosen]:
            self._close(self._bound.pop(seat_id))
        slots = {seat: self._bound.get(seat) or await self._bind(seat) for seat in chosen}
        results = await asyncio.gather(*(self._analyze_seat(slots[seat], crops[seat], frame_id, monotonic_ms)
                                         for seat in chosen))
        return dict(zip(chosen, results, strict=True))

    def release(self, seat_id: UUID) -> None:
        slot = self._bound.pop(seat_id, None)
        if slot is not None:
            self._close(slot)

    def close(self) -> None:
        for seat_id in list(self._bound):
            self.release(seat_id)


__all__ = ["FocusAnalyzers", "FocusResult"]
