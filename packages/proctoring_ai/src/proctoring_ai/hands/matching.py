"""Keep hand identities stable between hand passes.

MediaPipe returns hands in no particular order, so pairing by list index reports
motion that never happened. Hands are matched by handedness first, then by the
nearest wrist; a hand that moved implausibly far is treated as a new hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist


@dataclass(slots=True)
class TrackedHand:
    hand_id: int
    handedness: str
    wrist: tuple[float, float]
    seen_ms: int


class HandMatcher:
    def __init__(self, *, max_jump: float = 0.25, max_gap_ms: int = 1000) -> None:
        self.max_jump = max_jump
        self.max_gap_ms = max_gap_ms
        self._tracks: list[TrackedHand] = []
        self._next_id = 1

    def update(self, hands: list[tuple[str, tuple[float, float]]], timestamp_ms: int) -> list[tuple[int, float, int]]:
        """Return ``(hand_id, displacement, dt_ms)`` per input hand, in input order.

        ``displacement`` is the wrist move in normalized image units since the previous
        pass (0 for a new hand); ``dt_ms`` is the time since that pass.
        """

        live = [track for track in self._tracks if 0 <= timestamp_ms - track.seen_ms <= self.max_gap_ms]
        free = list(live)
        pairs: list[tuple[float, int, TrackedHand]] = []
        for index, (handedness, wrist) in enumerate(hands):
            for track in free:
                same_side = track.handedness == handedness or "unknown" in {track.handedness, handedness}
                distance = dist(track.wrist, wrist)
                if same_side and distance <= self.max_jump:
                    pairs.append((distance, index, track))
        assigned: dict[int, TrackedHand] = {}
        results: list[tuple[int, float, int]] = []
        used: set[int] = set()
        for _distance, index, track in sorted(pairs, key=lambda item: item[0]):
            if index in assigned or track.hand_id in used:
                continue
            assigned[index] = track
            used.add(track.hand_id)
        updated: list[TrackedHand] = []
        for index, (handedness, wrist) in enumerate(hands):
            matched = assigned.get(index)
            if matched is None:
                matched = TrackedHand(self._next_id, handedness, wrist, timestamp_ms)
                self._next_id += 1
                results.append((matched.hand_id, 0.0, 0))
            else:
                results.append((matched.hand_id, dist(matched.wrist, wrist), timestamp_ms - matched.seen_ms))
                matched.wrist, matched.seen_ms, matched.handedness = wrist, timestamp_ms, handedness
            updated.append(matched)
        self._tracks = updated + [track for track in live if track.hand_id not in {t.hand_id for t in updated}]
        return results


__all__ = ["HandMatcher", "TrackedHand"]
