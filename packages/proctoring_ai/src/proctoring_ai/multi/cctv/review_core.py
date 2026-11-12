"""One suspect window through the deep tier: crop, run the big models in one batch, apply the rules.

Shared by the offline ``scripts/review_suspects.py`` and the live ``ReviewWorker``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .crops import crop_image, crop_region
from .review_rules import ReviewFrame, ReviewRuleConfig, assign_poses, review_window
from .suspects import SuspectWindow


def review_seats(window: SuspectWindow, limit: int = 2) -> dict[Any, Any]:
    """The window's seat and its ``limit`` nearest neighbours: a crop over a whole row would shrink, not enlarge."""

    seats = dict(window.seats) or {window.seat_id: window.bbox}
    main = seats.get(window.seat_id, window.bbox)
    others = sorted((key for key in seats if key != window.seat_id),
                    key=lambda key: (seats[key].center[0] - main.center[0]) ** 2
                    + (seats[key].center[1] - main.center[1]) ** 2)
    return {window.seat_id: main, **{key: seats[key] for key in others[:limit]}}


class ReviewHistory:
    """Frames the deep tier already analysed, per seat, for the next piece of the same seat.

    The live tier reviews a long window in 10 s pieces. The rules hold a behaviour for seconds and learn the
    desk line and head baselines from what came before; without the earlier piece's frames a lap posture
    that started 2 s before the cut is cut in two and missed. Those frames cost no model time.
    """

    def __init__(self, keep_ms: int = 20000, max_gap_ms: int = 1000) -> None:
        self.keep_ms, self.max_gap_ms = keep_ms, max_gap_ms
        self._frames: dict[Any, list[ReviewFrame]] = {}

    def before(self, window: SuspectWindow) -> list[ReviewFrame]:
        seats = set(review_seats(window))
        found = [frame for frame in self._frames.get(window.seat_id, [])
                 if window.start_ms - self.keep_ms <= frame.offset_ms < window.start_ms and set(frame.seats) == seats]
        # Only a piece that runs straight on: across a hole the old frames are no warm-up.
        return found if found and window.start_ms - found[-1].offset_ms <= self.max_gap_ms else []

    def add(self, window: SuspectWindow, frames: Sequence[ReviewFrame]) -> None:
        kept = {frame.offset_ms: frame for frame in self._frames.get(window.seat_id, [])}
        kept.update({frame.offset_ms: frame for frame in frames})
        newest = max(kept, default=0)
        self._frames[window.seat_id] = [kept[t] for t in sorted(kept) if t >= newest - self.keep_ms]


def analyse_window(window: SuspectWindow, images: Sequence[tuple[int, Any]], detectors: Any, *,
                   config: ReviewRuleConfig | None = None,
                   context: Sequence[ReviewFrame] = ()) -> tuple[list[dict[str, Any]], list[ReviewFrame]]:
    """``context``: frames already analysed just before (``ReviewHistory.before``); the rules see them, the
    models do not run on them again, and only the new frames are returned."""

    seats = review_seats(window)
    by_body = (config or ReviewRuleConfig()).assign_by_body
    frames: list[ReviewFrame] = []
    if images:
        region = crop_region(list(seats.values()))
        crops, transforms = zip(*(crop_image(image, region) for _, image in images), strict=True)
        people = detectors.pose_many(list(crops), list(transforms))
        objects = detectors.objects_many(list(crops), list(transforms))
        for (offset, image), found, things in zip(images, people, objects, strict=True):
            height, width = image.shape[:2]
            frames.append(ReviewFrame(offset, seats, assign_poses(seats, found, by_body=by_body), things, width / height))
    earlier = [frame for frame in context if not frames or frame.offset_ms < frames[0].offset_ms]
    return review_window(window, [*earlier, *frames], config), frames


__all__ = ["ReviewHistory", "analyse_window", "review_seats"]
