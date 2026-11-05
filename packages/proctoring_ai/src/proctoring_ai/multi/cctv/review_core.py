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


def analyse_window(window: SuspectWindow, images: Sequence[tuple[int, Any]], detectors: Any, *,
                   config: ReviewRuleConfig | None = None) -> tuple[list[dict[str, Any]], list[ReviewFrame]]:
    seats = review_seats(window)
    frames: list[ReviewFrame] = []
    if images:
        region = crop_region(list(seats.values()))
        crops, transforms = zip(*(crop_image(image, region) for _, image in images), strict=True)
        people = detectors.pose_many(list(crops), list(transforms))
        objects = detectors.objects_many(list(crops), list(transforms))
        for (offset, image), found, things in zip(images, people, objects, strict=True):
            height, width = image.shape[:2]
            frames.append(ReviewFrame(offset, seats, assign_poses(seats, found), things, width / height))
    return review_window(window, frames, config), frames


__all__ = ["analyse_window", "review_seats"]
