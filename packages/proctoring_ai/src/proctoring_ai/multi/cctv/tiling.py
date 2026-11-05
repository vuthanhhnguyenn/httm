"""Look closer at the far rows: extra YOLO-pose passes on enlarged tiles of the top of the frame.

Far candidates are ~40 px tall in a 720p room view, too small for one 960 px pass. Tiles over the
upper part of the frame are enlarged by the detector, and their people are mapped back and merged
with the full-frame pass. Duplicates (same head) are merged here too, also within a single pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from proctoring_core.observations import Box

from .head import distance, head_point


def tiles_for(*, cols: int = 2, top: float = 0.6, overlap: float = 0.15) -> list[tuple[float, float, float, float]]:
    """Normalised (x0, y0, x1, y1) tiles side by side over the top ``top`` of the frame."""

    width = 1.0 / cols
    pad = width * overlap / 2
    return [(max(0.0, i * width - pad), 0.0, min(1.0, (i + 1) * width + pad), top) for i in range(cols)]


def map_from_tile(person: dict[str, Any], tile: tuple[float, float, float, float]) -> dict[str, Any]:
    x0, y0, x1, y1 = tile
    sx, sy = x1 - x0, y1 - y0
    box = person["bbox"]
    return {**person,
            "bbox": {"x": x0 + box["x"] * sx, "y": y0 + box["y"] * sy,
                     "width": box["width"] * sx, "height": box["height"] * sy},
            "keypoints": [[x0 + p[0] * sx, y0 + p[1] * sy, p[2]] for p in person.get("keypoints", [])]}


def merge_people(people: Sequence[dict[str, Any]], *, same_head: float = 0.5) -> list[dict[str, Any]]:
    """Keep the most confident of people whose heads are closer than ``same_head`` × the narrower box."""

    kept: list[tuple[dict[str, Any], tuple[float, float], float]] = []
    for person in sorted(people, key=lambda item: -float(item.get("confidence", 0.0))):
        box = Box.parse(person.get("bbox"))
        if box is None:
            continue
        head = head_point(person.get("keypoints"), box)
        if any(distance(head, other) < same_head * min(box.width, width) for _, other, width in kept):
            continue
        kept.append((person, head, box.width))
    return [person for person, _, _ in kept]


class TiledPoseDetector:
    def __init__(self, base: Any, *, every: int = 2, cols: int = 2, top: float = 0.6, overlap: float = 0.15) -> None:
        self.base = base
        self.every = max(1, every)
        self.tiles = tiles_for(cols=cols, top=top, overlap=overlap)
        self._calls = 0
        self._tile_people: list[dict[str, Any]] = []

    @property
    def active_device(self) -> str:
        return str(getattr(self.base, "active_device", "auto"))

    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        people = list(await self.base.detect(frame))
        image = frame.get("image")
        if image is not None and self._calls % self.every == 0:
            height, width = image.shape[:2]
            found: list[dict[str, Any]] = []
            for tile in self.tiles:
                x0, y0, x1, y1 = (int(tile[0] * width), int(tile[1] * height),
                                  int(tile[2] * width), int(tile[3] * height))
                crop = {**frame, "image": image[y0:y1, x0:x1], "frame_size": (x1 - x0, y1 - y0)}
                found += [map_from_tile(person, tile) for person in await self.base.detect(crop)]
            # Seated people barely move, so the far rows found here stay valid until the next tile pass.
            self._tile_people = found
        self._calls += 1
        return merge_people(people + self._tile_people)

    def close(self) -> None:
        self.base.close()


__all__ = ["TiledPoseDetector", "map_from_tile", "merge_people", "tiles_for"]
