"""Per-subject crops and the mapping of crop-relative results back to the frame.

Analyzers run on an enlarged crop of one candidate, so a small face far from the
camera still has enough pixels. Every analyzer returns coordinates normalized to
the crop; :class:`CropTransform` turns them into coordinates of the full frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from proctoring_core.observations import Box

cv2: Any
try:
    import cv2
except ImportError:  # pragma: no cover - exercised in environments without CV wheels
    cv2 = None

_MIN_PIXELS = 4


@dataclass(frozen=True, slots=True)
class CropTransform:
    """The crop's rectangle in source-frame pixels; independent of any resize."""

    x0: int
    y0: int
    width: int
    height: int
    frame_width: int
    frame_height: int

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        return ((self.x0 + x * self.width) / self.frame_width, (self.y0 + y * self.height) / self.frame_height)

    def map_box(self, box: Mapping[str, float]) -> dict[str, float]:
        x, y = self.map_point(float(box["x"]), float(box["y"]))
        return {"x": x, "y": y, "width": float(box["width"]) * self.width / self.frame_width,
                "height": float(box["height"]) * self.height / self.frame_height}

    def _map_landmarks(self, points: Any) -> list[dict[str, Any]]:
        mapped = []
        for point in points or []:
            x, y = self.map_point(float(point["x"]), float(point["y"]))
            mapped.append({**point, "x": x, "y": y})
        return mapped

    def _map_xy(self, value: Any) -> Any:
        return self.map_point(float(value[0]), float(value[1])) if value is not None else None

    def map_face(self, face: Mapping[str, Any]) -> dict[str, Any]:
        mapped = dict(face)
        if isinstance(face.get("bbox"), Mapping):
            mapped["bbox"] = self.map_box(face["bbox"])
        if isinstance(face.get("keypoints"), Mapping):
            mapped["keypoints"] = {name: self._map_xy(point) for name, point in face["keypoints"].items()}
        if face.get("face_width") is not None:
            mapped["face_width"] = float(face["face_width"]) * self.width / self.frame_width
        mapped["frame_width"], mapped["frame_height"] = self.frame_width, self.frame_height
        return mapped

    def map_hand(self, hand: Mapping[str, Any]) -> dict[str, Any]:
        mapped = dict(hand)
        if isinstance(hand.get("bbox"), Mapping):
            mapped["bbox"] = self.map_box(hand["bbox"])
        mapped["landmarks"] = self._map_landmarks(hand.get("landmarks"))
        for key in ("wrist", "index_tip"):
            if key in hand:
                mapped[key] = self._map_xy(hand[key])
        return mapped

    def map_pose(self, pose: Mapping[str, Any]) -> dict[str, Any]:
        mapped = dict(pose)
        if isinstance(pose.get("bbox"), Mapping):
            mapped["bbox"] = self.map_box(pose["bbox"])
        mapped["landmarks"] = self._map_landmarks(pose.get("landmarks"))
        return mapped


@dataclass(frozen=True, slots=True)
class Crop:
    image: Any
    transform: CropTransform


class SubjectCropper:
    def __init__(self, *, head_fraction: float = 0.45, margin: float = 0.15, min_side: int = 256,
                 body_margin: float = 0.10) -> None:
        self.head_fraction = head_fraction
        self.margin = margin
        self.min_side = min_side
        self.body_margin = body_margin

    def _crop(self, image: Any, left: float, top: float, width: float, height: float) -> Crop | None:
        frame_height, frame_width = image.shape[:2]
        x0, y0 = max(0, round(left)), max(0, round(top))
        x1, y1 = min(frame_width, round(left + width)), min(frame_height, round(top + height))
        if x1 - x0 < _MIN_PIXELS or y1 - y0 < _MIN_PIXELS:
            return None
        region = image[y0:y1, x0:x1]
        scale = self.min_side / min(region.shape[:2])
        if scale > 1.0 and cv2 is not None:
            size = (round(region.shape[1] * scale), round(region.shape[0] * scale))
            region = cv2.resize(region, size, interpolation=cv2.INTER_LINEAR)
        return Crop(np.ascontiguousarray(region),
                    CropTransform(x0, y0, x1 - x0, y1 - y0, int(frame_width), int(frame_height)))

    def crop_head(self, image: Any, box: Box) -> Crop | None:
        frame_height, frame_width = image.shape[:2]
        width, head = box.width * frame_width, box.height * frame_height * self.head_fraction
        margin_x, margin_y = round(width * self.margin), round(round(head) * self.margin)
        return self._crop(image, box.x * frame_width - margin_x, box.y * frame_height - margin_y,
                          width + 2 * margin_x, round(head) + 2 * margin_y)

    def crop_body(self, image: Any, box: Box) -> Crop | None:
        frame_height, frame_width = image.shape[:2]
        width, height = box.width * frame_width, box.height * frame_height
        margin_x, margin_y = round(width * self.body_margin), round(height * self.body_margin)
        return self._crop(image, box.x * frame_width - margin_x, box.y * frame_height - margin_y,
                          width + 2 * margin_x, height + 2 * margin_y)


__all__ = ["Crop", "CropTransform", "SubjectCropper"]
