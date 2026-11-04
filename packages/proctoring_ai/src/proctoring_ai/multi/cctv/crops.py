"""Enlarged crops for the deep tier and the way back to full-frame coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
from proctoring_core.observations import Box


@dataclass(frozen=True, slots=True)
class CropTransform:
    x0: int
    y0: int
    scale: float
    frame_size: tuple[int, int]  # (width, height) of the full frame

    def to_frame(self, x_px: float, y_px: float) -> tuple[float, float]:
        """A pixel of the enlarged crop as normalised full-frame coordinates."""

        width, height = self.frame_size
        return (self.x0 + x_px / self.scale) / width, (self.y0 + y_px / self.scale) / height


def crop_region(boxes: Sequence[Box], margin: float = 0.3) -> Box:
    """Box around all ``boxes`` grown by ``margin`` of its size on each side, kept inside the frame."""

    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x + box.width for box in boxes)
    y1 = max(box.y + box.height for box in boxes)
    dx, dy = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0, x1, y1 = max(0.0, x0 - dx), max(0.0, y0 - dy), min(1.0, x1 + dx), min(1.0, y1 + dy)
    return Box(x0, y0, x1 - x0, y1 - y0)


def crop_image(image: Any, region: Box, long_side: int = 960) -> tuple[Any, CropTransform]:
    height, width = image.shape[:2]
    x0, y0 = int(region.x * width), int(region.y * height)
    x1 = max(x0 + 1, min(width, round((region.x + region.width) * width)))
    y1 = max(y0 + 1, min(height, round((region.y + region.height) * height)))
    crop = image[y0:y1, x0:x1]
    scale = long_side / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, CropTransform(x0, y0, scale, (width, height))


__all__ = ["CropTransform", "crop_image", "crop_region"]
