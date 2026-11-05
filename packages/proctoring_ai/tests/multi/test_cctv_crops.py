from __future__ import annotations

import numpy as np
import pytest
from proctoring_ai.multi.cctv.crops import crop_image, crop_region
from proctoring_core.observations import Box


def test_crop_region_clamped_at_frame_edge() -> None:
    region = crop_region([Box(0.95, 0.9, 0.05, 0.1)])
    assert region.x >= 0 and region.y >= 0
    assert region.x + region.width <= 1.0 + 1e-9 and region.y + region.height <= 1.0 + 1e-9


def test_crop_roundtrip_to_frame() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    region = crop_region([Box(0.5, 0.5, 0.1, 0.1)])
    crop, transform = crop_image(image, region, long_side=960)
    assert max(crop.shape[:2]) == 960
    x, y = transform.to_frame(crop.shape[1] / 2, crop.shape[0] / 2)
    assert x == pytest.approx(0.55, abs=1e-3) and y == pytest.approx(0.55, abs=1e-3)
