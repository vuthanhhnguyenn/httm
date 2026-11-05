from __future__ import annotations

import numpy as np

from proctoring_ai.multi.cctv.frame_ring import FrameRing


def img(value: int = 0):
    return np.full((72, 128, 3), value, dtype=np.uint8)


def test_keeps_five_per_second():
    ring = FrameRing(span_ms=20000, fps=5.0)
    for t in range(0, 1000, 33):
        ring.add(t, img())
    assert ring.offsets() == [0, 198, 396, 594, 792, 990]


def test_drops_older_than_span():
    ring = FrameRing(span_ms=20000, fps=5.0)
    for t in range(0, 30001, 200):
        ring.add(t, img())
    assert ring.oldest_ms() == 10000
    assert ring.get(5000) is None


def test_get_nearest_within_period():
    ring = FrameRing(span_ms=20000, fps=5.0)
    ring.add(1000, img(1))
    ring.add(1200, img(2))
    offset, image = ring.get(1150)
    assert offset == 1200 and image[0, 0, 0] == 2
    assert ring.get(1500) is None


def test_jpeg_mode_same_shape():
    ring = FrameRing(fps=5.0, jpeg_quality=90)
    ring.add(0, img(100))
    offset, image = ring.get(0)
    assert image.shape == (72, 128, 3) and abs(int(image[5, 5, 0]) - 100) <= 3
