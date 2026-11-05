from __future__ import annotations

from uuid import uuid4

import numpy as np
from cctv_kps import person
from proctoring_ai.multi.cctv.review_core import analyse_window
from proctoring_ai.multi.cctv.review_rules import review_window
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box


class _Detectors:
    def __init__(self, kps):
        self.kps, self.calls = kps, []

    def pose_many(self, crops, transforms):
        self.calls.append(("pose", len(crops)))
        return [[{"class": "person", "confidence": 0.9, "bbox": {"x": 0.45, "y": 0.3, "width": 0.1, "height": 0.2},
                  "keypoints": self.kps}] for _ in crops]

    def objects_many(self, crops, transforms):
        self.calls.append(("objects", len(crops)))
        return [[] for _ in crops]


def test_analyse_window_one_batch_and_same_records_as_rules():
    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    window = SuspectWindow(seat, 1, (), 1000, 3000, ("LEAN",), box, {seat: box})
    images = [(t, np.zeros((72, 128, 3), np.uint8)) for t in range(0, 3001, 250)]
    detectors = _Detectors(person(0.5, 0.3))
    records, frames = analyse_window(window, images, detectors)
    assert detectors.calls == [("pose", 13), ("objects", 13)]
    assert [f.offset_ms for f in frames] == [t for t, _ in images]
    assert frames[0].poses[seat] == person(0.5, 0.3)
    assert abs(frames[0].frame_aspect - 128 / 72) < 1e-6
    strip = lambda items: [{k: v for k, v in item.items() if k != "id"} for item in items]  # noqa: E731
    assert strip(records) == strip(review_window(window, frames))


def test_analyse_window_no_images():
    seat = uuid4()
    box = Box(0.45, 0.3, 0.1, 0.2)
    window = SuspectWindow(seat, 1, (), 1000, 3000, ("LEAN",), box, {seat: box})
    detectors = _Detectors(person())
    records, frames = analyse_window(window, [], detectors)
    assert frames == [] and detectors.calls == []
    assert [item["action"] for item in records] == ["dismissed"]
