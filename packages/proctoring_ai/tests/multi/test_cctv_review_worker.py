from __future__ import annotations

from uuid import uuid4

import cv2
import numpy as np
from cctv_kps import person
from proctoring_ai.multi.cctv.frame_ring import FrameRing
from proctoring_ai.multi.cctv.review_worker import ReviewWorker, ReviewWorkerConfig
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box

BOX = Box(0.45, 0.3, 0.1, 0.2)


class _Detectors:
    """Seated person with a phone at the wrist: every window gives a PHONE_DETECTED."""

    def __init__(self):
        self.images = 0
        self.status, self.devices, self.reasons = {"fake": "ok"}, {}, {}

    def pose_many(self, crops, transforms):
        self.images += len(crops)
        kps = person(0.5, 0.3, wrist=(0.5, 0.36))
        return [[{"class": "person", "confidence": 0.9, "keypoints": kps}] for _ in crops]

    def objects_many(self, crops, transforms):
        from proctoring_ai.multi.cctv.review_detectors import ReviewObject

        return [[ReviewObject("phone", 0.8, Box(0.495, 0.355, 0.01, 0.01), "fake")] for _ in crops]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def window(start=8000, end=10000):
    seat = uuid4()
    return SuspectWindow(seat, 1, (), start, end, ("OBJECT_HINT",), BOX, {seat: BOX}, None)


def ring_with(start=0, end=20000):
    ring = FrameRing(span_ms=20000, fps=5.0)
    for t in range(start, end, 200):
        ring.add(t, np.zeros((72, 128, 3), np.uint8))
    return ring


def _video(path, seconds=12.0, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (128, 72))
    for _ in range(int(seconds * fps)):
        writer.write(np.zeros((72, 128, 3), np.uint8))
    writer.release()
    return path


def opened(records):
    return [item["type"] for item, _image in records if item["action"] == "opened"]


def test_worker_reviews_from_ring():
    detectors = _Detectors()
    worker = ReviewWorker(detectors, ring_with(), video_path=None)
    worker.submit([window(8000, 10000), window(12000, 14000), window(15000, 17000)])
    records = worker.close(flush=True)
    assert opened(records).count("PHONE_DETECTED") == 3
    assert detectors.images <= 3 * ReviewWorkerConfig().max_frames
    assert all(image is not None for item, image in records if item["action"] == "opened")
    stats = worker.stats()
    assert stats["review_windows"] == 3 and stats["review_skipped"] == 0 and "p95" in stats["review_latency_ms"]


def test_worker_reads_file_when_ring_too_new(tmp_path):
    worker = ReviewWorker(_Detectors(), ring_with(9000, 20000), video_path=_video(tmp_path / "v.avi"))
    worker.submit([window(4000, 6000)])
    assert "PHONE_DETECTED" in opened(worker.close(flush=True))


def test_worker_skips_when_no_file():
    worker = ReviewWorker(_Detectors(), ring_with(9000, 20000), video_path=None)
    worker.submit([window(4000, 6000)])
    records = worker.close(flush=True)
    assert [(item["type"], item["skip_reason"]) for item, _ in records] == [("REVIEW_SKIPPED", "frames_gone")]


def test_deadline_skips_with_record():
    clock = _Clock()
    worker = ReviewWorker(_Detectors(), ring_with(), video_path=None, clock=clock, start=False)
    worker.submit([window()])
    clock.now = 61.0
    worker.step()
    (item, _image), = worker.drain()
    assert item["type"] == "REVIEW_SKIPPED" and item["skip_reason"] == "deadline"


def test_close_flush_leaves_no_window_untracked():
    worker = ReviewWorker(_Detectors(), ring_with(), video_path=None,
                          config=ReviewWorkerConfig(deadline_ms=60000))
    windows = [window(t, t + 2000) for t in (8000, 9000, 12000, 15000)]  # the first two are different seats
    worker.submit(windows)
    records = worker.close(flush=True)
    seats = {item["seat_id"] for item, _ in records}
    assert seats == {str(w.seat_id) for w in windows}


def test_close_without_flush_marks_shutdown():
    worker = ReviewWorker(_Detectors(), ring_with(), video_path=None, start=False)
    worker.submit([window()])
    records = worker.close(flush=False)
    assert [item["skip_reason"] for item, _ in records] == ["shutdown"]


def test_next_piece_of_a_seat_reuses_analysed_frames():
    detectors = _Detectors()
    worker = ReviewWorker(detectors, ring_with(), video_path=None, start=False)
    first = window(8000, 10000)
    second = SuspectWindow(first.seat_id, 1, (), 10000, 12000, ("OBJECT_HINT",), BOX, {first.seat_id: BOX}, None)
    worker.submit([first])
    worker.step()
    before = detectors.images
    worker.submit([second])
    worker.step()
    cfg = ReviewWorkerConfig()
    # No warm-up frames read again: the first piece's frames are the context.
    assert detectors.images - before <= cfg.per_s * 2 + 1
    assert worker.history.before(SuspectWindow(first.seat_id, 1, (), 12000, 14000, ("OBJECT_HINT",), BOX,
                                               {first.seat_id: BOX}, None))


def test_live_density_defaults():
    # Clip 1 replay (phase 2f): 3 frames a second and a dense 6 s warm-up keep 9/9; 12 frames a window found 0/9.
    cfg = ReviewWorkerConfig()
    assert cfg.per_s == 3.0 and cfg.warmup_frames == 18 and cfg.max_frames == 48
