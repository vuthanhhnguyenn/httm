"""The deep tier running live, beside the fast tier, on the same machine (phase 2f).

``ReviewWorker`` takes suspect windows as the fast tier settles them, reviews the most suspicious
first on one background thread (frames from the ``FrameRing``, or the video file when the ring has
already dropped them) and hands back records with the frame for their evidence picture. A window that
waits longer than ``deadline_ms`` becomes ``REVIEW_SKIPPED`` so the invigilator still sees it.
``LiveReview`` plugs the worker into ``run_multi_replay`` as a side task.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import cv2

from .frame_ring import FrameRing
from .review_core import ReviewHistory, analyse_window
from .review_dedupe import ReviewDeduper
from .review_frames import select_offsets
from .review_queue import ReviewQueue, skipped_record
from .suspects import SuspectWindow

Record = tuple[dict[str, Any], Any]  # (event record, frame for its evidence picture or None)


@dataclass(frozen=True, slots=True)
class ReviewWorkerConfig:
    # Clip 1 replayed with these (scripts/replay_review.py --split-ms 10000): 3 frames a second inside the
    # piece and a dense 6 s warm-up keep 9/9 true alerts; 12 frames a window with 2 warm-up frames found 0/9,
    # because a hold of 3 s and the desk line need frames close together.
    max_frames: int = 48
    per_s: float = 3.0
    warmup_frames: int = 18
    warmup_ms: int = 6000
    history_ms: int = 20000  # frames of the previous piece of the same seat the rules see again
    deadline_ms: int = 60000
    dedupe_ms: int = 10000
    max_open_ms: int = 10000


def _file_frames(path: Path, offsets: Sequence[int]) -> list[tuple[int, Any]]:
    capture = cv2.VideoCapture(str(path))
    out: list[tuple[int, Any]] = []
    try:
        for offset in offsets:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0, offset))
            ok, image = capture.read()
            if ok:
                out.append((offset, image))
    finally:
        capture.release()
    return out


def _evidence(image: Any, window: SuspectWindow, item: dict[str, Any]) -> Any:
    picture = image.copy()
    height, width = picture.shape[:2]
    seat = window.seats.get(window.seat_id, window.bbox)
    for key, box in window.seats.items():
        colour = (0, 0, 255) if str(key) == item.get("seat_id") else (0, 200, 255)
        cv2.rectangle(picture, (int(box.x * width), int(box.y * height)),
                      (int((box.x + box.width) * width), int((box.y + box.height) * height)), colour, 2)
    if not window.seats:
        cv2.rectangle(picture, (int(seat.x * width), int(seat.y * height)),
                      (int((seat.x + seat.width) * width), int((seat.y + seat.height) * height)), (0, 0, 255), 2)
    return picture


class ReviewWorker:
    def __init__(self, detectors: Any, ring: FrameRing, *, video_path: Path | None,
                 config: ReviewWorkerConfig | None = None, clock: Callable[[], float] = time.monotonic,
                 start: bool = True) -> None:
        self.detectors, self.ring, self.video_path = detectors, ring, video_path
        self.config = config or ReviewWorkerConfig()
        self.clock = clock
        self.queue = ReviewQueue(self.config.deadline_ms, self.config.max_frames)
        self.dedupe = ReviewDeduper(self.config.dedupe_ms)
        self.history = ReviewHistory(self.config.history_ms)
        self._done: list[Record] = []
        self._latency_ms: list[float] = []
        self._windows = self._skipped = 0
        self._lock = threading.Condition()
        self._closing = False
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._run, name="cctv-review", daemon=True)
            self._thread.start()

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    def submit(self, windows: Sequence[SuspectWindow]) -> None:
        with self._lock:
            for window in windows:
                self.queue.push(window, self._now_ms())
            self._lock.notify()

    def drain(self) -> list[Record]:
        with self._lock:
            out, self._done = self._done, []
        return out

    def _skip(self, window: SuspectWindow, reason: str) -> None:
        self._skipped += 1
        self._done.append((skipped_record(window, reason), self._peak_image(window)))

    def _peak_image(self, window: SuspectWindow) -> Any:
        found = self.ring.get(window.peak_ms if window.peak_ms is not None else window.start_ms)
        return found[1] if found is not None else None

    def step(self) -> bool:
        """Review one window (or expire late ones); False when there was nothing to do."""

        with self._lock:
            for late in self.queue.expired(self._now_ms()):
                self._skip(late, "deadline")
            popped = self.queue.pop_timed()
        if popped is None:
            return False
        waited_from, window = popped  # latency counts the wait in the queue too
        cfg = self.config
        context = self.history.before(window)
        warmup_ms = 0 if context else cfg.warmup_ms  # the previous piece already is the warm-up
        warmup_frames = 0 if context else cfg.warmup_frames
        oldest = self.ring.oldest_ms()
        needs_file = oldest is None or window.start_ms - warmup_ms < oldest - 1000 / 5
        if needs_file and self.video_path is None and (oldest is None or window.start_ms < oldest):
            with self._lock:
                self._skip(window, "frames_gone")
            return True
        if needs_file and self.video_path is not None:
            dense = list(range(max(0, window.start_ms - warmup_ms), window.end_ms + 1, 200))
            offsets = select_offsets(window, dense, per_s=cfg.per_s, warmup_frames=warmup_frames,
                                     warmup_ms=warmup_ms, max_frames=cfg.max_frames)
            images = _file_frames(self.video_path, offsets)
        else:
            offsets = select_offsets(window, self.ring.offsets(), per_s=cfg.per_s, warmup_frames=warmup_frames,
                                     warmup_ms=warmup_ms, max_frames=cfg.max_frames)
            images = [found for t in offsets if (found := self.ring.get(t)) is not None]
        records, frames = analyse_window(window, images, self.detectors, context=context)
        self.history.add(window, frames)
        by_offset = dict(images)
        out: list[Record] = []
        for item in self.dedupe.admit(records):
            image = None
            if item.get("action") == "opened" and by_offset:
                nearest = min(by_offset, key=lambda t: abs(t - int(item.get("video_offset_ms", 0))))
                image = _evidence(by_offset[nearest], window, item)
            out.append((item, image))
        with self._lock:
            self._done += out
            self._windows += 1
            self._latency_ms.append(self._now_ms() - waited_from)
        return True

    def _run(self) -> None:
        while True:
            if self.step():
                continue
            with self._lock:
                if self._closing and not len(self.queue):
                    return
                self._lock.wait(timeout=0.1)

    def close(self, flush: bool = True, timeout_s: float = 600) -> list[Record]:
        with self._lock:
            self._closing = True
            if not flush:
                for window in self.queue.drain_all():
                    self._skip(window, "shutdown")
            self._lock.notify()
        if self._thread is not None:
            self._thread.join(timeout_s)
        elif flush:
            while self.step():
                pass
        with self._lock:
            for window in self.queue.drain_all():  # thread timed out
                self._skip(window, "shutdown")
        return self.drain()

    def stats(self) -> dict[str, Any]:
        latency = sorted(self._latency_ms)

        def pct(q: float) -> float:
            return round(latency[min(len(latency) - 1, int(q * len(latency)))], 1) if latency else 0.0

        return {"review_windows": self._windows, "review_skipped": self._skipped,
                "review_queue_max": self.queue.max_len,
                "review_latency_ms": {"p50": round(median(latency), 1) if latency else 0.0, "p95": pct(0.95),
                                      "max": round(latency[-1], 1) if latency else 0.0}}


class LiveReview:
    """Side task of ``run_multi_replay``: feed the ring, hand settled windows to the worker, return results."""

    def __init__(self, tracker: Any, worker: ReviewWorker, ring: FrameRing, *, max_open_ms: int = 10000) -> None:
        self.tracker, self.worker, self.ring, self.max_open_ms = tracker, worker, ring, max_open_ms
        self.windows: list[SuspectWindow] = []

    def on_frame(self, frame: dict[str, Any], offset_ms: int) -> list[Record]:
        if (image := frame.get("image")) is not None:
            self.ring.add(offset_ms, image)
        settled = self.tracker.poll(offset_ms, max_open_ms=self.max_open_ms)
        self.windows += settled
        self.worker.submit(settled)
        return self.worker.drain()

    def finish(self, offset_ms: int) -> list[Record]:
        rest = self.tracker.finish(None)
        self.windows += rest
        self.worker.submit(rest)
        return self.worker.close(flush=True)

    def summary(self) -> dict[str, Any]:
        stats = self.worker.stats()
        devices = getattr(self.worker.detectors, "devices", {})
        return {**stats, "review_devices": dict(devices),
                "review_device_reasons": dict(getattr(self.worker.detectors, "reasons", {}))}


__all__ = ["LiveReview", "ReviewWorker", "ReviewWorkerConfig"]
