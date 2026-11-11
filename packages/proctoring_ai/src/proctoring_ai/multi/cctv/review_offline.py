"""The deep tier over a whole recorded video in one pass (phase 2g, lượt kỹ).

``scripts/review_suspects.py`` seeks and decodes the video again for every suspect window and runs the
models crop by crop. Here the video is read once, in order: every frame some window's grid wants
(``FrameCursor``, the same rule as the per-window path) is cropped once per crop region and dropped at
once; windows whose regions overlap by ``share_iou`` in the same frame share one crop over the union of
their regions; crops queue up and go through ``pose_many`` / ``objects_many`` ``batch_size`` at a time
(on two threads when the two models sit on two different devices). Each window then takes, from its
crop's results, only what lay inside its own region and assigns poses to its own seats, so the rules see
what ``analyse_window`` would have given them; with no shared crop the results are identical.

No image is kept after the models ran (ruling P1): evidence pictures are read again by seeking the video.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2
from proctoring_core.observations import Box

from ..tracker import iou
from .crops import CropTransform, crop_image, crop_region
from .pose_smooth import merge_flipped
from .review_cache import CacheKey, save_window, window_key
from .review_core import review_seats
from .review_dedupe import ReviewDeduper
from .review_frames import FrameCursor
from .review_rules import ReviewFrame, ReviewRuleConfig, assign_poses, review_window
from .suspects import SuspectWindow

DEFAULT_BATCH = 16


@dataclass(slots=True)
class OfflineResult:
    """``frames``: each window's analysed frames (keypoints and boxes in full-frame coordinates, no images).
    ``stats``: ``crops`` sent to the models, ``crops_shared`` (window frames served by a crop made for another
    window), ``frames`` decoded and used, seconds spent in ``decode_s``, ``pose_s``, ``objects_s``, ``rules_s``,
    and ``parallel`` (whether the two models really ran side by side)."""

    records: list[dict[str, Any]]
    frames: dict[CacheKey, list[ReviewFrame]]
    stats: dict[str, Any] = field(default_factory=dict)


def _end_ms(window: SuspectWindow, duration_ms: int | None) -> int:
    return window.end_ms if duration_ms is None else min(window.end_ms, duration_ms)


def plan_frames(windows: Sequence[SuspectWindow], *, fps: float = 5.0, warmup_ms: int = 6000,
                duration_ms: int | None = None) -> dict[int, list[int]]:
    """Target offset (ms) -> indices of the windows that want it, sorted by offset: each window's grid from
    ``start - warmup`` (not before 0) to its end (not past ``duration_ms``) every ``1000 / fps`` ms, as
    ``FrameCursor`` takes it on a video with a frame at every target."""

    plan: dict[int, list[int]] = {}
    for index, window in enumerate(windows):
        cursor = FrameCursor(window.start_ms - warmup_ms, _end_ms(window, duration_ms), fps)
        while not cursor.done:
            at = cursor.target
            cursor.take(at)
            plan.setdefault(int(round(at)), []).append(index)
    return dict(sorted(plan.items()))


def _device_key(device: str | None) -> str:
    name = "default" if device is None else str(device)
    if name.isdigit():
        return f"cuda:{name}"
    return "cuda:0" if name == "cuda" else name


def _two_devices(detectors: Any) -> bool:
    """True when every pose model sits on another device than every object model."""

    kinds: Mapping[str, str] = getattr(detectors, "kinds", None) or {}
    devices: Mapping[str, str | None] = getattr(detectors, "devices", None) or {}
    pose = {_device_key(devices.get(name)) for name, kind in kinds.items() if kind == "pose"}
    objects = {_device_key(devices.get(name)) for name, kind in kinds.items() if kind == "objects"}
    return bool(pose) and bool(objects) and pose.isdisjoint(objects)


class _Models:
    def __init__(self, detectors: Any, parallel: bool, pose_flip: bool = False) -> None:
        self.detectors = detectors
        self.pose_flip = pose_flip
        self.parallel = parallel and _two_devices(detectors)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="review") if self.parallel else None
        self.pose_s = self.objects_s = 0.0

    def _pose(self, crops: list[Any], transforms: list[CropTransform]) -> list[list[dict[str, Any]]]:
        began = time.perf_counter()
        try:
            people = list(self.detectors.pose_many(crops, transforms))
            if not self.pose_flip:
                return people
            # Test-time flip: the same crops mirrored, each person mirrored back around the crop's centre line.
            flipped = list(self.detectors.pose_many([cv2.flip(crop, 1) for crop in crops], transforms))
            return [merge_flipped(list(found), list(mirror), transform.to_frame(crop.shape[1] / 2, 0)[0])
                    for found, mirror, crop, transform in zip(people, flipped, crops, transforms, strict=True)]
        finally:
            self.pose_s += time.perf_counter() - began

    def _objects(self, crops: list[Any], transforms: list[CropTransform]) -> list[list[Any]]:
        began = time.perf_counter()
        try:
            return list(self.detectors.objects_many(crops, transforms))
        finally:
            self.objects_s += time.perf_counter() - began

    def run(self, crops: list[Any], transforms: list[CropTransform]) -> tuple[list[list[dict[str, Any]]], list[list[Any]]]:
        if self._pool is None:
            return self._pose(crops, transforms), self._objects(crops, transforms)
        people = self._pool.submit(self._pose, crops, transforms)
        objects = self._pool.submit(self._objects, crops, transforms)
        return people.result(), objects.result()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)


@dataclass(slots=True)
class _Crop:
    image: Any
    transform: CropTransform
    region: Box
    offset_ms: int
    aspect: float
    members: list[int]  # windows served by this crop


def _union(a: Box, b: Box) -> Box:
    x0, y0 = min(a.x, b.x), min(a.y, b.y)
    x1, y1 = max(a.x + a.width, b.x + b.width), max(a.y + a.height, b.y + b.height)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _inside(point: tuple[float, float], box: Box) -> bool:
    return box.x <= point[0] <= box.x + box.width and box.y <= point[1] <= box.y + box.height


def _group(wanted: Sequence[int], regions: Sequence[Box], share_iou: float) -> list[tuple[Box, list[int]]]:
    """Windows wanting one frame, gathered into crops: a window joins the crop it overlaps most, if by
    ``share_iou`` or more, and the crop grows to the union of both regions."""

    groups: list[tuple[Box, list[int]]] = []
    for index in wanted:
        region = regions[index]
        best, best_iou = -1, 0.0
        for at, (other, _members) in enumerate(groups):
            if (value := iou(other, region)) > best_iou:
                best, best_iou = at, value
        if best >= 0 and best_iou >= share_iou:
            other, members = groups[best]
            groups[best] = (other if other == region else _union(other, region), [*members, index])
        else:
            groups.append((region, [index]))
    return groups


def review_offline(video: Path, windows: list[SuspectWindow], detectors: Any, *, fps: float = 5.0,
                   warmup_ms: int = 6000, share_iou: float = 0.7, cache_path: Path | None = None,
                   parallel: bool = False, on_progress: Callable[[int, int], None] | None = None,
                   batch_size: int | None = None, config: ReviewRuleConfig | None = None,
                   pose_flip: bool = False) -> OfflineResult:
    """Every window of ``windows`` reviewed over one sequential read of ``video``.

    ``records`` come out in the order of ``windows`` through one ``ReviewDeduper``, as the per-window path
    gives them (no ``label_vi``/evidence: the caller adds those). ``cache_path`` gets one ``save_window`` line
    per window (appended, like ``review_suspects``). ``batch_size`` (default: the detectors' own, else 16) is
    how many crops queue up before the models run; ``on_progress(done, total)`` is called about once per
    second of video read, and once at the end with ``done == total``. ``pose_flip``: the pose model also runs
    on every crop mirrored, and the two results are merged (``merge_flipped``); about twice the pose time."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"không mở được video {video}")
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    rate = capture.get(cv2.CAP_PROP_FPS) or 0
    duration_ms = int(1000 * count / rate) if count and rate else None
    total = len(plan_frames(windows, fps=fps, warmup_ms=warmup_ms, duration_ms=duration_ms))
    seats: list[dict[UUID, Box]] = [review_seats(window) for window in windows]
    regions = [crop_region(list(window_seats.values())) for window_seats in seats]
    cursors = [FrameCursor(window.start_ms - warmup_ms, _end_ms(window, duration_ms), fps) for window in windows]
    frames: list[list[ReviewFrame]] = [[] for _ in windows]
    limit = max(1, int(batch_size or getattr(detectors, "batch_size", 0) or DEFAULT_BATCH))
    by_body = (config or ReviewRuleConfig()).assign_by_body
    models = _Models(detectors, parallel, pose_flip)
    queue: list[_Crop] = []
    stats: dict[str, Any] = {"crops": 0, "crops_shared": 0, "frames": 0, "decode_s": 0.0}

    def flush() -> None:
        if not queue:
            return
        people, things = models.run([item.image for item in queue], [item.transform for item in queue])
        for item, found, objects in zip(queue, people, things, strict=True):
            for index in item.members:
                # Poses go only to this window's own seats (every seat box lies inside its own region); of a
                # union crop's objects it keeps those its own crop would have held.
                own = regions[index]
                mine_objects = list(objects) if item.region == own else \
                    [obj for obj in objects if _inside(obj.bbox.center, own)]
                frames[index].append(ReviewFrame(item.offset_ms, seats[index], assign_poses(seats[index], found, by_body=by_body),
                                                 mine_objects, item.aspect))
        queue.clear()

    pending = [index for index, cursor in enumerate(cursors) if not cursor.done]
    reported = None
    try:
        while pending:
            began = time.perf_counter()
            if not capture.grab():
                break
            position = capture.get(cv2.CAP_PROP_POS_MSEC)
            wanted = [index for index in pending if cursors[index].take(position)]
            if wanted:
                ok, image = capture.retrieve()
                if not ok:
                    break
                offset = int(round(position))
                height, width = image.shape[:2]
                groups = _group(wanted, regions, share_iou)
                for region, members in groups:
                    crop, transform = crop_image(image, region)
                    queue.append(_Crop(crop, transform, region, offset, width / height, members))
                del image  # nothing keeps the full frame
                stats["frames"] += 1
                stats["crops"] += len(groups)
                stats["crops_shared"] += len(wanted) - len(groups)
            pending = [index for index in pending if not cursors[index].done]
            stats["decode_s"] += time.perf_counter() - began
            if len(queue) >= limit:
                flush()
            if on_progress is not None and (reported is None or position - reported >= 1000):
                reported = position
                on_progress(min(stats["frames"], total), total)
        flush()
    finally:
        capture.release()
        models.close()
    if on_progress is not None:
        on_progress(total, total)

    began = time.perf_counter()
    dedupe = ReviewDeduper()
    records: list[dict[str, Any]] = []
    for window, analysed in zip(windows, frames, strict=True):
        if cache_path is not None:
            save_window(cache_path, window, analysed)
        records += dedupe.admit(review_window(window, analysed, config))
    stats.update(pose_s=models.pose_s, objects_s=models.objects_s, rules_s=time.perf_counter() - began,
                 parallel=models.parallel, pose_flip=pose_flip)
    for name in ("decode_s", "pose_s", "objects_s", "rules_s"):
        stats[name] = round(stats[name], 3)
    return OfflineResult(records, {window_key(window): analysed for window, analysed in zip(windows, frames, strict=True)},
                         stats)


__all__ = ["OfflineResult", "plan_frames", "review_offline"]
