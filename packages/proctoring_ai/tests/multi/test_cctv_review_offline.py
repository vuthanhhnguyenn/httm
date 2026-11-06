from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from cctv_kps import person
from proctoring_ai.multi.cctv.review_cache import load_cache, window_key
from proctoring_ai.multi.cctv.review_core import analyse_window, review_seats
from proctoring_ai.multi.cctv.review_dedupe import ReviewDeduper
from proctoring_ai.multi.cctv.review_detectors import ReviewObject
from proctoring_ai.multi.cctv.review_offline import plan_frames, review_offline
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_ai.multi.tracker import iou
from proctoring_core.observations import Box

ROOT = Path(__file__).resolve().parents[4]
SIZE = (640, 360)
CENTRES = (0.2, 0.4, 0.6, 0.8)
SEATS = [uuid4() for _ in CENTRES]
BOXES = {seat: Box(cx - 0.05, 0.3, 0.1, 0.2) for seat, cx in zip(SEATS, CENTRES, strict=True)}


def _review_suspects():
    spec = importlib.util.spec_from_file_location("review_suspects_t5", ROOT / "scripts" / "review_suspects.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _square(image, x: float, y: float, colour, half: int = 6) -> None:
    px, py = int(x * SIZE[0]), int(y * SIZE[1])
    image[py - half:py + half, px - half:px + half] = colour


def _video(path: Path, seconds: float = 6.0, fps: float = 10.0, phone_at: tuple[float, float] = (0.4, 0.42)) -> Path:
    """One head marker per seat, moving a little every frame (a wrong frame shows as a wrong head place).
    Seat 0's head is red (bowed, hands in the lap) from 1 s to 5 s; a green phone shows from 2 s to 4 s."""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, SIZE)
    for index in range(int(seconds * fps)):
        image = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        for j, cx in enumerate(CENTRES):
            red = j == 0 and 10 <= index < 50
            _square(image, cx + 0.004 * ((index + j) % 5), 0.38 if red else 0.33, (0, 0, 255) if red else (255, 255, 255))
        if 20 <= index < 40:
            _square(image, *phone_at, (0, 255, 0))
        writer.write(image)
    writer.release()
    return path


def _blobs(crop, mask) -> list[tuple[float, float]]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(float(centroids[i][0]), float(centroids[i][1])) for i in range(1, count) if stats[i][4] >= 6]


class _PixelDetectors:
    """Reads the markers off the crop's pixels and maps them back through the crop's transform."""

    def __init__(self) -> None:
        self.status = {"fake": "ok"}
        self.calls: list[tuple[str, int, str]] = []

    def pose_many(self, crops, transforms):
        self.calls.append(("pose", len(crops), threading.current_thread().name))
        out = []
        for crop, transform in zip(crops, transforms, strict=True):
            b, g, r = (crop[:, :, i].astype(int) for i in range(3))
            people = []
            for mask, bent in (((b > 170) & (g > 170) & (r > 170), False), ((r > 150) & (g < 90) & (b < 90), True)):
                for x, y in _blobs(crop, mask):
                    fx, fy = transform.to_frame(x, y)
                    top = fy - (0.08 if bent else 0.03)
                    kps = person(cx=fx, top=top, bent=bent)
                    kps[9], kps[10] = ([fx - 0.02, top + 0.25, 0.9], [fx + 0.02, top + 0.25, 0.9]) if bent else \
                        ([fx, top + 0.12, 0.9], [fx + 0.03, top + 0.15, 0.9])
                    people.append({"class": "person", "confidence": 0.9, "keypoints": kps})
            out.append(people)
        return out

    def objects_many(self, crops, transforms):
        self.calls.append(("objects", len(crops), threading.current_thread().name))
        out = []
        for crop, transform in zip(crops, transforms, strict=True):
            b, g, r = (crop[:, :, i].astype(int) for i in range(3))
            found = []
            for x, y in _blobs(crop, (g > 170) & (r < 100) & (b < 100)):
                fx, fy = transform.to_frame(x, y)
                found.append(ReviewObject("phone", 0.6, Box(fx - 0.01, fy - 0.01, 0.02, 0.02), "fake"))
            out.append(found)
        return out


def _window(seat, start: int, end: int, reasons=("OBJECT_HINT",)) -> SuspectWindow:
    return SuspectWindow(seat, 1, tuple(key for key in SEATS if key != seat), start, end, tuple(reasons),
                         BOXES[seat], dict(BOXES))


def _strip(records):
    return [{k: v for k, v in item.items() if k not in ("id", "merged_into")} for item in records]


def _same_frames(got, want, tol: float = 1e-4) -> None:
    assert [f.offset_ms for f in got] == [f.offset_ms for f in want]
    for a, b in zip(got, want, strict=True):
        assert set(a.seats) == set(b.seats)
        for key, box in a.seats.items():
            other = b.seats[key]
            assert np.allclose([box.x, box.y, box.width, box.height], [other.x, other.y, other.width, other.height],
                               atol=tol)
        assert abs(a.frame_aspect - b.frame_aspect) <= tol
        assert set(a.poses) == set(b.poses)
        for key in a.poses:
            assert np.allclose(np.asarray(a.poses[key]), np.asarray(b.poses[key]), atol=tol)
        assert [o.label for o in a.objects] == [o.label for o in b.objects]
        for x, y in zip(a.objects, b.objects, strict=True):
            assert np.allclose([x.confidence, x.bbox.x, x.bbox.y, x.bbox.width, x.bbox.height],
                               [y.confidence, y.bbox.x, y.bbox.y, y.bbox.width, y.bbox.height], atol=tol)


def _per_window(video: Path, windows, detectors, fps: float = 5.0, warmup_ms: int = 6000, duration_ms: int = 6000):
    frames_of = _review_suspects()._frames
    capture = cv2.VideoCapture(str(video))
    dedupe, records, frames = ReviewDeduper(), [], {}
    try:
        for window in windows:
            images = frames_of(capture, window.start_ms - warmup_ms, min(window.end_ms, duration_ms), fps)
            found, analysed = analyse_window(window, images, detectors)
            records += dedupe.admit(found)
            frames[window_key(window)] = analysed
    finally:
        capture.release()
    return records, frames


def test_plan_frames_matches_frames_grid(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    windows = [_window(SEATS[0], 1500, 3050), _window(SEATS[1], 2500, 5050)]
    plan = plan_frames(windows, fps=5.0, warmup_ms=500, duration_ms=6000)
    frames_of = _review_suspects()._frames
    capture = cv2.VideoCapture(str(video))
    try:
        for index, window in enumerate(windows):
            want = [t for t, _ in frames_of(capture, window.start_ms - 500, window.end_ms, 5.0)]
            assert sorted(t for t, users in plan.items() if index in users) == want
    finally:
        capture.release()
    assert plan[2000] == [0, 1] and plan[1000] == [0] and plan[5000] == [1]
    assert list(plan) == sorted(plan)
    # Clamped to the clip: no target before 0 or after its end.
    clamped = plan_frames([_window(SEATS[0], 1000, 9000)], fps=5.0, warmup_ms=6000, duration_ms=6000)
    assert min(clamped) == 0 and max(clamped) == 6000


def test_same_records_as_per_window(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    windows = [_window(SEATS[0], 1000, 4000, ("HEAD_DOWN_HANDS_LOW",)),  # starts at 1 s: warm-up cut at 0
               _window(SEATS[3], 2000, 4500),
               _window(SEATS[1], 3000, 6000),  # runs to the end of the clip
               _window(SEATS[1], 2500, 5000)]  # the same phone again: merged
    want_records, want_frames = _per_window(video, windows, _PixelDetectors())
    assert any(item["action"] == "opened" for item in want_records)  # the comparison is not between empty lists
    assert any(item["action"] == "merged" for item in want_records)
    cache = tmp_path / "cache.json.gz"
    result = review_offline(video, windows, _PixelDetectors(), fps=5.0, warmup_ms=6000, cache_path=cache)
    assert _strip(result.records) == _strip(want_records)
    merged = [item for item in result.records if item["action"] == "merged"]
    opened_ids = {item["id"] for item in result.records if item["action"] == "opened"}
    assert all(item["merged_into"] in opened_ids for item in merged)
    assert set(result.frames) == set(want_frames)
    for key, frames in want_frames.items():
        assert frames, key
        _same_frames(result.frames[key], frames)
    cached = load_cache(cache)
    assert set(cached) == set(want_frames)
    for key, frames in cached.items():
        _same_frames(frames, result.frames[key])
    for name in ("crops", "crops_shared", "decode_s", "pose_s", "objects_s", "rules_s"):
        assert name in result.stats
    assert result.stats["crops_shared"] > 0  # seats 0 and 1 have the same three seats: one crop serves both


def test_shared_crop_calls_model_once(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    windows = [_window(SEATS[0], 0, 3050), _window(SEATS[0], 2000, 5050)]
    detectors = _PixelDetectors()
    result = review_offline(video, windows, detectors, fps=5.0, warmup_ms=0, batch_size=4)
    sizes = [n for kind, n, _ in detectors.calls if kind == "pose"]
    assert sum(sizes) == 26 and result.stats["crops"] == 26  # 16 + 16 frames, 6 of them (2.0 s to 3.0 s) shared
    assert result.stats["crops_shared"] == 6
    assert max(sizes) <= 4  # flushed by batch size, the whole clip is never queued
    assert sum(n for kind, n, _ in detectors.calls if kind == "objects") == 26
    first, second = (result.frames[window_key(w)] for w in windows)
    assert len(first) == len(second) == 16
    want_records, want_frames = _per_window(video, windows, _PixelDetectors(), warmup_ms=0)
    for window in windows:
        _same_frames(result.frames[window_key(window)], want_frames[window_key(window)])
    assert _strip(result.records) == _strip(want_records)


def test_shared_crop_keeps_seats_apart(tmp_path: Path) -> None:
    # Seat 1's crop holds seats 0-2, seat 2's holds seats 1-3: one union crop over all four, the phone by seat 3.
    video = _video(tmp_path / "clip.avi", phone_at=(0.86, 0.42))
    one, two = _window(SEATS[1], 0, 4050), _window(SEATS[2], 0, 4050)
    region_one = review_seats(one)
    region_two = review_seats(two)
    assert set(region_one) == {SEATS[0], SEATS[1], SEATS[2]} and set(region_two) == {SEATS[1], SEATS[2], SEATS[3]}
    from proctoring_ai.multi.cctv.crops import crop_region

    overlap = iou(crop_region(list(region_one.values())), crop_region(list(region_two.values())))
    assert 0.5 <= overlap < 0.9
    detectors = _PixelDetectors()
    result = review_offline(video, [one, two], detectors, fps=5.0, warmup_ms=0, share_iou=0.5)
    assert result.stats["crops_shared"] == 21 and result.stats["crops"] == 21
    for window, seats in ((one, (0, 1, 2)), (two, (1, 2, 3))):
        frames = result.frames[window_key(window)]
        assert len(frames) == 21
        for frame in frames:
            assert set(frame.poses) == {SEATS[j] for j in seats}
            assert dict(frame.seats) == review_seats(window)
            for j in seats:
                nose_x = frame.poses[SEATS[j]][0][0]
                assert CENTRES[j] - 0.01 <= nose_x <= CENTRES[j] + 0.03, (j, nose_x)
    # The phone by seat 3 is outside seat 1's own crop: its frames never see it; seat 2's do.
    assert not any(frame.objects for frame in result.frames[window_key(one)])
    assert any(frame.objects for frame in result.frames[window_key(two)])
    # Same people in the same seats as each window's own crop would give (the union crop is less enlarged).
    _records, want_frames = _per_window(video, [one, two], _PixelDetectors(), warmup_ms=0)
    for window in (one, two):
        _same_frames(result.frames[window_key(window)], want_frames[window_key(window)], tol=0.004)


class _DeviceDetectors(_PixelDetectors):
    def __init__(self, pose_device: str, objects_device: str, barrier: threading.Barrier | None) -> None:
        super().__init__()
        self.devices = {"pose-x": pose_device, "objects-x": objects_device}
        self.kinds = {"pose-x": "pose", "objects-x": "objects"}
        self.barrier = barrier

    def pose_many(self, crops, transforms):
        if self.barrier is not None:
            self.barrier.wait()  # breaks (timeout) unless objects_many runs at the same time
        return super().pose_many(crops, transforms)

    def objects_many(self, crops, transforms):
        if self.barrier is not None:
            self.barrier.wait()
        return super().objects_many(crops, transforms)


def test_parallel_two_devices(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi", seconds=2.0)
    windows = [_window(SEATS[0], 0, 2000)]
    want = review_offline(video, windows, _PixelDetectors(), fps=5.0, warmup_ms=0)
    detectors = _DeviceDetectors("cuda:0", "cuda:1", threading.Barrier(2, timeout=5))
    result = review_offline(video, windows, detectors, fps=5.0, warmup_ms=0, parallel=True, batch_size=4)
    threads = {(kind, name) for kind, _, name in detectors.calls}
    pose_threads = {name for kind, name in threads if kind == "pose"}
    object_threads = {name for kind, name in threads if kind == "objects"}
    assert threading.current_thread().name not in pose_threads | object_threads
    assert result.stats["parallel"] is True
    assert _strip(result.records) == _strip(want.records)
    _same_frames(result.frames[window_key(windows[0])], want.frames[window_key(windows[0])])


@pytest.mark.parametrize("devices", [("cuda:0", "cuda:0"), ("0", "cuda:0"), ("cpu", "cpu")])
def test_parallel_one_device_runs_in_turn(tmp_path: Path, devices) -> None:
    video = _video(tmp_path / "clip.avi", seconds=1.0)
    detectors = _DeviceDetectors(*devices, None)
    result = review_offline(video, [_window(SEATS[0], 0, 1000)], detectors, fps=5.0, warmup_ms=0, parallel=True)
    assert {name for _, _, name in detectors.calls} == {threading.current_thread().name}
    assert result.stats["parallel"] is False


def test_progress_about_once_a_second(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    seen: list[tuple[int, int]] = []
    windows = [_window(SEATS[0], 0, 6000)]
    review_offline(video, windows, _PixelDetectors(), fps=5.0, warmup_ms=0,
                   on_progress=lambda done, total: seen.append((done, total)))
    total = len(plan_frames(windows, fps=5.0, warmup_ms=0, duration_ms=6000))
    assert 2 <= len(seen) <= 8 and seen[-1] == (total, total)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_no_windows_and_bad_video(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi", seconds=1.0)
    detectors = _PixelDetectors()
    result = review_offline(video, [], detectors)
    assert result.records == [] and result.frames == {} and detectors.calls == []
    with pytest.raises(RuntimeError):
        review_offline(tmp_path / "missing.avi", [_window(SEATS[0], 0, 1000)], detectors)


def test_window_past_the_end_has_no_frames_and_is_dismissed(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi", seconds=1.0)
    window = _window(SEATS[0], 8000, 9000)
    result = review_offline(video, [window], _PixelDetectors(), fps=5.0, warmup_ms=0)
    assert result.frames[window_key(window)] == []
    assert [item["action"] for item in result.records] == ["dismissed"]
