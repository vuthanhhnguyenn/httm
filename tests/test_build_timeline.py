"""Phase 3 tier 1 on CPU: ``scripts/build_timeline.py`` with a fake fast tier and fake models."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from proctoring_ai.multi.cctv.review_cache import load_cache

ROOT = Path(__file__).resolve().parents[1]
SIZE, FPS = (320, 180), 10.0
SEAT_A, SEAT_B = uuid4(), uuid4()


def _module():
    spec = importlib.util.spec_from_file_location("build_timeline", ROOT / "scripts" / "build_timeline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(path: Path, seconds: float = 4.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)  # type: ignore[attr-defined]
    for index in range(int(seconds * FPS)):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 60 + index % 7, np.uint8))
    writer.release()
    return path


def _fast(video: Path, fast_dir: Path, tracks_path: Path) -> None:
    fast_dir.mkdir(parents=True, exist_ok=True)
    (fast_dir / "suspects.jsonl").write_text("", encoding="utf-8")
    (fast_dir / "events.jsonl").write_text("", encoding="utf-8")
    with tracks_path.open("w", encoding="utf-8") as handle:
        for t in range(0, 4000, 200):
            handle.write(json.dumps({"t": t, "seats": {str(SEAT_A): [0.2, 0.3, 0.2, 0.4, "candidate"],
                                                       str(SEAT_B): [0.5, 0.3, 0.2, 0.4, "candidate"]}}) + "\n")


class _Detectors:
    def __init__(self) -> None:
        self.status = {"fake-pose": "ok", "fake-objects": "ok"}
        self.kinds = {"fake-pose": "pose", "fake-objects": "objects"}
        self.devices = {name: "cpu" for name in self.kinds}

    def pose_many(self, crops, transforms):
        return [[] for _ in crops]

    def objects_many(self, crops, transforms):
        return [[] for _ in crops]


def test_build_writes_one_timeline_line_per_seat(tmp_path: Path) -> None:
    out = tmp_path / "tl"
    _module().build(_video(tmp_path / "in.mp4"), out, detectors=_Detectors(), fast_runner=_fast)
    cache = load_cache(out / "timeline.json.gz")
    assert len(cache) == 2
    for frames in cache.values():
        assert 18 <= len(frames) <= 22


def test_timing_has_ratio(tmp_path: Path) -> None:
    out = tmp_path / "tl"
    _module().build(_video(tmp_path / "in.mp4"), out, detectors=_Detectors(), fast_runner=_fast)
    timing = json.loads((out / "timeline_timing.json").read_text(encoding="utf-8"))
    assert timing["seats"] == 2 and timing["ratio"] > 0
    for key in ("video_s", "fast_s", "timeline_s", "total_s", "crops", "crops_shared"):
        assert key in timing, key


class _Counting(_Detectors):
    def __init__(self) -> None:
        super().__init__()
        self.crops = 0

    def pose_many(self, crops, transforms):
        self.crops += len(crops)
        return [[] for _ in crops]


def test_chunks_give_the_same_timeline(tmp_path: Path) -> None:
    out = tmp_path / "tl"
    _module().build(_video(tmp_path / "in.mp4"), out, detectors=_Detectors(), fast_runner=_fast, chunk_seats=1)
    assert len(load_cache(out / "timeline.json.gz")) == 2


def test_resume_skips_seats_already_done(tmp_path: Path) -> None:
    import gzip

    module, video, out = _module(), _video(tmp_path / "in.mp4"), tmp_path / "tl"
    module.build(video, out, detectors=_Detectors(), fast_runner=_fast)
    cache = out / "timeline.json.gz"
    with gzip.open(cache, "rt", encoding="utf-8") as handle:
        first = handle.readline()
    with gzip.open(cache, "wt", encoding="utf-8") as handle:  # seat B lost, and a line cut off mid-write
        handle.write(first + '{"key": ["x"')
    detectors = _Counting()
    module.build(video, out, detectors=detectors, fast_runner=_fast, reuse_fast=True, resume=True)
    assert len(load_cache(cache)) == 2
    again = _Counting()
    module.build(video, out, detectors=again, fast_runner=_fast, reuse_fast=True, resume=True)
    assert detectors.crops > 0 and again.crops == 0
