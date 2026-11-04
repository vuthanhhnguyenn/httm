from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/proctoring_ai/tests/multi"))

from cctv_kps import person  # noqa: E402
from proctoring_ai.multi.cctv.review_detectors import ReviewObject  # noqa: E402
from proctoring_ai.multi.cctv.suspects import SuspectWindow  # noqa: E402
from proctoring_core.observations import Box  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location("review_suspects", ROOT / "scripts" / "review_suspects.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(path: Path, seconds: float = 3.0, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    for index in range(int(seconds * fps)):
        image = np.full((360, 640, 3), index % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


class _Detectors:
    def __init__(self, phone: bool = False) -> None:
        self.phone, self.status = phone, {"fake": "ok"}

    def pose(self, _crop, _transform):
        kps = person(cx=0.3, top=0.3)
        kps[9] = [0.3, 0.42, 0.9]
        return [{"class": "person", "confidence": 0.9, "keypoints": kps}]

    def objects(self, _crop, _transform):
        return [ReviewObject("phone", 0.6, Box(0.29, 0.41, 0.02, 0.02), "fake")] if self.phone else []


def _window(start: int, end: int) -> SuspectWindow:
    seat = uuid4()
    box = Box(0.25, 0.3, 0.1, 0.2)
    return SuspectWindow(seat, 1, (), start, end, ("OBJECT_HINT",), box, {seat: box})


def test_empty_suspects_writes_empty_outputs(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    timing = _module().review(video, [], _Detectors(), tmp_path / "out")
    assert (tmp_path / "out" / "events.jsonl").read_text(encoding="utf-8") == ""
    saved = json.loads((tmp_path / "out" / "timing.json").read_text(encoding="utf-8"))
    assert saved["windows"] == 0 and timing["windows"] == 0 and saved["models"] == {"fake": "ok"}


def test_window_past_video_end_is_clamped(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    timing = _module().review(video, [_window(2000, 6000)], _Detectors(), tmp_path / "out")
    assert timing["windows"] == 1
    lines = (tmp_path / "out" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["action"] for line in lines] == ["dismissed"]


def test_phone_window_produces_event_and_evidence(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.avi")
    _module().review(video, [_window(0, 3000)], _Detectors(phone=True), tmp_path / "out")
    records = [json.loads(line) for line in (tmp_path / "out" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    opened = [item for item in records if item["action"] == "opened"]
    assert [item["type"] for item in opened] == ["PHONE_DETECTED"]
    assert (tmp_path / "out" / opened[0]["evidence"]).is_file()
