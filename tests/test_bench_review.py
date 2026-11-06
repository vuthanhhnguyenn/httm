from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
from proctoring_ai.multi.cctv.review_detectors import ModelSpec

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("bench_review", ROOT / "scripts" / "bench_review.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(path: Path, seconds: float = 2.0, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(int(seconds * fps)):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


class _Data:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return []


class _Result:
    names = {0: "cell phone"}

    def __init__(self):
        self.boxes = type("B", (), {"data": _Data()})()
        self.keypoints = None


class _SlowModel:
    """Sleeps 1 ms per image like a real predict would take time."""

    def predict(self, **kwargs):
        images = kwargs["source"]
        time.sleep(0.001 * len(images))
        return [_Result() for _ in images]


def _specs(tmp_path: Path) -> list[ModelSpec]:
    specs = []
    for name, kind in (("fake-pose", "pose"), ("fake-objects", "objects")):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        specs.append(ModelSpec(name, kind, path, {"cell phone": "phone"}, 0.2, None))
    return specs


def test_bench_rows(tmp_path: Path) -> None:
    module = _module()
    video = _video(tmp_path / "clip.mp4")
    rows = module.bench(video, batches=(4, 8), halves=(False, True), devices=("cpu", "0"), image_sizes=(640, 960),
                        crops=8, loader=lambda _path: _SlowModel(), specs=_specs(tmp_path))
    assert len(rows) == 2 * 2 * 2 * 2 * 2
    assert {row["model"] for row in rows} == {"fake-pose", "fake-objects"}
    assert all(row["ms_per_crop"] > 0 for row in rows)
    assert set(rows[0]) >= {"model", "batch", "half", "device", "imgsz", "ms_per_crop"}


def test_cli_writes_bench_json(tmp_path: Path, capsys) -> None:
    module = _module()
    video = _video(tmp_path / "clip.mp4")
    out = tmp_path / "bench.json"
    module.main([str(video), "--out", str(out), "--batches", "4", "--halves", "off", "--devices", "cpu",
                 "--image-sizes", "640", "--crops", "4"], loader=lambda _path: _SlowModel(), specs=_specs(tmp_path))
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert "ms/ảnh" in capsys.readouterr().out
