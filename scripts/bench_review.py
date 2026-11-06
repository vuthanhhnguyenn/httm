"""Đo tốc độ tầng kỹ (ms mỗi ảnh cắt) theo batch, FP16, card GPU và kích thước ảnh vào.

    python scripts/bench_review.py video.mp4 --out bench.json

Mỗi cấu hình chạy 1 lần khởi động rồi đo 3 lần, lấy trung vị. Mỗi dòng: ``model, batch, half, device, imgsz,
ms_per_crop`` (``ran_on`` là nơi model thật sự chạy, khác ``device`` khi máy không có card đó).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from proctoring_ai.multi.cctv.crops import CropTransform  # noqa: E402
from proctoring_ai.multi.cctv.review_detectors import (  # noqa: E402
    ModelSpec,
    ReviewDetectors,
    load_specs,
)

CROP_SIZE = 960
REPEATS = 3
SAMPLE_FRAMES = 32


def sample_crops(video: Path, count: int, seed: int = 0) -> list[Any]:
    """``count`` random square regions of frames spread over the video, each resized to 960x960."""

    capture = cv2.VideoCapture(str(video))
    frames: list[Any] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or SAMPLE_FRAMES
        wanted = set(np.linspace(0, max(0, total - 1), min(SAMPLE_FRAMES, max(1, total))).astype(int).tolist())
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index in wanted:
                frames.append(image)
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Không đọc được khung nào từ {video}")
    rng = np.random.default_rng(seed)
    crops: list[Any] = []
    for _ in range(count):
        frame = frames[int(rng.integers(len(frames)))]
        height, width = frame.shape[:2]
        side = int(min(height, width) * rng.uniform(0.5, 1.0))
        x0, y0 = int(rng.integers(0, width - side + 1)), int(rng.integers(0, height - side + 1))
        crops.append(cv2.resize(frame[y0:y0 + side, x0:x0 + side], (CROP_SIZE, CROP_SIZE)))
    return crops


def _run(detectors: ReviewDetectors, spec: ModelSpec, crops: list[Any], transforms: list[CropTransform]) -> None:
    if spec.kind == "pose":
        detectors.pose_many(crops, transforms)
    else:
        detectors.objects_many(crops, transforms)


def bench(video: Path, *, batches: Sequence[int], halves: Sequence[bool], devices: Sequence[str],
          image_sizes: Sequence[int], crops: int = 64, loader: Callable[[Path], Any] | None = None,
          specs: list[ModelSpec] | None = None) -> list[dict[str, Any]]:
    images = sample_crops(video, crops)
    transforms = [CropTransform(0, 0, 1.0, (CROP_SIZE, CROP_SIZE)) for _ in images]
    chosen = specs if specs is not None else load_specs(ROOT / "configs" / "review_models.yaml", ROOT / "models")
    rows: list[dict[str, Any]] = []
    for spec, batch, half, device, imgsz in itertools.product(chosen, batches, halves, devices, image_sizes):
        row: dict[str, Any] = {"model": spec.name, "batch": batch, "half": half, "device": device, "imgsz": imgsz}
        try:
            detectors = ReviewDetectors([spec], device=device, loader=loader, image_size=imgsz, batch_size=batch,
                                        half=half, device_override={spec.name: device})
            if detectors.status.get(spec.name) != "ok":
                print(f"Bỏ qua {spec.name}: {detectors.status.get(spec.name)}", file=sys.stderr)
                continue
            _run(detectors, spec, images, transforms)  # warm-up: CUDA init, cuDNN autotune, first allocation
            samples: list[float] = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                _run(detectors, spec, images, transforms)
                samples.append((time.perf_counter() - start) * 1000.0 / len(images))
            row |= {"ran_on": detectors.devices.get(spec.name), "ms_per_crop": round(median(samples), 3)}
        except Exception as exc:  # e.g. CUDA out of memory at batch 32: keep the row, go on with the rest
            row |= {"ran_on": None, "ms_per_crop": None, "error": str(exc)}
        rows.append(row)
    return rows


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    header = f"{'model':<22}{'batch':>6}{'half':>6}{'device':>8}{'imgsz':>7}{'chạy trên':>11}{'ms/ảnh':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(f"{row['model']:<22}{row['batch']:>6}{'on' if row['half'] else 'off':>6}{row['device']:>8}"
                     f"{row['imgsz']:>7}{str(row['ran_on']):>11}"
                     + (f"{row['ms_per_crop']:>9.1f}" if row["ms_per_crop"] is not None
                        else f"{'LỖI':>9}  {row.get('error', '')}"))
    return "\n".join(lines)


def _halves(text: str) -> list[bool]:
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part not in ("on", "off") for part in parts):
        raise argparse.ArgumentTypeError(f"--halves chỉ nhận on hoặc off (cách nhau bằng dấu phẩy), không phải {text!r}")
    return [part == "on" for part in parts]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None, *, loader: Callable[[Path], Any] | None = None,
         specs: list[ModelSpec] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("bench.json"))
    parser.add_argument("--batches", default="8,16,32", help="các cỡ batch, cách nhau bằng dấu phẩy")
    parser.add_argument("--halves", type=_halves, default="off,on", help="FP16: off, on hoặc off,on")
    parser.add_argument("--devices", default="0,1", help="cpu hoặc chỉ số card, vd. 0,1")
    parser.add_argument("--image-sizes", default="960,640")
    parser.add_argument("--crops", type=int, default=64)
    args = parser.parse_args(argv)
    rows = bench(args.video, batches=_ints(args.batches), halves=args.halves,
                 devices=[part.strip() for part in args.devices.split(",")], image_sizes=_ints(args.image_sizes),
                 crops=args.crops, loader=loader, specs=specs)
    print(format_table(rows))
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
