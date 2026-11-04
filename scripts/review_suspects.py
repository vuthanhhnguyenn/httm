"""Tầng kỹ CCTV (phase 2e): xem lại từng đoạn nghi vấn bằng model train sẵn lớn hơn trên ảnh cắt phóng to.

    python scripts/review_suspects.py video.mp4 storage/multi/p2e-fast/suspects.jsonl --out storage/multi/p2e-review

Ghi ``events.jsonl`` (sự kiện tầng kỹ và các đoạn bị loại ``dismissed``), ``evidence/`` và ``timing.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

import cv2  # noqa: E402

from proctoring_ai.multi.cctv.crops import crop_image, crop_region  # noqa: E402
from proctoring_ai.multi.cctv.review_rules import ReviewFrame, assign_poses, review_window  # noqa: E402
from proctoring_ai.multi.cctv.suspects import SuspectWindow  # noqa: E402

WARMUP_MS = 8000  # head baselines need a few seconds of usual posture before the window


def _frames(capture: Any, start_ms: int, end_ms: int, fps: float) -> list[tuple[int, Any]]:
    step = 1000.0 / fps
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0, start_ms))
    out: list[tuple[int, Any]] = []
    target = float(max(0, start_ms))
    while target <= end_ms:
        ok, image = capture.read()
        if not ok:
            break
        position = capture.get(cv2.CAP_PROP_POS_MSEC)
        if position + 1e-6 < target:
            continue
        out.append((int(round(position)), image))
        target = position + step
    return out


def _draw(image: Any, window: SuspectWindow, frame: ReviewFrame | None, item: dict[str, Any]) -> Any:
    canvas = image.copy()
    height, width = canvas.shape[:2]

    def px(x: float, y: float) -> tuple[int, int]:
        return int(x * width), int(y * height)

    for seat_id, box in window.seats.items():
        colour = (0, 0, 255) if str(seat_id) == item["seat_id"] else (0, 200, 255)
        cv2.rectangle(canvas, px(box.x, box.y), px(box.x + box.width, box.y + box.height), colour, 2)
    if frame is not None:
        for kps in frame.poses.values():
            for x, y, conf in kps:
                if conf >= 0.3:
                    cv2.circle(canvas, px(x, y), 3, (0, 255, 0), -1)
        for obj in frame.objects:
            box = obj.bbox
            cv2.rectangle(canvas, px(box.x, box.y), px(box.x + box.width, box.y + box.height), (255, 0, 255), 2)
            cv2.putText(canvas, f"{obj.label} {obj.confidence:.2f}", px(box.x, box.y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1)
    cv2.putText(canvas, item["type"], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return canvas


def review(video: Path, windows: list[SuspectWindow], detectors: Any, out_dir: Path, fps: float = 10.0,
           warmup_ms: int = WARMUP_MS) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence").mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"không mở được video {video}")
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    rate = capture.get(cv2.CAP_PROP_FPS) or 0
    duration_ms = int(1000 * count / rate) if count and rate else None
    records: list[dict[str, Any]] = []
    seconds: list[float] = []
    try:
        for window in windows:
            began = time.perf_counter()
            end = window.end_ms if duration_ms is None else min(window.end_ms, duration_ms)
            seats = dict(window.seats) or {window.seat_id: window.bbox}
            region = crop_region(list(seats.values()))
            analysed: list[ReviewFrame] = []
            images: dict[int, Any] = {}
            for offset, image in _frames(capture, window.start_ms - warmup_ms, end, fps):
                crop, transform = crop_image(image, region)
                height, width = image.shape[:2]
                people = detectors.pose(crop, transform)
                analysed.append(ReviewFrame(offset, seats, assign_poses(seats, people),
                                            detectors.objects(crop, transform), width / height))
                images[offset] = image
            found = review_window(window, analysed)
            by_offset = {frame.offset_ms: frame for frame in analysed}
            for item in found:
                if item["action"] == "opened" and images:
                    nearest = min(images, key=lambda t: abs(t - item["video_offset_ms"]))
                    name = f"evidence/{item['id']}.jpg"
                    cv2.imwrite(str(out_dir / name), _draw(images[nearest], window, by_offset.get(nearest), item))
                    item["evidence"] = name
            records += found
            seconds.append(time.perf_counter() - began)
    finally:
        capture.release()
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    timing = {"windows": len(windows), "fps": fps, "warmup_ms": warmup_ms,
              "seconds_per_window": {"p50": round(median(seconds), 2) if seconds else 0.0,
                                     "max": round(max(seconds), 2) if seconds else 0.0},
              "events_opened": dict(Counter(item["type"] for item in records if item["action"] == "opened")),
              "dismissed": sum(1 for item in records if item["action"] == "dismissed"),
              "models": dict(detectors.status)}
    (out_dir / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    return timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("suspects", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "configs" / "review_models.yaml")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default=None, help="vd. 0 cho GPU NVIDIA đầu tiên")
    args = parser.parse_args(argv)
    from proctoring_ai.multi.cctv.review_detectors import ReviewDetectors, load_specs

    windows = [SuspectWindow.from_json(json.loads(line))
               for line in args.suspects.read_text(encoding="utf-8").splitlines() if line.strip()]
    detectors = ReviewDetectors(load_specs(args.models, ROOT / "models"), device=args.device)
    for name, status in detectors.status.items():
        print(f"Model {name}: {status}")
    timing = review(args.video, windows, detectors, args.out, fps=args.fps)
    print(f"Xem {timing['windows']} đoạn, mỗi đoạn {timing['seconds_per_window']['p50']} s (tối đa "
          f"{timing['seconds_per_window']['max']} s); sự kiện: {timing['events_opened']}, loại {timing['dismissed']}")
    print(f"Kết quả: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
