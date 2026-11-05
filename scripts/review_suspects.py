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
import numpy as np  # noqa: E402
from proctoring_ai.multi.cctv.review_cache import save_window  # noqa: E402
from proctoring_ai.multi.cctv.review_core import analyse_window  # noqa: E402
from proctoring_ai.multi.cctv.review_dedupe import ReviewDeduper  # noqa: E402
from proctoring_ai.multi.cctv.review_frames import select_offsets  # noqa: E402
from proctoring_ai.multi.cctv.review_rules import (  # noqa: E402
    ReviewFrame,
)
from proctoring_ai.multi.cctv.suspects import SuspectWindow  # noqa: E402
from proctoring_runtime.messages import event_name  # noqa: E402

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


_FONTS = ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial Unicode.ttf")


def put_text(image: Any, text: str, origin: tuple[int, int], size: int = 18,
             colour: tuple[int, int, int] = (255, 255, 0)) -> Any:
    """Vietnamese text on a BGR image (OpenCV's own fonts have no diacritics)."""

    from PIL import Image, ImageDraw, ImageFont

    font: Any = None
    for path in _FONTS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    canvas = Image.fromarray(image[:, :, ::-1].copy())
    draw = ImageDraw.Draw(canvas)
    x, y = origin
    box = draw.textbbox((x, y), text, font=font or ImageFont.load_default())
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour, font=font or ImageFont.load_default())
    return np.asarray(canvas)[:, :, ::-1].copy()


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
            name = {"phone": "điện thoại", "paper": "giấy", "book": "sách/giấy"}.get(obj.label, obj.label)
            canvas = put_text(canvas, f"{name} {obj.confidence:.2f}", (px(box.x, box.y)[0], max(0, px(box.x, box.y)[1] - 18)),
                              size=14, colour=(255, 0, 255))
    return put_text(canvas, f"{item['video_offset_ms'] / 1000:.1f}s  {item.get('label_vi', item['type'])}", (10, 8), size=22)


def review(video: Path, windows: list[SuspectWindow], detectors: Any, out_dir: Path, fps: float = 10.0,
           warmup_ms: int = WARMUP_MS, cache_path: Path | None = None,
           max_frames: int | None = None) -> dict[str, Any]:
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
    dedupe = ReviewDeduper()
    try:
        for window in windows:
            began = time.perf_counter()
            end = window.end_ms if duration_ms is None else min(window.end_ms, duration_ms)
            images = _frames(capture, window.start_ms - warmup_ms, end, fps)
            if max_frames is not None:
                keep = set(select_offsets(window, [t for t, _ in images], max_frames=max_frames, warmup_ms=warmup_ms))
                images = [(t, image) for t, image in images if t in keep]
            found, analysed = analyse_window(window, images, detectors)
            if cache_path is not None:
                save_window(cache_path, window, analysed)
            found = dedupe.admit(found)
            for item in found:
                if "type" in item:
                    item["label_vi"] = event_name(item["type"])
            by_offset = {frame.offset_ms: frame for frame in analysed}
            for item in found:
                # Read the frame again rather than keeping every frame of a long window in memory.
                if item["action"] == "opened" and by_offset:
                    nearest = min(by_offset, key=lambda t: abs(t - item["video_offset_ms"]))
                    capture.set(cv2.CAP_PROP_POS_MSEC, nearest)
                    ok, image = capture.read()
                    if ok:
                        name = f"evidence/{item['id']}.jpg"
                        cv2.imwrite(str(out_dir / name), _draw(image, window, by_offset[nearest], item))
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
              "models": dict(detectors.status),
              "devices": dict(getattr(detectors, "devices", {})),
              "device_reasons": dict(getattr(detectors, "reasons", {}))}
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
    parser.add_argument("--warmup-ms", type=int, default=WARMUP_MS, help="đọc thêm trước mỗi đoạn để có mốc tư thế")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="số khung tối đa mỗi đoạn (0 = mọi khung ở --fps, như phase 2e; đoạn dài mà ít khung "
                             "thì sót sự kiện, xem replay_review.py --split-ms)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="ghi đầu ra model của từng đoạn vào <thư mục>/cache.json.gz để chạy lại luật bằng replay_review.py")
    args = parser.parse_args(argv)
    from proctoring_ai.multi.cctv.review_detectors import ReviewDetectors, load_specs

    windows = [SuspectWindow.from_json(json.loads(line))
               for line in args.suspects.read_text(encoding="utf-8").splitlines() if line.strip()]
    detectors = ReviewDetectors(load_specs(args.models, ROOT / "models"), device=args.device)
    for name, status in detectors.status.items():
        print(f"Model {name}: {status}")
    if not any(spec.kind == "pose" and detectors.status.get(spec.name) == "ok"
               for spec in load_specs(args.models, ROOT / "models")):
        print("Không có model khung xương nào dùng được; tầng kỹ sẽ loại gần hết các đoạn.", file=sys.stderr)
    cache_path = None
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = args.cache_dir / "cache.json.gz"
        cache_path.unlink(missing_ok=True)
    timing = review(args.video, windows, detectors, args.out, fps=args.fps, warmup_ms=args.warmup_ms,
                    cache_path=cache_path, max_frames=args.max_frames or None)
    print(f"Xem {timing['windows']} đoạn, mỗi đoạn {timing['seconds_per_window']['p50']} s (tối đa "
          f"{timing['seconds_per_window']['max']} s); sự kiện: {timing['events_opened']}, loại {timing['dismissed']}")
    print(f"Kết quả: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
