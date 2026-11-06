"""Ảnh ghép để gán nhãn lõi học (phase 3).

    python scripts/label_sheets.py video.mp4 timeline/lop --out timeline/lop/sheets

Đọc thư mục của ``build_timeline.py``, tính đặc trưng cửa sổ 3 s (ghi ``features.csv`` vào đó), chọn các đoạn
cần gán nhãn (luật cũ kích hoạt, đặc trưng nổi bật, và 20 % đoạn yên lặng ngẫu nhiên) rồi xuất mỗi đoạn một
ảnh 2×3 khung quanh ghế, viền đỏ là ghế cần gán. ``index.jsonl`` ghi đoạn của từng ảnh.
Ảnh có hình thí sinh: chỉ để trong project hoặc Kaggle Private, không đưa vào repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from proctoring_ai.multi.cctv.crops import crop_image, crop_region  # noqa: E402
from proctoring_ai.multi.cctv.features import video_features, write_csv  # noqa: E402
from proctoring_ai.multi.cctv.labels import Segment, select_segments  # noqa: E402
from proctoring_ai.multi.cctv.timeline import load_timeline  # noqa: E402
from proctoring_core.observations import Box  # noqa: E402

TILE = 360  # long side of one of the six tiles, pixels


def _tile(image: Any, region: Box, seat: Box, label: str) -> Any:
    crop, transform = crop_image(image, region, long_side=TILE)
    width, height = transform.frame_size
    x0 = int((seat.x * width - transform.x0) * transform.scale)
    y0 = int((seat.y * height - transform.y0) * transform.scale)
    x1 = int(((seat.x + seat.width) * width - transform.x0) * transform.scale)
    y1 = int(((seat.y + seat.height) * height - transform.y0) * transform.scale)
    cv2.rectangle(crop, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    canvas = np.zeros((TILE, TILE, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop[:TILE, :TILE]
    return canvas


def write_sheets(video: Path, segments: Sequence[Segment], boxes: Mapping[UUID, Sequence[Box]], out: Path) -> int:
    """One 2×3 picture per segment (``out/<n:04d>.jpg``) and ``out/index.jsonl``; returns how many."""

    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"không mở được video {video}")
    written = 0
    with (out / "index.jsonl").open("w", encoding="utf-8") as index:
        for n, seg in enumerate(segments):
            region = crop_region(list(boxes.get(seg.seat_id) or [seg.seat_box]))
            tiles = []
            for k in range(6):
                t = seg.start_ms + (seg.end_ms - seg.start_ms) * k / 5
                capture.set(cv2.CAP_PROP_POS_MSEC, t)
                ok, image = capture.read()
                if not ok:
                    image = np.zeros((int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360,
                                      int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, 3), np.uint8)
                tiles.append(_tile(image, region, seg.seat_box, f"#{n} t={t / 1000:.1f}s"))
            sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
            cv2.imwrite(str(out / f"{n:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b = seg.seat_box
            index.write(json.dumps({"n": n, "video": seg.video, "seat_id": str(seg.seat_id),
                                    "seat_box": [round(b.x, 6), round(b.y, 6), round(b.width, 6), round(b.height, 6)],
                                    "start_ms": seg.start_ms, "end_ms": seg.end_ms, "why": seg.why}) + "\n")
            written += 1
    capture.release()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ảnh ghép để gán nhãn lõi học.")
    parser.add_argument("video", type=Path)
    parser.add_argument("timeline", type=Path, help="thư mục ra của build_timeline.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    windows, frames = load_timeline(args.timeline)
    rows = video_features(args.video.stem, windows, frames)
    write_csv(args.timeline / "features.csv", rows)
    segments = select_segments(rows, quiet_fraction=args.quiet_fraction, seed=args.seed)
    boxes = {w.seat_id: list(w.seats.values()) for w in windows}
    count = write_sheets(args.video, segments, boxes, args.out)
    print(f"{len(rows)} cửa sổ, {count} ảnh ghép trong {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
