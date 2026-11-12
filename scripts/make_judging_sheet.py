"""Tờ chấm rộng: ảnh ghép các cảnh báo chưa có nhãn để Nguyên chấm đúng/sai.

    python scripts/make_judging_sheet.py video.mp4 /home/claude/rc/clip1-loose/events.jsonl \
        --labels data/labels/EPf9O7UXRoM-review.yaml --out /mnt/project-files/rules-core/clip1/ --limit 10

Chọn các cảnh báo chưa được nhãn nào phủ (cùng luật phủ như ``score_alerts``), gộp các cảnh báo cùng đầu cách nhau
<= 2 s thành một ô, rồi vẽ mỗi ô: khung hình lúc cảnh báo mở, cắt quanh ghế (rộng gấp 3 ô ghế, thấy tay và bàn),
viền đỏ quanh thí sinh, số ô lớn, dưới ô ghi loại cảnh báo và giờ. Tối đa 12 ô một ảnh ``sheet-N.jpg``;
``judging.yaml`` ghi ``verdict: null`` cho từng ô (Nguyên điền, ``load_verdicts`` bỏ qua mục null).
Ảnh có hình thí sinh: chỉ để trong project, không đưa vào repo; ``judging.yaml`` chỉ có toạ độ.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from proctoring_ai.multi.cctv.alert_score import alert_windows, covered_labels  # noqa: E402
from proctoring_ai.multi.cctv.review_score import Verdict, load_verdicts  # noqa: E402
from proctoring_core.observations import Box  # noqa: E402

from label_sheets import _tile  # type: ignore[import-not-found]  # noqa: E402

MERGE_S = 2.0
MERGE_HEAD = 0.04
PER_SHEET = 12
COLUMNS = 4
TILE_PX = 440
CAPTION_PX = 56
CONTEXT = 3.0  # vùng cắt rộng gấp mấy lần ô ghế


def select_candidates(records: Iterable[dict[str, Any]], verdicts: Sequence[Verdict], *, limit: int = 30,
                      max_head: float = 0.06, pad_s: float = 2.0) -> list[dict[str, Any]]:
    """Cảnh báo chưa nhãn nào phủ, gộp cùng đầu trong 2 s, theo thời gian; quá ``limit`` thì lấy đều theo thời gian."""

    cells: list[dict[str, Any]] = []
    for alert, start, end in sorted(alert_windows(records), key=lambda item: item[1]):
        if covered_labels(alert, start, end, verdicts, max_head=max_head, pad_s=pad_s):
            continue
        head = (float(alert["head"][0]), float(alert["head"][1]))
        for cell in cells:
            if start - cell["last_s"] <= MERGE_S and any(math.dist(head, h) <= MERGE_HEAD for h in cell["heads"]):
                cell["last_s"] = max(cell["last_s"], start)
                if head not in cell["heads"]:
                    cell["heads"].append(head)
                break
        else:
            cells.append({"id": str(alert.get("id")), "type": str(alert.get("type")), "t_s": round(start, 1),
                          "heads": [head], "seat_id": alert.get("seat_id"), "last_s": start})
    if limit and len(cells) > limit:
        picks = sorted({round(i * (len(cells) - 1) / max(1, limit - 1)) for i in range(limit)})
        cells = [cells[i] for i in picks]
    for cell in cells:
        del cell["last_s"]
    return cells


def _seat_boxes(folder: Path) -> dict[str, Box]:
    """Ô ghế theo seat_id từ ``tracks.jsonl`` + ``timeline.json.gz`` cạnh events (không có thì rỗng)."""

    if not (folder / "tracks.jsonl").is_file() or not (folder / "timeline.json.gz").is_file():
        return {}
    from proctoring_ai.multi.cctv.timeline import load_timeline

    windows, _ = load_timeline(folder)
    return {str(w.seat_id): w.bbox for w in windows}


def _region(seat: Box) -> Box:
    cx, cy = seat.center
    width, height = seat.width * CONTEXT, seat.height * CONTEXT
    x0, y0 = min(max(0.0, cx - width / 2), max(0.0, 1 - width)), min(max(0.0, cy - height / 2), max(0.0, 1 - height))
    return Box(x0, y0, min(width, 1.0), min(height, 1.0))


def _cell_image(frame: Any, number: int, cell: dict[str, Any], seat: Box) -> Any:
    tile = _tile(frame, _region(seat), seat, "", TILE_PX)
    cv2.putText(tile, str(number), (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(tile, str(number), (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 3, cv2.LINE_AA)
    caption = np.full((CAPTION_PX, TILE_PX, 3), 32, np.uint8)
    cv2.putText(caption, f"{cell['type']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(caption, f"t = {cell['t_s']:.1f} s", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([tile, caption])


def write_sheets(video: Path, cells: Sequence[dict[str, Any]], seats: dict[str, Box], out: Path) -> list[Path]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"không mở được video {video}")
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for number, cell in enumerate(cells, start=1):
        capture.set(cv2.CAP_PROP_POS_MSEC, cell["t_s"] * 1000)
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360,
                              int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, 3), np.uint8)
        head = cell["heads"][0]
        seat = seats.get(str(cell.get("seat_id"))) or Box(max(0.0, head[0] - 0.05), max(0.0, head[1] - 0.05), 0.1, 0.2)
        images.append(_cell_image(frame, number, cell, seat))
    capture.release()
    blank = np.zeros_like(images[0]) if images else None
    paths = []
    for index in range(0, len(images), PER_SHEET):
        chunk = images[index:index + PER_SHEET]
        chunk += [blank] * (-len(chunk) % COLUMNS)
        rows = [np.hstack(chunk[i:i + COLUMNS]) for i in range(0, len(chunk), COLUMNS)]
        path = out / f"sheet-{index // PER_SHEET + 1}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tờ chấm rộng cho các cảnh báo chưa có nhãn.")
    parser.add_argument("video", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("--labels", type=Path, nargs="*", default=[], help="tệp nhãn đã có (cảnh báo đã phủ thì bỏ)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts: list[Verdict] = []
    for path in args.labels:
        verdicts.extend(load_verdicts(path))
    cells = select_candidates(records, verdicts, limit=args.limit)
    paths = write_sheets(args.video, cells, _seat_boxes(args.events.parent), args.out)
    items = [{"type": c["type"], "t_s": c["t_s"], "heads": [[round(h[0], 3), round(h[1], 3)] for h in c["heads"]],
              "verdict": None} for c in cells]
    lines = [f"# Tờ chấm {args.video.name}: ô số N là mục thứ N; verdict true = cảnh báo đúng, false = nhầm, null = chưa chấm.",
             f"video: {args.video.name}", "verdicts:"]
    lines += [f"  - {json.dumps(item, ensure_ascii=False)}  # ô {i}" for i, item in enumerate(items, start=1)]
    (args.out / "judging.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert yaml.safe_load((args.out / "judging.yaml").read_text(encoding="utf-8"))["verdicts"] is not None
    print(f"{len(cells)} ô, {len(paths)} ảnh trong {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
