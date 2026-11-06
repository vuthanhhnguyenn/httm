"""Tầng 1 của lõi học (phase 3): mọi ghế, suốt cả video, qua model lớn.

    python scripts/build_timeline.py video.mp4 --out timeline/lop [--quality max|balanced] [--device 0] [--device2 1]

Tầng nhanh chỉ còn dựng bản đồ ghế (``tracks.jsonl``). Mỗi ghế là một đoạn phủ cả video, đi qua
``review_offline`` (một lần giải mã, ảnh cắt phóng to, 5 khung/giây). Ghi ``timeline.json.gz`` (định dạng
``review_cache``, mỗi ghế một dòng) và ``timeline_timing.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

from proctoring_ai.multi.cctv.review_offline import review_offline  # noqa: E402
from proctoring_ai.multi.cctv.timeline import all_seat_windows, load_tracks  # noqa: E402

TIMELINE = "timeline.json.gz"
TIMING = "timeline_timing.json"


@cache
def _process_video() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bt_process_video", Path(__file__).with_name("process_video.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timeline_pass(video: Path, out: Path, video_s: float, detectors: Any, *, share_iou: float,
                  batch: int | None = None) -> tuple[list[Any], dict[str, Any]]:
    """All-seat windows from ``out/tracks.jsonl`` through ``review_offline``; the cache goes to
    ``out/timeline.json.gz``. Returns the windows and the run's stats."""

    pv = _process_video()
    windows = all_seat_windows(load_tracks(out / "tracks.jsonl"), int(video_s * 1000))
    cache_path = out / TIMELINE
    cache_path.unlink(missing_ok=True)  # save_window appends
    if not windows:
        return windows, {"seats": 0}

    def progress(done: int, total: int) -> None:
        print(f"\rDòng thời gian: {done}/{total} khung", end="\n" if done >= total else "", flush=True)

    result = review_offline(video, windows, detectors, fps=pv.REVIEW_FPS, warmup_ms=0, share_iou=share_iou,
                            cache_path=cache_path, parallel=True, batch_size=batch, on_progress=progress)
    return windows, {"seats": len(windows), **result.stats}


def build(video: Path, out: Path, *, quality: str = "max", device: str | None = None, device2: str | None = None,
          detectors: Any = None, fast_runner: Any = None, reuse_fast: bool = False, half: bool = True,
          batch: int | None = None) -> dict[str, Any]:
    pv = _process_video()
    started = time.perf_counter()
    video_s = pv._probe(video)
    out.mkdir(parents=True, exist_ok=True)
    fast_dir, tracks_path = out / "fast", out / "tracks.jsonl"
    began = time.perf_counter()
    if not (reuse_fast and tracks_path.is_file()):
        shutil.rmtree(fast_dir, ignore_errors=True)
        tracks_path.unlink(missing_ok=True)
        (fast_runner or pv.run_fast)(video, fast_dir, tracks_path)
    fast_s = time.perf_counter() - began
    began = time.perf_counter()
    if detectors is None:
        detectors = pv.build_detectors(quality, device, device2, batch, half=half)
    windows, stats = timeline_pass(video, out, video_s, detectors, share_iou=pv.SHARE_IOU, batch=batch)
    timeline_s = time.perf_counter() - began
    total_s = time.perf_counter() - started
    timing = {"video_s": round(video_s, 2), "fast_s": round(fast_s, 2), "timeline_s": round(timeline_s, 2),
              "total_s": round(total_s, 2), "ratio": round(total_s / video_s, 2) if video_s else 0.0,
              "seats": len(windows), "crops": stats.get("crops", 0), "crops_shared": stats.get("crops_shared", 0),
              "quality": quality, "half": half, "stats": stats}
    (out / TIMING).write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Xong: {len(windows)} ghế, {total_s:.1f} s cho video {video_s:.1f} s (x{timing['ratio']})", flush=True)
    return timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dòng thời gian mọi ghế cho lõi học (phase 3).")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", choices=("max", "balanced"), default="max")
    parser.add_argument("--device")
    parser.add_argument("--device2")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--reuse-fast", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args(argv)
    try:
        build(args.video, args.out, quality=args.quality, device=args.device, device2=args.device2,
              reuse_fast=args.reuse_fast, half=not args.fp32, batch=args.batch)
    except _process_video().ProcessError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
