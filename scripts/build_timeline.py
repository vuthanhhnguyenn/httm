"""Tầng 1 của lõi học (phase 3): mọi ghế, suốt cả video, qua model lớn.

    python scripts/build_timeline.py video.mp4 --out timeline/lop [--quality max|balanced] [--device 0] [--device2 1]
        [--chunk-seats 3 --resume]   # máy dễ bị dừng giữa chừng; --device intel:cpu = OpenVINO trên CPU

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


def _keep_good_lines(cache_path: Path) -> set[str]:
    """Rewrite ``cache_path`` with only its complete lines (a run killed mid-write leaves a cut-off one) and
    return the seats those lines hold."""

    import gzip
    import zlib

    good: list[str] = []
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    break
                good.append(json.dumps(item))
    except (EOFError, OSError, zlib.error):
        pass  # the rest of the file is cut off: keep what was read
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in good)
    return {str(json.loads(line)["key"][0]) for line in good}


def timeline_pass(video: Path, out: Path, video_s: float, detectors: Any, *, share_iou: float,
                  batch: int | None = None, chunk_seats: int | None = None,
                  resume: bool = False) -> tuple[list[Any], dict[Any, list[Any]], dict[str, Any]]:
    """All-seat windows from ``out/tracks.jsonl`` through ``review_offline``; the cache goes to
    ``out/timeline.json.gz``. Returns the windows, their analysed frames and the run's stats.

    ``chunk_seats``: run that many seats per pass (each pass decodes the video again) so a killed run loses
    at most one pass. ``resume``: keep the seats an earlier run already wrote and do only the others."""

    from proctoring_ai.multi.cctv.review_cache import load_cache

    pv = _process_video()
    windows = all_seat_windows(load_tracks(out / "tracks.jsonl"), int(video_s * 1000))
    cache_path = out / TIMELINE
    done = _keep_good_lines(cache_path) if resume and cache_path.is_file() else set()
    if not done:
        cache_path.unlink(missing_ok=True)  # save_window appends
    if not windows:
        return windows, {}, {"seats": 0}
    todo = [window for window in windows if str(window.seat_id) not in done]
    size = max(1, chunk_seats or len(todo) or 1)
    stats: dict[str, Any] = {"seats_resumed": len(windows) - len(todo)}
    for first in range(0, len(todo), size):
        chunk = todo[first:first + size]

        def progress(done_frames: int, total: int, first: int = first, count: int = len(chunk)) -> None:
            print(f"\rDòng thời gian (ghế {first + 1}-{first + count}/{len(todo)}): {done_frames}/{total} khung",
                  end="\n" if done_frames >= total else "", flush=True)

        result = review_offline(video, chunk, detectors, fps=pv.REVIEW_FPS, warmup_ms=0, share_iou=share_iou,
                                cache_path=cache_path, parallel=True, batch_size=batch, on_progress=progress)
        for key, value in result.stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats[key] = round(stats.get(key, 0) + value, 3)
            else:
                stats.setdefault(key, value)
    frames = load_cache(cache_path) if cache_path.is_file() else {}
    return windows, frames, {"seats": len(windows), **stats}


def build(video: Path, out: Path, *, quality: str = "max", device: str | None = None, device2: str | None = None,
          detectors: Any = None, fast_runner: Any = None, reuse_fast: bool = False, half: bool = True,
          batch: int | None = None, chunk_seats: int | None = None, resume: bool = False) -> dict[str, Any]:
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
    windows, _frames, stats = timeline_pass(video, out, video_s, detectors, share_iou=pv.SHARE_IOU, batch=batch,
                                            chunk_seats=chunk_seats, resume=resume)
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
    parser.add_argument("--chunk-seats", type=int, help="số ghế mỗi lượt (lượt bị dừng chỉ mất lượt đó)")
    parser.add_argument("--resume", action="store_true", help="giữ các ghế đã xong, làm tiếp ghế còn lại")
    args = parser.parse_args(argv)
    try:
        build(args.video, args.out, quality=args.quality, device=args.device, device2=args.device2,
              reuse_fast=args.reuse_fast or args.resume, half=not args.fp32, batch=args.batch,
              chunk_seats=args.chunk_seats, resume=args.resume)
    except _process_video().ProcessError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
