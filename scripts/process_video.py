"""Một lệnh từ video phòng thi (CCTV) ra video đã đánh dấu cảnh báo đỏ/vàng và báo cáo (phase 2g).

    python scripts/process_video.py video.mp4 --out ket-qua/lop [--quality max|balanced] [--device 0] [--device2 1]

Ba lượt: tầng nhanh (đoạn nghi vấn, khung ghế), tầng kỹ trọn đoạn (một lần giải mã), rồi mức tin cậy,
ảnh bằng chứng, video và báo cáo. Ghi vào ``--out``: ``annotated.mp4``, ``report.html``, ``events.csv``,
``events.jsonl``, ``evidence/``, ``review_cache.json.gz``, ``timing.json``, ``suspects.jsonl``, ``tracks.jsonl``
(và ``fast/``: đầu ra riêng của tầng nhanh). Mốc: tổng thời gian ≤ 2 × độ dài video trên Kaggle T4.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

import cv2  # noqa: E402
from proctoring_ai.multi.cctv.confidence import (  # noqa: E402
    ConfidenceRules,
    assign_levels,
    load_confidence,
)
from proctoring_ai.multi.cctv.review_cache import CacheKey, window_key  # noqa: E402
from proctoring_ai.multi.cctv.review_dedupe import ReviewDeduper  # noqa: E402
from proctoring_ai.multi.cctv.review_detectors import ModelSpec  # noqa: E402
from proctoring_ai.multi.cctv.review_offline import review_offline  # noqa: E402
from proctoring_ai.multi.cctv.review_queue import skipped_record  # noqa: E402
from proctoring_ai.multi.cctv.review_rules import (  # noqa: E402
    ReviewFrame,
    ReviewRuleConfig,
    review_window,
)
from proctoring_ai.multi.cctv.suspects import SuspectWindow  # noqa: E402
from proctoring_runtime.messages import event_name  # noqa: E402
from proctoring_runtime.render import render_video  # noqa: E402
from proctoring_runtime.report import write_report  # noqa: E402
from proctoring_runtime.tracks import load_tracks  # noqa: E402

QUALITY_IMGSZ = {"max": 960, "balanced": 640}  # deep-tier crop size; balanced if max misses the 2x budget
BUDGET_RATIO = 2.0
REVIEW_FPS = 5.0
WARMUP_MS = 6000
DEFAULT_DEVICE = "cuda"

FastRunner = Callable[[Path, Path, Path], None]
WindowRecords = list[tuple[SuspectWindow, list[dict[str, Any]]]]


class ProcessError(Exception):
    """A failure that stops the run; the message is Vietnamese, for the person running it."""


@cache
def _sibling(name: str) -> Any:
    """Another script of ``scripts/`` (not a package) as a module."""

    spec = importlib.util.spec_from_file_location(f"_pv_{name}", Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fast(video: Path, fast_dir: Path, tracks_path: Path) -> None:
    """The fast tier as a function call: CCTV, every frame at 5 fps, suspects and tracks, no video."""

    fast = _sibling("analyze_video_multi")
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    args = fast.parse_args([str(video), "--out", str(fast_dir), "--scene", "cctv", "--mode", "full", "--fps", "5",
                            "--suspects", "--no-video", "--tracks", str(tracks_path)])
    code = asyncio.run(fast.run(args))
    if code != 0:
        raise ProcessError(f"Tầng nhanh không chạy được (mã {code}); xem thông báo ở trên.")


def device_overrides(specs: Sequence[ModelSpec], device: str | None, device2: str | None) -> dict[str, str]:
    """Where each deep-tier model runs: pose models on ``device``, object models on ``device2`` (default: the
    same card); ``cuda`` when nothing is given. The ``device`` of ``review_models.yaml`` is ignored (a Kaggle
    machine has no Intel card)."""

    first = device or DEFAULT_DEVICE
    second = device2 or first
    return {spec.name: second if spec.kind == "objects" else first for spec in specs}


def build_detectors(quality: str, device: str | None, device2: str | None, batch: int | None) -> Any:
    from proctoring_ai.multi.cctv.review_detectors import ReviewDetectors, load_specs

    specs = load_specs(ROOT / "configs" / "review_models.yaml", ROOT / "models")
    return ReviewDetectors(specs, image_size=QUALITY_IMGSZ[quality], batch_size=batch or 16, half=True,
                           device_override=device_overrides(specs, device, device2))


def _probe(video: Path) -> float:
    """Video length in seconds; ProcessError when it cannot be opened or has no frame."""

    if not video.is_file():
        raise ProcessError(f"Không tìm thấy video: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ProcessError(f"Không mở được video: {video}")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        rate = capture.get(cv2.CAP_PROP_FPS) or 0
        ok, _image = capture.read()
    finally:
        capture.release()
    if not ok or count <= 0:
        raise ProcessError(f"Video không có khung hình nào: {video}")
    return float(count / rate) if rate else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def review_records(windows: Sequence[SuspectWindow], frames: Mapping[CacheKey, list[ReviewFrame]], *,
                   errors: Mapping[int, str] | None = None,
                   config: ReviewRuleConfig | None = None) -> WindowRecords:
    """The deep-tier rules per window, in window order through one ``ReviewDeduper``, as ``review_offline``
    gives them, but kept per window (for its evidence picture). A window whose models failed (``errors``),
    whose rules fail, or that got no analysed frame inside its span becomes ``REVIEW_SKIPPED``."""

    errors = errors or {}
    dedupe = ReviewDeduper()
    out: WindowRecords = []
    for index, window in enumerate(windows):
        analysed = frames.get(window_key(window), [])
        if index in errors:
            found = [skipped_record(window, "error")]
        elif not any(window.start_ms <= frame.offset_ms <= window.end_ms for frame in analysed):
            found = [skipped_record(window, "frames_gone")]
        else:
            try:
                found = review_window(window, analysed, config)
            except Exception as exc:  # one window's bad data must not cost the others
                print(f"Đoạn {window_key(window)}: luật tầng kỹ lỗi ({exc}); chưa kiểm tra kỹ.", file=sys.stderr)
                found = [skipped_record(window, "error")]
        out.append((window, dedupe.admit(found)))
    return out


def finish_records(per_window: WindowRecords, fast_events: list[dict[str, Any]],
                   rules: ConfidenceRules) -> list[dict[str, Any]]:
    """All records in order, with ``label_vi``, and ``level`` / ``level_reason`` on opened and merged ones.
    The fast tier's own rule events only confirm deep-tier events (``fast_agrees_s``), never add rows."""

    records = [item for _window, found in per_window for item in found]
    for item in records:
        if "type" in item:
            item["label_vi"] = event_name(str(item["type"]))
    return assign_levels(records, fast_events, rules)


def _deep_pass(video: Path, windows: list[SuspectWindow], detectors: Any, cache_path: Path,
               batch: int | None) -> tuple[WindowRecords, dict[CacheKey, list[ReviewFrame]], dict[str, Any]]:
    cache_path.unlink(missing_ok=True)  # save_window appends
    stats: dict[str, Any] = {"windows": len(windows), "skipped": 0,
                             "models": dict(getattr(detectors, "status", {}) or {})}
    if not windows:
        return [], {}, stats
    kinds: Mapping[str, str] = getattr(detectors, "kinds", {}) or {}
    if not any(kind == "pose" for kind in kinds.values()):
        print("Tầng kỹ: không nạp được model khung xương; mọi đoạn nghi vấn thành \"chưa kiểm tra kỹ\".",
              file=sys.stderr)
        per_window = review_records(windows, {}, errors=dict.fromkeys(range(len(windows)), "no_model"))
        stats.update(skipped=len(windows), error="không có model khung xương")
        return per_window, {}, stats

    def progress(done: int, total: int) -> None:
        print(f"\rTầng kỹ: {done}/{total} khung", end="\n" if done >= total else "", flush=True)

    errors: dict[int, str] = {}
    try:
        result = review_offline(video, windows, detectors, fps=REVIEW_FPS, warmup_ms=WARMUP_MS,
                                cache_path=cache_path, parallel=True, batch_size=batch, on_progress=progress)
        frames, run_stats = result.frames, dict(result.stats)
    except Exception as exc:
        # Crops of many windows share a batch: find the windows that fail by running each on its own.
        print(f"\nTầng kỹ lỗi ({exc}); xem lại từng đoạn riêng.", file=sys.stderr)
        cache_path.unlink(missing_ok=True)
        frames, run_stats = {}, {"fallback": True, "first_error": str(exc)}
        for index, window in enumerate(windows):
            try:
                single = review_offline(video, [window], detectors, fps=REVIEW_FPS, warmup_ms=WARMUP_MS,
                                        cache_path=cache_path, parallel=True, batch_size=batch)
            except Exception as window_exc:
                errors[index] = str(window_exc)
                continue
            frames.update(single.frames)
            for key, value in single.stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    run_stats[key] = round(run_stats.get(key, 0) + value, 3)
    began = time.perf_counter()
    per_window = review_records(windows, frames, errors=errors)
    run_stats["rules_s"] = round(time.perf_counter() - began, 3)
    skipped = sum(1 for _w, found in per_window for item in found
                  if item.get("action") == "opened" and item.get("type") == "REVIEW_SKIPPED")
    stats.update(run_stats, skipped=skipped, errors=len(errors))
    return per_window, frames, stats


def _evidence(video: Path, out: Path, per_window: WindowRecords, records: list[dict[str, Any]],
              frames: Mapping[CacheKey, list[ReviewFrame]]) -> int:
    """One picture per opened alert, read again at the window's nearest analysed frame (ruling P1) and drawn
    as ``review_suspects`` draws it; ``evidence`` is set on the record when the picture was written."""

    draw = _sibling("review_suspects")._draw
    directory = out / "evidence"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    by_id = {str(item.get("id")): item for item in records if item.get("action") == "opened"}
    jobs: list[tuple[int, SuspectWindow, ReviewFrame | None, dict[str, Any]]] = []
    for window, found in per_window:
        analysed = frames.get(window_key(window), [])
        for raw in found:
            item = by_id.get(str(raw.get("id"))) if raw.get("action") == "opened" else None
            if item is None:
                continue
            at = int(item.get("video_offset_ms", 0))
            nearest = min(analysed, key=lambda frame: abs(frame.offset_ms - at)) if analysed else None
            jobs.append((nearest.offset_ms if nearest is not None else at, window, nearest, item))
    written = 0
    capture = cv2.VideoCapture(str(video))
    try:
        for at, window, frame, item in sorted(jobs, key=lambda job: job[0]):
            capture.set(cv2.CAP_PROP_POS_MSEC, at)
            ok, image = capture.read()
            if not ok:
                continue
            name = f"evidence/{item['id']}.jpg"
            if cv2.imwrite(str(out / name), draw(image, window, frame, item)):
                item["evidence"] = name
                written += 1
    finally:
        capture.release()
    return written


def process(video: Path, out: Path, *, quality: str = "max", device: str | None = None,
            device2: str | None = None, detectors: Any = None, fast_runner: FastRunner | None = None,
            batch: int | None = None, ffmpeg: str | None = "ffmpeg") -> dict[str, Any]:
    """Run the three passes; returns what ``timing.json`` holds. ``detectors`` / ``fast_runner`` replace the
    real models and fast tier (tests). Raises ``ProcessError`` when the video cannot be read."""

    if quality not in QUALITY_IMGSZ:
        raise ProcessError(f"--quality chỉ nhận {', '.join(QUALITY_IMGSZ)}")
    started = time.perf_counter()
    video_s = _probe(video)
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: fast tier.
    fast_dir, tracks_path = out / "fast", out / "tracks.jsonl"
    shutil.rmtree(fast_dir, ignore_errors=True)
    tracks_path.unlink(missing_ok=True)
    began = time.perf_counter()
    (fast_runner or run_fast)(video, fast_dir, tracks_path)
    fast_s = time.perf_counter() - began
    if (fast_dir / "suspects.jsonl").is_file():
        shutil.copyfile(fast_dir / "suspects.jsonl", out / "suspects.jsonl")
    windows = [SuspectWindow.from_json(item) for item in _read_jsonl(out / "suspects.jsonl")]
    fast_events = [item for item in _read_jsonl(fast_dir / "events.jsonl")
                   if item.get("action") == "opened" and item.get("stage") != "review"]
    print(f"Tầng nhanh: {fast_s:.1f} s, {len(windows)} đoạn nghi vấn, {len(fast_events)} sự kiện luật", flush=True)

    # Pass 2: deep tier over whole windows.
    began = time.perf_counter()
    if detectors is None and windows:
        detectors = build_detectors(quality, device, device2, batch)
    devices = dict(getattr(detectors, "devices", {}) or {}) if detectors is not None else {}
    reasons = dict(getattr(detectors, "reasons", {}) or {}) if detectors is not None else {}
    if detectors is not None:
        for name, status in getattr(detectors, "status", {}).items():
            print(f"Tầng kỹ - model {name}: {status} ({devices.get(name) or 'mặc định'}"
                  f"{'; ' + reasons[name] if name in reasons else ''})", flush=True)
    if any(str(value) == "cpu" for value in devices.values()):
        print("Cảnh báo: tầng kỹ chạy trên CPU, sẽ chậm hơn nhiều.", file=sys.stderr)
    per_window, frames, deep_stats = _deep_pass(video, windows, detectors, out / "review_cache.json.gz", batch)
    if not windows:
        (out / "review_cache.json.gz").unlink(missing_ok=True)
    review_s = time.perf_counter() - began

    # Pass 3: levels, evidence, video, report.
    began = time.perf_counter()
    records = finish_records(per_window, fast_events, load_confidence(ROOT / "configs" / "confidence.yaml"))
    deep_stats["evidence"] = _evidence(video, out, per_window, records, frames)
    with (out / "events.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in records)
    report_s = time.perf_counter() - began
    tracks = load_tracks(tracks_path) if tracks_path.is_file() else []
    rendered = render_video(video, out / "annotated.mp4", tracks, records, ffmpeg=ffmpeg)
    render_s = float(rendered.get("seconds", 0.0))

    def summary() -> dict[str, Any]:
        total_s = time.perf_counter() - started
        ratio = total_s / video_s if video_s > 0 else 0.0
        timing: dict[str, Any] = {
            "video_s": round(video_s, 3), "fast_s": round(fast_s, 3), "review_s": round(review_s, 3),
            "render_s": round(render_s, 3), "report_s": round(report_s, 3), "total_s": round(total_s, 3),
            "ratio": round(ratio, 3), "over_budget": ratio > BUDGET_RATIO, "devices": devices,
            "device_reasons": reasons, "deep_stats": deep_stats, "encoder": rendered.get("encoder")}
        if rendered.get("error"):
            timing["render_error"] = rendered["error"]
        return timing

    began = time.perf_counter()
    write_report(out, records, video_name=video.name, timing=summary())
    report_s += time.perf_counter() - began
    timing = summary()
    (out / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")

    opened = [item for item in records if item.get("action") == "opened"]
    red = sum(1 for item in opened if item.get("level") == "red")
    print(f"Sự kiện: đỏ {red}, vàng {len(opened) - red}. Thời gian {timing['total_s']:.1f} s cho video "
          f"{video_s:.1f} s (tỉ lệ {timing['ratio']:.2f}; nhanh {fast_s:.1f}, kỹ {review_s:.1f}, "
          f"dựng {render_s:.1f}, báo cáo {report_s:.1f}), mã hoá {timing['encoder']}")
    if rendered.get("error"):
        print(f"Cảnh báo: dựng video gặp lỗi ({rendered['error']}).", file=sys.stderr)
    if timing["over_budget"]:
        print(f"Cảnh báo: vượt mốc {BUDGET_RATIO:g}× độ dài video (tỉ lệ {timing['ratio']:.2f}).", file=sys.stderr)
    print(f"Kết quả: {out}")
    return timing


def main(argv: list[str] | None = None, *, fast_runner: FastRunner | None = None) -> int:
    parser = argparse.ArgumentParser(description="Từ video phòng thi ra video đã đánh dấu và báo cáo sự kiện.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", choices=tuple(QUALITY_IMGSZ), default="max",
                        help="balanced: ảnh cắt tầng kỹ 640 thay vì 960, nhanh hơn")
    parser.add_argument("--device", default=None, help="card cho model khung xương (vd. 0); mặc định cuda")
    parser.add_argument("--device2", default=None, help="card cho model vật thể (vd. 1); mặc định như --device")
    parser.add_argument("--batch", type=int, default=None, help="số ảnh cắt mỗi lần chạy model (mặc định 16)")
    args = parser.parse_args(argv)
    try:
        process(args.video, args.out, quality=args.quality, device=args.device, device2=args.device2,
                fast_runner=fast_runner, batch=args.batch)
    except ProcessError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
