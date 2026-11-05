"""Analyse one recorded video with several candidates (phase 2, offline).

Outputs ``annotated.mp4``, ``events.jsonl``, ``evidence/`` and ``timing.json`` in ``--out``.
Example (GPU environment on Windows):

    .\\.venv-gpu\\Scripts\\python.exe scripts/analyze_video_multi.py storage/videos/lop.mp4 \\
        --out storage/multi/lop --mode realtime
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for source in (
    ROOT / "packages" / "proctoring_core" / "src",
    ROOT / "packages" / "proctoring_ai" / "src",
    ROOT / "packages" / "proctoring_runtime" / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

# Recorded video: inference may be slow, but slowness must never become missing observations.
FULL_MODE_TIMEOUT_MS = 120_000


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích nhiều thí sinh trên một file video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "realtime"), default="full")
    parser.add_argument("--fps", type=float, default=10.0, help="tốc độ lấy mẫu ở chế độ full")
    parser.add_argument("--yolo-size", type=int, choices=(416, 640), default=416)
    parser.add_argument("--yolo-every", type=int, choices=(1, 2), default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-subjects", type=int, default=6)
    parser.add_argument("--no-motion-skip", action="store_true")
    parser.add_argument("--no-focus", action="store_true")
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exam-mode", choices=("computer", "paper"), default="computer")
    parser.add_argument("--scene", choices=("webcam", "cctv"), default="webcam",
                        help="webcam: 2-6 người, phân tích mặt; cctv: camera cả phòng, khung xương YOLO-pose")
    parser.add_argument("--pose-size", type=int, default=960, help="cctv: kích thước ảnh vào YOLO-pose")
    parser.add_argument("--pose-conf", type=float, default=0.25, help="cctv: ngưỡng tin cậy người")
    parser.add_argument("--phone-every", type=int, default=3, help="cctv: tìm điện thoại mỗi N frame")
    parser.add_argument("--tile-every", type=int, default=4,
                        help="cctv: soi kỹ hàng xa (2 ô phóng to) mỗi N frame; 0 để tắt")
    parser.add_argument("--debug", action="store_true", help="cctv: ghi debug.jsonl (hướng đầu, khoảng cách từng cặp)")
    parser.add_argument("--phone-device", choices=("auto", "cuda", "intel:gpu", "cpu"), default=None,
                        help="cctv: thiết bị cho máy dò điện thoại (intel:gpu = OpenVINO trên card Intel)")
    parser.add_argument("--review-live", action="store_true",
                        help="cctv: chạy tầng kỹ ngay trong lúc phân tích (kèm --suspects), kết quả có stage=review")
    parser.add_argument("--review-models", type=Path, default=ROOT / "configs" / "review_models.yaml")
    parser.add_argument("--ring-jpeg", action="store_true",
                        help="cctv --review-live: giữ 20 s khung gần nhất dạng JPEG (~20 MB thay vì ~280 MB)")
    parser.add_argument("--no-seat-map", action="store_true", help="cctv: theo dõi từng người thay vì bản đồ ghế")
    parser.add_argument("--suspects", action="store_true",
                        help="cctv: ghi suspects.jsonl (đoạn nghi vấn) cho scripts/review_suspects.py")
    return parser.parse_args(argv)


def _hardware() -> dict[str, str]:
    info = {"os": platform.platform(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def _write_suspects(video: Path, out: Path, tracker: Any) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video))
    count, rate = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0, capture.get(cv2.CAP_PROP_FPS) or 0
    capture.release()
    duration_ms = int(1000 * count / rate) if count and rate else None
    windows = tracker.finish(duration_ms)
    with (out / "suspects.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(window.to_json()) + "\n" for window in windows)
    spans = sorted((window.start_ms, window.end_ms) for window in windows)
    covered, reach = 0, 0
    for start, end in spans:
        start = max(start, reach)
        covered += max(0, end - start)
        reach = max(reach, end)
    share = f", chiếm {covered / duration_ms:.0%} clip" if duration_ms else ""
    print(f"Đoạn nghi vấn: {len(windows)}{share} -> {out / 'suspects.jsonl'}")


async def run(args: argparse.Namespace) -> int:
    from proctoring_ai.capture.video_file_source import VideoFileFrameSource
    from proctoring_core.config.schema import AppConfig
    from proctoring_runtime.bootstrap import analyzer_options, create_factory, load_settings
    from proctoring_runtime.multi_replay import run_multi_replay

    settings = load_settings(ROOT)
    detection = dict(settings.raw_config.get("detection", {}))
    detection["image_size"] = args.yolo_size
    if args.mode == "full":
        detection["analyzer_timeout_ms"] = FULL_MODE_TIMEOUT_MS
    raw_config = {**settings.raw_config, "detection": detection}
    config = AppConfig.model_validate(raw_config)
    factory = create_factory(ROOT, settings)
    cctv = args.scene == "cctv"
    # CCTV: phones are small in a whole-room view, so the object detector runs at the pose size.
    factory.detector_image_size = args.pose_size if cctv else args.yolo_size
    factory.analyzer_options = analyzer_options(config.detection)
    if cctv and args.phone_device:
        factory.phone_device = args.phone_device
    health = await factory.initialize()
    rules: Any = None
    if cctv:
        from proctoring_ai.multi.cctv.rules import CctvSeatRules

        health = await factory.initialize_pose_detector(image_size=args.pose_size, confidence=args.pose_conf)
        print("AI:", health, flush=True)
        if factory.pose_detector is None:
            print(f"Không tải được YOLO-pose: {factory.health_details()['pose_detector']['reason']}. "
                  "Chạy scripts/download_model.py để tải model.", file=sys.stderr)
            factory.close()
            return 1
        timeout = FULL_MODE_TIMEOUT_MS if args.mode == "full" else 1200
        pipeline: Any = factory.build_cctv_pipeline(max_subjects=80, phone_every=args.phone_every,
                                                    timeout_ms=timeout, seat_map=not args.no_seat_map,
                                                    tile_every=args.tile_every)
        suspects: Any = None
        if args.suspects or args.review_live:
            from proctoring_ai.multi.cctv.suspects import SuspectTracker

            suspects = SuspectTracker()
        if args.no_seat_map:
            rules = CctvSeatRules(debug=args.debug, suspects=suspects)
        else:
            from proctoring_ai.multi.cctv.seat_map import SeatMapRoles

            rules = CctvSeatRules(roles=SeatMapRoles(pipeline.tracker), debug=args.debug, suspects=suspects)
    else:
        print("AI:", health, flush=True)
        pipeline = factory.build_multi_pipeline(raw_config, max_subjects=args.max_subjects,
                                                crop_size=args.crop_size, yolo_every=args.yolo_every,
                                                motion_skip=not args.no_motion_skip, focus=not args.no_focus)
    rate = args.fps if args.mode == "full" else 240.0  # realtime: every frame, dropped when late
    source = VideoFileFrameSource(args.video, analysis_rate_hz=rate)
    run_config: dict[str, Any] = {key: (str(value) if isinstance(value, Path) else value)
                                  for key, value in vars(args).items()}
    run_config["hardware"] = _hardware()
    run_config["analyzers"] = factory.health_details()  # load failures and their reasons
    try:
        await source.open()
        await source.close()
    except Exception as exc:
        print(f"Không mở được video: {exc}", file=sys.stderr)
        return 1
    video_fps = args.fps if args.mode == "full" else (source.fps or 30.0)
    from proctoring_ai.capture.prefetch_source import PrefetchFrameSource

    # Decoding the next frames overlaps the AI on the current one.
    prefetched = PrefetchFrameSource(VideoFileFrameSource(args.video, analysis_rate_hz=rate), max_queue=8)
    side_tasks: list[Any] = []
    live: Any = None
    if cctv and args.review_live:
        from proctoring_ai.multi.cctv.frame_ring import FrameRing
        from proctoring_ai.multi.cctv.review_detectors import ReviewDetectors, load_specs
        from proctoring_ai.multi.cctv.review_worker import LiveReview, ReviewWorker

        detectors = ReviewDetectors(load_specs(args.review_models, ROOT / "models"))
        for name, status in detectors.status.items():
            print(f"Tầng kỹ - model {name}: {status} ({detectors.devices.get(name) or 'auto'}"
                  f"{'; ' + detectors.reasons[name] if name in detectors.reasons else ''})", flush=True)
        ring = FrameRing(span_ms=20000, fps=5.0, jpeg_quality=90 if args.ring_jpeg else None)
        live = LiveReview(rules.suspects, ReviewWorker(detectors, ring, video_path=args.video), ring)
        side_tasks.append(live)
    try:
        result = await run_multi_replay(prefetched, pipeline, policy=settings.raw_policy, raw_config=raw_config,
                                        out_dir=args.out, mode=args.mode, overlap=not args.no_overlap,
                                        write_video=not args.no_video, video_fps=video_fps,
                                        exam_mode=args.exam_mode, run_config=run_config, rules=rules,
                                        side_tasks=side_tasks)
    finally:
        factory.close()
    if rules is not None and getattr(rules, "debug_rows", None) is not None:
        with (args.out / "debug.jsonl").open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rules.debug_rows)
    if live is not None:
        with (args.out / "suspects.jsonl").open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(window.to_json()) + "\n" for window in live.windows)
        lat = result.timing.get("review_latency_ms", {})
        print(f"Tầng kỹ: {result.timing.get('review_windows')} đoạn, bỏ qua {result.timing.get('review_skipped')}, "
              f"độ trễ p50 {lat.get('p50')} ms, p95 {lat.get('p95')} ms")
    elif rules is not None and getattr(rules, "suspects", None) is not None:
        _write_suspects(args.video, args.out, rules.suspects)
    timing = result.timing
    print(f"Xong {timing['frames']} frame, AI FPS {timing['ai_fps']}, độ trễ p95 {timing['latency_ms']['p95']} ms, "
          f"bỏ {timing['dropped_frames']} frame, "
          f"{timing['max_visible'] if cctv else len(timing['seats'])} người, đổi ID {timing['id_switches']}")
    if cctv:
        print(f"  Vai trò: {timing.get('roles')}, sự kiện: {timing.get('events_opened')}")
    for seat, stats in timing["seats"].items():
        print(f"  #{seat}: mặt {stats['face_hz']} lần/giây (mới {stats['fresh_face_hz']}), "
              f"hiệu chỉnh {stats['calibration_status']}")
    print(f"Kết quả: {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        print(f"Thư mục {args.out} đã có dữ liệu; thêm --overwrite để ghi đè.", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"Không tìm thấy video: {args.video}", file=sys.stderr)
        return 1
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
