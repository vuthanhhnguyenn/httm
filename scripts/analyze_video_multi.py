"""Analyse one recorded video with several candidates (phase 2, offline).

Outputs ``annotated.mp4``, ``events.jsonl``, ``evidence/`` and ``timing.json`` in ``--out``.
Example (GPU environment on Windows):

    .\\.venv-gpu\\Scripts\\python.exe scripts/analyze_video_multi.py storage/videos/lop.mp4 \\
        --out storage/multi/lop --mode realtime
"""

from __future__ import annotations

import argparse
import asyncio
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
    factory.detector_image_size = args.yolo_size
    factory.analyzer_options = analyzer_options(config.detection)
    health = await factory.initialize()
    print("AI:", health, flush=True)
    pipeline = factory.build_multi_pipeline(raw_config, max_subjects=args.max_subjects, crop_size=args.crop_size,
                                            yolo_every=args.yolo_every, motion_skip=not args.no_motion_skip,
                                            focus=not args.no_focus)
    rate = args.fps if args.mode == "full" else 240.0  # realtime: every frame, dropped when late
    source = VideoFileFrameSource(args.video, analysis_rate_hz=rate)
    run_config: dict[str, Any] = {key: (str(value) if isinstance(value, Path) else value)
                                  for key, value in vars(args).items()}
    run_config["hardware"] = _hardware()
    run_config["detector_device"] = factory.health_details().get("object_detector", {}).get("device", "")
    try:
        await source.open()
        await source.close()
    except Exception as exc:
        print(f"Không mở được video: {exc}", file=sys.stderr)
        return 1
    video_fps = args.fps if args.mode == "full" else (source.fps or 30.0)
    source = VideoFileFrameSource(args.video, analysis_rate_hz=rate)
    try:
        result = await run_multi_replay(source, pipeline, policy=settings.raw_policy, raw_config=raw_config,
                                        out_dir=args.out, mode=args.mode, overlap=not args.no_overlap,
                                        write_video=not args.no_video, video_fps=video_fps,
                                        exam_mode=args.exam_mode, run_config=run_config)
    finally:
        factory.close()
    timing = result.timing
    print(f"Xong {timing['frames']} frame, AI FPS {timing['ai_fps']}, độ trễ p95 {timing['latency_ms']['p95']} ms, "
          f"bỏ {timing['dropped_frames']} frame, {len(timing['seats'])} thí sinh, đổi ID {timing['id_switches']}")
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
