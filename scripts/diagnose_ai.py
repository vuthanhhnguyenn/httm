"""Report the active AI environment; optionally profile a supplied local image.

No camera is opened and no model is downloaded. A repeated still image measures
compute latency, not webcam FPS or recognition accuracy on moving subjects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "packages/proctoring_ai/src", ROOT / "packages/proctoring_core/src"):
    sys.path.insert(0, str(source))


async def profile(args):
    import cv2
    import numpy as np
    from proctoring_ai.analyzer_factory import AnalyzerFactory

    image = (np.full((720, 1280, 3), 127, dtype=np.uint8) if args.synthetic
             else cv2.imread(str(args.image)))
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")
    factory = AnalyzerFactory(ROOT / "models/manifest.yaml", detector_image_size=args.image_size,
                              detector_device=args.device)
    try:
        await factory.initialize()
        if factory.health().get("object_detector") != "ready":
            raise RuntimeError(f"Detector initialization failed: {factory.health_details()['object_detector']}")
        pipeline = factory.build_pipeline({"detection": {"image_size": args.image_size, "device": args.device,
                                          "pose_interval_ms": 0 if args.every_frame else 250,
                                          "hands_interval_ms": 0 if args.every_frame else 200}})
        samples, timings, statuses = [], {}, {}
        session = uuid4()
        for index in range(args.frames + 3):
            started = perf_counter()
            bundle = await pipeline.analyze({"image": image, "frame_id": index,
                                              "captured_at_monotonic_ms": int(started * 1000),
                                              "frame_size": (image.shape[1], image.shape[0])}, session_id=session)
            if index < 3:
                continue
            samples.append((perf_counter() - started) * 1000)
            for name, health in bundle.analyzer_health.items():
                if health.status != "skipped":
                    timings.setdefault(name, []).append(health.latency_ms)
                counts = statuses.setdefault(name, {})
                counts[health.status] = counts.get(health.status, 0) + 1
        return {"input": "synthetic_gray_frame" if args.synthetic else str(args.image),
                "kind": "repeated_still_image_not_live_accuracy_test",
                "image_size": args.image_size, "frames": args.frames, "health": factory.health_details(),
                "schedule": "every_frame" if args.every_frame else "pose_250ms_hands_200ms",
                "analyzer_performance": pipeline.performance(),
                "pipeline_fps": round(1000 / float(np.mean(samples)), 2),
                "pipeline_p50_ms": round(float(np.percentile(samples, 50)), 1),
                "pipeline_p95_ms": round(float(np.percentile(samples, 95)), 1),
                "analyzer_p95_ms": {name: round(float(np.percentile(values, 95)), 1)
                                    for name, values in timings.items()}, "status_counts": statuses}
    finally:
        factory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Local image for optional inference timing")
    parser.add_argument("--synthetic", action="store_true", help="Generate a gray frame for a compute-only benchmark")
    parser.add_argument("--device", default="cpu", help="cpu or 0 for the first CUDA GPU")
    parser.add_argument("--image-size", type=int, default=416, choices=(416, 512, 640))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--threads", type=int, help="CPU intra-op threads to compare (e.g. 2, 4)")
    parser.add_argument("--every-frame", action="store_true", help="Compare with all analyzers running on every frame")
    args = parser.parse_args()
    if args.image and args.synthetic:
        parser.error("choose --image or --synthetic")
    if args.frames < 1 or (args.threads is not None and args.threads < 1):
        parser.error("frames and threads must be positive")
    import torch

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    available = torch.cuda.is_available()
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cuda_available": available,
              "compiled_architectures": torch.cuda.get_arch_list(), "cpu_threads": torch.get_num_threads()}
    if available:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = torch.cuda.get_device_capability(0)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.image or args.synthetic:
        if args.device != "cpu" and not available:
            parser.error("CUDA is unavailable in this Python environment; GPU benchmark not started")
        print(json.dumps(asyncio.run(profile(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
