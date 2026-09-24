"""Benchmark Qt painting with synthetic 720p frames, without opening a webcam."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in ("apps/desktop/src", "packages/proctoring_ai/src", "packages/proctoring_core/src"):
    sys.path.insert(0, str(ROOT / source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--paint-interval", type=int, default=16)
    parser.add_argument("--capture-fps", type=int, default=30, choices=(25, 30))
    parser.add_argument("--coarse", action="store_true", help="Compare with the previous default Qt timer")
    args = parser.parse_args()
    if args.seconds < 3 or args.paint_interval < 1:
        parser.error("seconds >= 3 and paint-interval >= 1 are required")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from proctoring_desktop.window import MainWindow
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    window = MainWindow(ROOT)
    window.show()
    window._timer.setInterval(args.paint_interval)
    window._timer.setTimerType(Qt.TimerType.CoarseTimer if args.coarse else Qt.TimerType.PreciseTimer)
    pixels = np.full((720, 1280, 3), 127, dtype=np.uint8)
    produced = 0
    times = []

    def capture():
        nonlocal produced
        produced += 1
        times.append(time.perf_counter())
        window.preview.append({"frame_id": produced, "captured_at_monotonic_ms": time.monotonic_ns() // 1_000_000,
                               "frame_size": (1280, 720), "image": pixels})

    timer = QTimer()
    timer.setTimerType(Qt.TimerType.PreciseTimer)
    timer.setInterval(round(1000 / args.capture_fps))
    timer.timeout.connect(capture)
    timer.start()
    QTimer.singleShot(args.seconds * 1000, app.quit)
    app.exec()
    timer.stop()
    window._timer.stop()
    print(json.dumps({"kind": "synthetic_preview_no_camera_no_ai_load", "seconds": args.seconds,
                      "requested_capture_fps": args.capture_fps, "paint_interval_ms": args.paint_interval,
                      "coarse": args.coarse, "produced_frames": produced,
                      "measured_source_fps": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0,
                      "measured_preview_fps": window._preview_fps}, indent=2))
    window.close()


if __name__ == "__main__":
    main()
