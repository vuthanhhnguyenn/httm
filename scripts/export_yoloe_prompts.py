"""Lưu YOLOE với sẵn lớp chữ cho tầng kỹ CCTV, để máy chạy không phải tải MobileCLIP.

    python scripts/export_yoloe_prompts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["phone", "paper", "notebook", "book"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/pretrained/yoloe-11l-seg.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "models/pretrained/yoloe-11l-seg-exam.pt")
    args = parser.parse_args(argv)
    from ultralytics import YOLOE

    if not args.source.exists():
        print(f"Thiếu {args.source}; tải từ https://github.com/ultralytics/assets/releases/download/v8.3.0/"
              f"{args.source.name}", file=sys.stderr)
        return 1
    model = YOLOE(str(args.source))
    model.set_classes(NAMES, model.get_text_pe(NAMES))
    model.save(str(args.out))
    print(f"Đã lưu {args.out} với lớp {NAMES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
