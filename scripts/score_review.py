"""Chấm sự kiện tầng kỹ (stage=review) trong events.jsonl theo kết quả Nguyên đã chấm.

    uv run python scripts/score_review.py storage/multi/lop/events.jsonl data/labels/EPf9O7UXRoM-review.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

from proctoring_ai.multi.cctv.review_score import load_verdicts, score_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    score = score_review([item for item in records if item.get("stage") == "review"], load_verdicts(args.labels))
    print(f"đúng {score['true_found']}/{score['true_total']}, nhầm {score['false_opened']} "
          f"({score['false_events']} sự kiện), chưa chấm {score['unjudged']}")
    print(f"tính theo cảnh báo đang hiện: đúng {score['true_covered']}/{score['true_total']}, "
          f"nhầm {score['false_covered']}")
    if score["covered_only"]:
        print("Chỉ bắt được nhờ cảnh báo đang mở / đã gộp:", ", ".join(score["covered_only"]))
    if score["missed"]:
        print("Sót:", ", ".join(score["missed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
