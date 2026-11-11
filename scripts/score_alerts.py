"""Chấm bộ cảnh báo trong events.jsonl theo các tệp nhãn của Nguyên.

    uv run python scripts/score_alerts.py /home/claude/rc/clip1/events.jsonl data/labels/EPf9O7UXRoM-*.yaml [--json]
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

from proctoring_ai.multi.cctv.alert_score import score_alerts  # noqa: E402
from proctoring_ai.multi.cctv.review_score import Verdict, load_verdicts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("labels", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true", help="in dict kết quả thay cho bảng")
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts: list[Verdict] = []
    for path in args.labels:
        verdicts.extend(load_verdicts(path))
    score = score_alerts(records, verdicts)
    if args.json:
        print(json.dumps(score, ensure_ascii=False, indent=2))
        return 0
    print(f"bắt được {score['caught']}/{score['true_total']}, đúng {score['alerts_correct']}/{score['alerts_judged']} "
          f"đã chấm, đúng loại {score['alerts_type_correct']}, nhầm bị phủ {score['false_flagged']}/{score['false_total']}, đỏ {score['red']}, "
          f"tổng cảnh báo {score['alerts']}")
    for kind, row in sorted(score["by_type"].items()):
        print(f"  {kind:<24} {row['alerts']:>4} cảnh báo, {row['correct']:>4} đúng")
    return 0


if __name__ == "__main__":
    sys.exit(main())
