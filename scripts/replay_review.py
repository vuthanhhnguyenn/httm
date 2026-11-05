"""Chạy lại luật tầng kỹ CCTV từ đầu ra model đã lưu (review_suspects.py --cache-dir), không cần model.

    uv run python scripts/replay_review.py storage/multi/lop/suspects.jsonl storage/multi/lop-review/cache.json.gz \\
        data/labels/EPf9O7UXRoM-review.yaml

In số sự kiện đúng/nhầm theo kết quả Nguyên chấm. Mỗi lần chỉnh luật chỉ mất vài giây.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

from proctoring_ai.multi.cctv.review_cache import load_cache, window_key  # noqa: E402
from proctoring_ai.multi.cctv.review_dedupe import ReviewDeduper  # noqa: E402
from proctoring_ai.multi.cctv.review_frames import select_offsets  # noqa: E402
from proctoring_ai.multi.cctv.review_rules import ReviewRuleConfig, review_window  # noqa: E402
from proctoring_ai.multi.cctv.review_score import load_verdicts, score_review  # noqa: E402
from proctoring_ai.multi.cctv.suspects import SuspectWindow  # noqa: E402


def replay(suspects: Path, cache: Path, *, config: ReviewRuleConfig | None = None,
           max_frames: int | None = None, allow_missing: bool = False) -> list[dict[str, Any]]:
    windows = [SuspectWindow.from_json(json.loads(line))
               for line in suspects.read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = load_cache(cache)
    dedupe = ReviewDeduper()
    records: list[dict[str, Any]] = []
    for window in windows:
        analysed = frames.get(window_key(window))
        if analysed is None:
            if allow_missing:
                continue
            raise KeyError(f"cache không có đoạn {window_key(window)}")
        if max_frames is not None:
            keep = set(select_offsets(window, [frame.offset_ms for frame in analysed], max_frames=max_frames))
            analysed = [frame for frame in analysed if frame.offset_ms in keep]
        records += dedupe.admit(review_window(window, analysed, config))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suspects", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--max-frames", type=int, default=None, help="số khung tối đa mỗi đoạn")
    parser.add_argument("--allow-missing", action="store_true", help="bỏ qua đoạn chưa có trong cache")
    parser.add_argument("--events", type=Path, default=None, help="ghi sự kiện ra file JSONL")
    args = parser.parse_args(argv)
    records = replay(args.suspects, args.cache, max_frames=args.max_frames, allow_missing=args.allow_missing)
    if args.events is not None:
        args.events.write_text("".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in records),
                               encoding="utf-8")
    score = score_review(records, load_verdicts(args.labels))
    print(f"đúng {score['true_found']}/{score['true_total']}, nhầm {score['false_opened']} "
          f"({score['false_events']} sự kiện), chưa chấm {score['unjudged']}")
    if score["found_by_merged"]:
        print("Bắt được qua sự kiện đã gộp:", ", ".join(score["found_by_merged"]),
              f"(tính cả: đúng {score['true_with_merged']}/{score['true_total']})")
    if score["missed"]:
        print("Sót:", ", ".join(score["missed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
