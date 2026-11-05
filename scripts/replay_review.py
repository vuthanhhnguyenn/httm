"""Chạy lại luật tầng kỹ CCTV từ đầu ra model đã lưu (review_suspects.py --cache-dir), không cần model.

    uv run python scripts/replay_review.py storage/multi/lop/suspects.jsonl storage/multi/lop-review/cache.json.gz \\
        data/labels/EPf9O7UXRoM-review.yaml

In số sự kiện đúng/nhầm theo kết quả Nguyên chấm. Mỗi lần chỉnh luật chỉ mất vài giây.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

from proctoring_ai.multi.cctv.review_cache import load_cache, window_key  # noqa: E402
from proctoring_ai.multi.cctv.review_core import ReviewHistory  # noqa: E402
from proctoring_ai.multi.cctv.review_dedupe import ReviewDeduper  # noqa: E402
from proctoring_ai.multi.cctv.review_frames import select_offsets  # noqa: E402
from proctoring_ai.multi.cctv.review_rules import (  # noqa: E402
    ReviewFrame,
    ReviewRuleConfig,
    review_window,
)
from proctoring_ai.multi.cctv.review_score import load_verdicts, score_review  # noqa: E402
from proctoring_ai.multi.cctv.review_worker import ReviewWorkerConfig  # noqa: E402
from proctoring_ai.multi.cctv.suspects import SuspectWindow  # noqa: E402


def replay(suspects: Path, cache: Path, *, config: ReviewRuleConfig | None = None,
           max_frames: int | None = None, allow_missing: bool = False,
           split_ms: int | None = None, live: ReviewWorkerConfig | None = None) -> list[dict[str, Any]]:
    """``split_ms`` replays the live deep tier: each window is cut into pieces of at most that length, as
    ``SuspectTracker.poll`` hands them out, each piece gets its own frames (``select_offsets`` with the
    ``ReviewWorkerConfig`` defaults, ``max_frames`` frames at most) and the rules also see the frames
    analysed for the previous piece of the same seat (``ReviewHistory``)."""

    windows = [SuspectWindow.from_json(json.loads(line))
               for line in suspects.read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = load_cache(cache)
    dedupe = ReviewDeduper()
    records: list[dict[str, Any]] = []
    pieces: list[tuple[SuspectWindow, list[ReviewFrame]]] = []
    for window in windows:
        analysed = frames.get(window_key(window))
        if analysed is None:
            if allow_missing:
                continue
            raise KeyError(f"cache không có đoạn {window_key(window)}")
        pieces += [(piece, analysed) for piece in _split(window, split_ms)]
    pieces.sort(key=lambda item: item[0].start_ms)
    live, history = live or ReviewWorkerConfig(), ReviewHistory()
    for window, analysed in pieces:
        context: list[ReviewFrame] = []
        if split_ms is not None:
            context = history.before(window)
            after = context[-1].offset_ms if context else -1
            analysed = [frame for frame in analysed if window.start_ms - live.warmup_ms <= frame.offset_ms
                        <= window.end_ms and frame.offset_ms > after]
        if max_frames is not None:
            keep = set(select_offsets(window, [frame.offset_ms for frame in analysed], per_s=live.per_s,
                                      warmup_frames=0 if context else live.warmup_frames,
                                      warmup_ms=live.warmup_ms, max_frames=max_frames))
            analysed = [frame for frame in analysed if frame.offset_ms in keep]
        if split_ms is not None:
            history.add(window, analysed)
        records += dedupe.admit(review_window(window, [*context, *analysed], config))
    return records


def _split(window: SuspectWindow, split_ms: int | None) -> list[SuspectWindow]:
    if split_ms is None or window.end_ms - window.start_ms <= split_ms:
        return [window]
    out = []
    for start in range(window.start_ms, window.end_ms, split_ms):
        end = min(window.end_ms, start + split_ms)
        peak = window.peak_ms if window.peak_ms is not None and start <= window.peak_ms <= end else None
        out.append(replace(window, start_ms=start, end_ms=end, peak_ms=peak))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suspects", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--max-frames", type=int, default=None, help="số khung tối đa mỗi đoạn")
    parser.add_argument("--split-ms", type=int, default=None,
                        help="cắt đoạn dài thành từng khúc như tầng kỹ chạy song song (10000)")
    parser.add_argument("--allow-missing", action="store_true", help="bỏ qua đoạn chưa có trong cache")
    parser.add_argument("--events", type=Path, default=None, help="ghi sự kiện ra file JSONL")
    args = parser.parse_args(argv)
    records = replay(args.suspects, args.cache, max_frames=args.max_frames, allow_missing=args.allow_missing,
                     split_ms=args.split_ms)
    if args.events is not None:
        args.events.write_text("".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in records),
                               encoding="utf-8")
    score = score_review(records, load_verdicts(args.labels))
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
