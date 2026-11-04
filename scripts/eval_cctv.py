"""Chấm events.jsonl của chế độ CCTV theo file nhãn tay: đúng / sót / báo sai.

    python scripts/eval_cctv.py storage/multi/cctv3/events.jsonl data/labels/EPf9O7UXRoM.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SLACK_S = 2.0  # sự kiện mở sớm/muộn hơn nhãn tối đa ngần này vẫn tính là khớp


def _inside(head: list[float] | None, region: list[float], margin: float = 0.02) -> bool:
    if head is None:
        return False
    x0, y0, x1, y1 = region
    return x0 - margin <= head[0] <= x1 + margin and y0 - margin <= head[1] <= y1 + margin


def score(events: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    opened = [e for e in events if e.get("action") == "opened"]
    closed = {e.get("id"): e["video_offset_ms"] / 1000 for e in events if e.get("action") == "closed"}
    targets = [label for label in labels if label.get("kind") != "none"]
    hits: dict[str, list[dict[str, Any]]] = {label["id"]: [] for label in targets}
    false: list[dict[str, Any]] = []
    for event in opened:
        start = event["video_offset_ms"] / 1000
        end = closed.get(event.get("id"), start)  # still open at the end of the clip: count the opening only
        matched = [label for label in targets if start <= label["end_s"] + SLACK_S and end >= label["start_s"] - SLACK_S
                   and _inside(event.get("head"), label["region"])]
        for label in matched:
            hits[label["id"]].append(event)
        if not matched:
            false.append(event)
    by_behavior: dict[str, dict[str, int]] = {}
    for label in targets:
        counts = by_behavior.setdefault(label.get("behavior", "look"), {"found": 0, "total": 0})
        counts["total"] += 1
        counts["found"] += 1 if hits[label["id"]] else 0
    return {"found": [key for key, items in hits.items() if items],
            "missed": [key for key, items in hits.items() if not items],
            "false": false, "events": len(opened), "by_behavior": by_behavior}


def coverage(windows: list[dict[str, Any]], labels: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    """Which labels the fast tier put inside a suspect window (same time, seat centre in the label region),
    and what share of the clip the windows cover (overlaps counted once)."""

    targets = [label for label in labels if label.get("kind") != "none"]
    covered: list[str] = []
    for label in targets:
        for window in windows:
            box = window["bbox"]
            centre = [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2]
            start, end = window["start_ms"] / 1000, window["end_ms"] / 1000
            if start <= label["end_s"] and end >= label["start_s"] and _inside(centre, label["region"]):
                covered.append(label["id"])
                break
    spans = sorted((w["start_ms"] / 1000, w["end_ms"] / 1000) for w in windows)
    total, reach = 0.0, float("-inf")
    for start, end in spans:
        start = max(start, reach)
        if end > start:
            total += end - start
        reach = max(reach, end)
    return {"covered": covered, "uncovered": [label["id"] for label in targets if label["id"] not in covered],
            "suspect_ratio": total / duration_s if duration_s > 0 else 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--suspects", type=Path, help="suspects.jsonl của tầng nhanh: đo độ phủ")
    parser.add_argument("--duration", type=float, default=0.0, help="độ dài clip (giây), cho --suspects")
    args = parser.parse_args(argv)
    events = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))["labels"]
    result = score(events, labels)
    total = len(result["found"]) + len(result["missed"])
    print(f"Bắt được {len(result['found'])}/{total} nhãn: {', '.join(result['found']) or '-'}")
    print(f"Sót: {', '.join(result['missed']) or '-'}")
    for behavior, counts in sorted(result["by_behavior"].items()):
        print(f"  {behavior}: {counts['found']}/{counts['total']}")
    print(f"Báo sai: {len(result['false'])}/{result['events']} sự kiện")
    if args.suspects is not None:
        windows = [json.loads(line) for line in args.suspects.read_text(encoding="utf-8").splitlines() if line.strip()]
        cover = coverage(windows, labels, args.duration)
        print(f"Tầng nhanh khoanh trúng {len(cover['covered'])}/{len(cover['covered']) + len(cover['uncovered'])} "
              f"nhãn, đoạn nghi vấn chiếm {cover['suspect_ratio']:.0%} clip; sót: {', '.join(cover['uncovered']) or '-'}")
    for event in result["false"]:
        print(f"  {event['video_offset_ms'] / 1000:6.1f}s  #{event.get('runtime_track_id')}  {event['type']}  "
              f"đầu {event.get('head')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
