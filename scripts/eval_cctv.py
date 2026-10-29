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
    targets = [label for label in labels if label.get("kind") != "none"]
    hits: dict[str, list[dict[str, Any]]] = {label["id"]: [] for label in targets}
    false: list[dict[str, Any]] = []
    for event in opened:
        t = event["video_offset_ms"] / 1000
        matched = [label for label in targets if label["start_s"] - SLACK_S <= t <= label["end_s"] + SLACK_S
                   and _inside(event.get("head"), label["region"])]
        for label in matched:
            hits[label["id"]].append(event)
        if not matched:
            false.append(event)
    return {"found": [key for key, items in hits.items() if items],
            "missed": [key for key, items in hits.items() if not items],
            "false": false, "events": len(opened)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args(argv)
    events = [json.loads(line) for line in args.events.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))["labels"]
    result = score(events, labels)
    total = len(result["found"]) + len(result["missed"])
    print(f"Bắt được {len(result['found'])}/{total} nhãn: {', '.join(result['found']) or '-'}")
    print(f"Sót: {', '.join(result['missed']) or '-'}")
    print(f"Báo sai: {len(result['false'])}/{result['events']} sự kiện")
    for event in result["false"]:
        print(f"  {event['video_offset_ms'] / 1000:6.1f}s  #{event.get('runtime_track_id')}  {event['type']}  "
              f"đầu {event.get('head')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
