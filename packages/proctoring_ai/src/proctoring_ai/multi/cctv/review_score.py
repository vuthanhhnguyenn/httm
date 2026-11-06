"""Score deep-tier events against Nguyên's verdicts (``data/labels/<clip>-review.yaml``).

A verdict is one picture Nguyên judged: event type, time, the heads of the seats in it, true or false.
An ``opened`` event matches it when the type is the same, it is at most ``max_dt_s`` away and its head is
within ``max_head`` of one of the verdict's heads. ``false_opened`` counts false verdicts hit (one per
picture, as Nguyên graded them); ``false_events`` counts every event that hit one.

``true_covered`` / ``false_covered`` count what the invigilator actually sees at the picture's moment: an
alert of that type on that head that is open then (from ``opened`` to its ``closed``, give or take
``max_dt_s``), or a ``merged`` repeat the deduper folded into an alert already on screen. A behaviour that
goes on (the same boy with his hands in his lap at 51.2, 59.0 and 63.0 s) is one alert, not three.

``true_red`` / ``false_red`` count the pictures (true / false verdicts) that a red alert covers in the same
sense; events need ``level`` from ``confidence.assign_levels`` (without it both are 0).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

IGNORED = {"UNCLASSIFIED_SUSPECT", "REVIEW_SKIPPED"}


@dataclass(frozen=True, slots=True)
class Verdict:
    type: str
    t_s: float
    heads: tuple[tuple[float, float], ...]
    verdict: bool


def load_verdicts(path: Path) -> list[Verdict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Verdict(str(item["type"]), float(item["t_s"]), tuple((float(h[0]), float(h[1])) for h in item["heads"]),
                    bool(item["verdict"])) for item in payload.get("verdicts", [])]


def _matches(event: dict[str, Any], verdict: Verdict, max_dt_s: float, max_head: float,
             end_s: float | None = None) -> bool:
    head = event.get("head")
    if event.get("type") != verdict.type or head is None:
        return False
    start_s = float(event.get("video_offset_ms", 0)) / 1000
    end_s = start_s if end_s is None else max(start_s, end_s)
    if not start_s - max_dt_s <= verdict.t_s <= end_s + max_dt_s:
        return False
    return any(math.dist((head[0], head[1]), h) <= max_head for h in verdict.heads)


def score_review(records: Iterable[dict[str, Any]], verdicts: Sequence[Verdict], *, max_dt_s: float = 1.5,
                 max_head: float = 0.05) -> dict[str, Any]:
    opened = [item for item in records if item.get("action") == "opened" and item.get("type") not in IGNORED]
    hit: set[int] = set()
    false_events = unjudged = 0
    for event in opened:
        found = [i for i, verdict in enumerate(verdicts) if _matches(event, verdict, max_dt_s, max_head)]
        if not found:
            unjudged += 1
            continue
        hit.update(found)
        if any(not verdicts[i].verdict for i in found):
            false_events += 1
    trues = [i for i, verdict in enumerate(verdicts) if verdict.verdict]
    ends = {str(item.get("id")): float(item.get("video_offset_ms", 0)) / 1000 for item in records
            if item.get("action") == "closed"}
    shown = [(item, ends.get(str(item.get("id"))) if item.get("action") == "opened" else None) for item in records
             if item.get("action") in ("opened", "merged") and item.get("type") not in IGNORED]
    covered = {i for i, verdict in enumerate(verdicts)
               if any(_matches(event, verdict, max_dt_s, max_head, end) for event, end in shown)} | hit
    # Red alerts on screen (opened, still open, or a merged repeat) at each picture, after ``assign_levels``.
    red = {i for i, verdict in enumerate(verdicts)
           if any(event.get("level") == "red" and _matches(event, verdict, max_dt_s, max_head, end)
                  for event, end in shown)}
    return {"true_red": sum(1 for i in red if verdicts[i].verdict),
            "false_red": sum(1 for i in red if not verdicts[i].verdict), "true_found": sum(1 for i in trues if i in hit), "true_total": len(trues),
            "true_covered": sum(1 for i in trues if i in covered),
            "false_covered": sum(1 for i in covered if not verdicts[i].verdict),
            "covered_only": [f"{verdicts[i].type}@{verdicts[i].t_s}" for i in trues if i in covered - hit],
            "false_opened": sum(1 for i in hit if not verdicts[i].verdict), "false_events": false_events,
            "unjudged": unjudged,
            "missed": [f"{verdicts[i].type}@{verdicts[i].t_s}" for i in trues if i not in hit]}


__all__ = ["Verdict", "load_verdicts", "score_review"]
