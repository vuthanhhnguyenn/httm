"""Chấm bộ cảnh báo (``events.jsonl``) theo nhãn của Nguyên: bắt được bao nhiêu, đúng bao nhiêu.

Cảnh báo = bản ghi ``opened`` có ``head``, loại khác ``UNCLASSIFIED_SUSPECT`` / ``REVIEW_SKIPPED``. Nó phủ một
nhãn khi đầu cách một đầu của nhãn <= ``max_head`` và thời điểm nhãn nằm trong [mở - pad_s, đóng + pad_s]
(thiếu bản ghi ``closed`` thì đóng = mở + 3 s). Không xét loại. Đúng nếu phủ ít nhất một nhãn đúng, nhầm nếu
chỉ phủ nhãn nhầm, chưa chấm nếu không phủ nhãn nào.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from proctoring_ai.multi.cctv.review_score import IGNORED, Verdict

DEFAULT_OPEN_S = 3.0


def alert_windows(records: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], float, float]]:
    """Mỗi cảnh báo (bản ghi ``opened`` có ``head``, không thuộc ``IGNORED``) cùng khoảng [mở, đóng] tính bằng giây."""

    records = list(records)
    ends = {str(r.get("id")): float(r.get("video_offset_ms", 0)) / 1000 for r in records
            if r.get("action") == "closed"}
    out = []
    for alert in records:
        if alert.get("action") == "opened" and alert.get("head") and alert.get("type") not in IGNORED:
            start = float(alert.get("video_offset_ms", 0)) / 1000
            out.append((alert, start, max(start, ends.get(str(alert.get("id")), start + DEFAULT_OPEN_S))))
    return out


def covered_labels(alert: dict[str, Any], start: float, end: float, verdicts: Sequence[Verdict], *,
                   max_head: float = 0.06, pad_s: float = 2.0) -> set[int]:
    """Chỉ số các nhãn mà cảnh báo này phủ."""

    head = (float(alert["head"][0]), float(alert["head"][1]))
    return {i for i, v in enumerate(verdicts)
            if start - pad_s <= v.t_s <= end + pad_s and any(math.dist(head, h) <= max_head for h in v.heads)}


def score_alerts(records: Iterable[dict[str, Any]], verdicts: Sequence[Verdict], *, max_head: float = 0.06,
                 pad_s: float = 2.0) -> dict[str, Any]:
    windows = alert_windows(records)
    alerts = [alert for alert, _, _ in windows]
    hit: set[int] = set()
    correct = judged = 0
    by_type: dict[str, dict[str, int]] = {}
    for alert, start, end in windows:
        found = covered_labels(alert, start, end, verdicts, max_head=max_head, pad_s=pad_s)
        hit |= found
        row = by_type.setdefault(str(alert.get("type")), {"alerts": 0, "correct": 0})
        row["alerts"] += 1
        if found:
            judged += 1
        if any(verdicts[i].verdict for i in found):
            correct += 1
            row["correct"] += 1
    return {
        "caught": sum(1 for i in hit if verdicts[i].verdict),
        "true_total": sum(1 for v in verdicts if v.verdict),
        "alerts": len(alerts),
        "alerts_correct": correct,
        "alerts_judged": judged,
        "false_flagged": sum(1 for i in hit if not verdicts[i].verdict),
        "false_total": sum(1 for v in verdicts if not v.verdict),
        "red": sum(1 for a in alerts if a.get("level") == "red"),
        "by_type": by_type,
    }
