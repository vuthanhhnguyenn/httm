"""Confidence level of deep-tier events: red (nearly certain) or yellow (the invigilator should look).

Rules live in ``configs/confidence.yaml``. The evidence fields an ``opened`` record carries
(``evidence_kind``, ``raised``, ``confidence``) were computed by ``review_rules``; nothing is recomputed here.
A ``merged`` repeat takes the level of the alert it was folded into (``merged_into``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

YELLOW_DEFAULT = "chưa đủ bằng chứng chắc chắn"


@dataclass(frozen=True, slots=True)
class ConfidenceRules:
    """Per event type: ``{"always": level}`` or ``{"red_if": {condition: value}}``."""

    rules: dict[str, dict[str, Any]]


def load_confidence(path: Path) -> ConfidenceRules:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules: dict[str, dict[str, Any]] = {}
    for kind, rule in payload.items():
        if not isinstance(rule, dict) or not ({"always", "red_if"} & rule.keys()):
            raise ValueError(f"{kind}: cần `always` hoặc `red_if`")
        if rule.get("always") not in (None, "red", "yellow"):
            raise ValueError(f"{kind}: always chỉ được là red hoặc yellow")
        rules[str(kind)] = rule
    return ConfidenceRules(rules)


def _fast_agrees(event: dict[str, Any], fast: Sequence[dict[str, Any]], within_s: float) -> bool:
    at = float(event.get("video_offset_ms", 0))
    return any(str(item.get("seat_id")) == str(event.get("seat_id"))
               and abs(float(item.get("video_offset_ms", 0)) - at) <= within_s * 1000 for item in fast)


def _judge(event: dict[str, Any], fast: Sequence[dict[str, Any]], rules: ConfidenceRules) -> tuple[str, str]:
    kind = str(event.get("type"))
    rule = rules.rules.get(kind)
    if rule is None:
        return "yellow", f"loại {kind} chưa có luật tin cậy"
    if "always" in rule:
        level = str(rule["always"])
        if level == "red":
            return "red", "luật chắc chắn cho loại này"
        return "yellow", "nghi vấn chưa rõ" if kind == "UNCLASSIFIED_SUSPECT" else "cần người xem lại"
    cond = rule["red_if"]
    if "evidence_kind" in cond:
        evidence = event.get("evidence_kind")
        if evidence in cond["evidence_kind"]:
            return "red", "có vật giữa hai tay" if evidence == "object_between" else "vật chuyển từ ghế này sang ghế kia"
        return "yellow", "chỉ thấy tay chạm tay"
    if "raised" in cond:
        confidence = float(event.get("confidence", 0.0))
        if bool(event.get("raised")) != bool(cond["raised"]):
            return "yellow", "điện thoại chưa giơ cao hơn mép bàn"
        if confidence < float(cond.get("min_confidence", 0.0)):
            return "yellow", f"điện thoại giơ cao nhưng tin cậy thấp {confidence:.2f}".replace(".", ",")
        return "red", f"điện thoại giơ cao, tin cậy {confidence:.2f}".replace(".", ",")
    if "fast_agrees_s" in cond:
        within = float(cond["fast_agrees_s"])
        if _fast_agrees(event, fast, within):
            return "red", f"tầng nhanh cũng báo cùng ghế (lệch tối đa {within:g} giây)"
        return "yellow", "tầng nhanh không báo cùng lúc"
    return "yellow", YELLOW_DEFAULT


def assign_levels(review: list[dict[str, Any]], fast: list[dict[str, Any]],
                  rules: ConfidenceRules) -> list[dict[str, Any]]:
    """Copies of the ``opened`` / ``merged`` records with ``level`` and ``level_reason`` (Vietnamese) added."""

    judged = {str(item.get("id")): _judge(item, fast, rules) for item in review if item.get("action") == "opened"}
    out: list[dict[str, Any]] = []
    for item in review:
        action = item.get("action")
        if action == "opened":
            level, reason = judged[str(item.get("id"))]
            out.append({**item, "level": level, "level_reason": reason})
        elif action == "merged":
            # A merged repeat is shown as part of the alert it was folded into: same level.
            level, reason = judged.get(str(item.get("merged_into")), ("yellow", "không tìm thấy cảnh báo gốc"))
            out.append({**item, "level": level, "level_reason": reason})
    return out


__all__ = ["ConfidenceRules", "assign_levels", "load_confidence"]
