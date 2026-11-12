"""Confidence level of deep-tier events: red (nearly certain) or yellow (the invigilator should look).

Rules live in ``configs/confidence.yaml``. The evidence fields an ``opened`` record carries
(``evidence_kind``, ``raised``, ``confidence``) were computed by ``review_rules``; nothing is recomputed here.
A ``merged`` repeat takes the level of the alert it was folded into (``merged_into``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

YELLOW_DEFAULT = "chưa đủ bằng chứng chắc chắn"


@dataclass(frozen=True, slots=True)
class ConfidenceRules:
    """Per event type: ``{"always": level}`` or ``{"red_if": {condition: value}}``."""

    rules: dict[str, dict[str, Any]]


RED_IF_KEYS = {"evidence_kind", "raised", "min_confidence", "fast_agrees_s"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(kind: str, rule: Any) -> None:
    if not isinstance(rule, dict):
        raise ValueError(f"{kind}: cần một bảng có `always` hoặc `red_if`")
    if ("always" in rule) == ("red_if" in rule):
        raise ValueError(f"{kind}: cần đúng một trong `always` và `red_if`")
    if "always" in rule:
        if rule["always"] not in ("red", "yellow"):
            raise ValueError(f"{kind}: always chỉ được là red hoặc yellow")
        return
    cond = rule["red_if"]
    if not isinstance(cond, dict) or not cond:
        raise ValueError(f"{kind}: red_if phải là bảng không rỗng")
    unknown = set(cond) - RED_IF_KEYS
    if unknown:
        raise ValueError(f"{kind}: red_if có khoá lạ {sorted(unknown)}; được phép {sorted(RED_IF_KEYS)}")
    kinds = cond.get("evidence_kind")
    if "evidence_kind" in cond and not (isinstance(kinds, list) and all(isinstance(k, str) for k in kinds)):
        raise ValueError(f"{kind}: evidence_kind phải là danh sách chuỗi")
    for key in ("min_confidence", "fast_agrees_s"):
        if key in cond and not _is_number(cond[key]):
            raise ValueError(f"{kind}: {key} phải là số")
    if "min_confidence" in cond and "raised" not in cond:
        raise ValueError(f"{kind}: min_confidence cần đi cùng raised")


def load_confidence(path: Path) -> ConfidenceRules:
    """Read and validate the YAML. All conditions of one ``red_if`` must hold together (AND)."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: cấp ngoài cùng phải là bảng loại sự kiện")
    for kind, rule in payload.items():
        _check(str(kind), rule)
    return ConfidenceRules({str(kind): rule for kind, rule in payload.items()})


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
        if rule["always"] == "red":
            return "red", "luật chắc chắn cho loại này"
        return "yellow", {"UNCLASSIFIED_SUSPECT": "nghi vấn chưa rõ",
                          "LOOK_AT_LAP": "tư thế không tách được, cần xem lại"}.get(kind, "cần người xem lại")
    cond = rule["red_if"]
    reasons: list[str] = []
    if "evidence_kind" in cond:
        evidence = event.get("evidence_kind")
        if evidence not in cond["evidence_kind"]:
            return "yellow", "không rõ bằng chứng" if evidence is None else "chỉ thấy tay chạm tay"
        reasons.append("có vật giữa hai tay" if evidence == "object_between" else "vật chuyển từ ghế này sang ghế kia")
    if "raised" in cond:
        confidence = float(event.get("confidence") or 0.0)
        shown = f"{confidence:.2f}".replace(".", ",")
        if bool(event.get("raised")) != bool(cond["raised"]):
            return "yellow", "điện thoại chưa giơ cao hơn mép bàn"
        if confidence < float(cond.get("min_confidence", 0.0)):
            return "yellow", f"điện thoại giơ cao nhưng tin cậy thấp {shown}"
        reasons.append(f"điện thoại giơ cao, tin cậy {shown}")
    if "fast_agrees_s" in cond:
        within = float(cond["fast_agrees_s"])
        if not _fast_agrees(event, fast, within):
            return "yellow", "tầng nhanh không báo cùng lúc"
        reasons.append(f"tầng nhanh cũng báo cùng ghế (lệch tối đa {within:g} giây)")
    return "red", "; ".join(reasons)


def assign_levels(review: list[dict[str, Any]], fast: list[dict[str, Any]],
                  rules: ConfidenceRules) -> list[dict[str, Any]]:
    """Copies of all records, in order; ``opened`` / ``merged`` ones get ``level`` and ``level_reason`` (Vietnamese)."""

    judged = {str(item.get("id")): _judge(item, fast, rules) for item in review if item.get("action") == "opened"}
    out: list[dict[str, Any]] = []
    for item in review:
        action = item.get("action")
        if action == "opened":
            level, reason = judged[str(item.get("id"))]
        elif action == "merged":
            # A merged repeat is shown as part of the alert it was folded into: same level.
            level, reason = judged.get(str(item.get("merged_into")), ("yellow", "không tìm thấy cảnh báo gốc"))
        else:
            out.append(dict(item))  # closed, skipped, ...: passed through, no level
            continue
        out.append({**item, "level": level, "level_reason": reason})
    return out


def require_sustained(records: list[dict[str, Any]], *, min_ms: int = 1000, red_ms: int = 2000,
                      keep: Sequence[str] = ("PHONE_DETECTED",),
                      min_ms_by_type: Mapping[str, int] | None = None,
                      hidden: Sequence[str] = ()) -> list[dict[str, Any]]:
    """For the all-seats deep tier, where every seat is watched and blips add up: an alert shorter than
    ``min_ms`` (or its type's entry in ``min_ms_by_type``) is dropped (with its closed and merged rows) unless its
    type is in ``keep``; a red alert shorter than ``red_ms`` becomes yellow. ``red_ms`` is 2 s by Nguyên's
    decision (2026-11-11); it was 3.2 s, measured on clip 1 (true reds 3.4 s and 9 s, false ones 3.0 s or less).
    An alert whose type is in ``hidden`` is dropped whatever its length (types judged nearly always wrong)."""

    ends = {str(item.get("id")): int(item.get("video_offset_ms", 0)) for item in records
            if item.get("action") == "closed"}
    drop: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in records:
        if item.get("action") == "opened":
            key = str(item.get("id"))
            length = ends.get(key, int(item.get("video_offset_ms", 0))) - int(item.get("video_offset_ms", 0))
            least = (min_ms_by_type or {}).get(str(item.get("type")), min_ms)
            if item.get("type") in hidden or (length < least and item.get("type") not in keep):
                drop.add(key)
                continue
            if item.get("level") == "red" and length < red_ms:
                item["level"] = "yellow"  # in place: evidence pictures look the records up by identity
                item["level_reason"] = f"đỏ cần kéo dài ít nhất {red_ms / 1000:g} giây; chỉ {length / 1000:.1f} s"
        out.append(item)
    return [item for item in out if str(item.get("id")) not in drop and str(item.get("merged_into")) not in drop]


__all__ = ["ConfidenceRules", "assign_levels", "load_confidence", "require_sustained"]
