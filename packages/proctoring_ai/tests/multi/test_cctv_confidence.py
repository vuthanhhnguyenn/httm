from __future__ import annotations

from pathlib import Path

import pytest
from proctoring_ai.multi.cctv.confidence import assign_levels, load_confidence

RULES = load_confidence(Path(__file__).resolve().parents[4] / "configs" / "confidence.yaml")


def _event(kind: str, ms: int = 10000, seat: str = "s1", **extra) -> dict:
    return {"action": "opened", "id": f"{kind}-{ms}", "type": kind, "seat_id": seat, "video_offset_ms": ms, **extra}


def _level(event: dict, fast: list[dict] | None = None) -> str:
    (out,) = assign_levels([event], fast or [], RULES)
    assert out["level_reason"]
    return out["level"]


@pytest.mark.parametrize("kind", ["LOOK_AT_NEIGHBOR", "HEADS_TOGETHER", "DOCUMENT_DETECTED"])
def test_always_red(kind):
    assert _level(_event(kind)) == "red"


def test_passing_red_with_object_or_transfer_yellow_hands_only():
    assert _level(_event("PASSING_OBJECT", evidence_kind="object_between")) == "red"
    assert _level(_event("PASSING_OBJECT", evidence_kind="transfer")) == "red"
    (out,) = assign_levels([_event("PASSING_OBJECT", evidence_kind="hands_only")], [], RULES)
    assert out["level"] == "yellow" and out["level_reason"] == "chỉ thấy tay chạm tay"


def test_phone_red_only_when_raised_and_confident():
    (out,) = assign_levels([_event("PHONE_DETECTED", raised=True, confidence=0.82)], [], RULES)
    assert out["level"] == "red" and out["level_reason"] == "điện thoại giơ cao, tin cậy 0,82"
    assert _level(_event("PHONE_DETECTED", raised=True, confidence=0.59)) == "yellow"
    assert _level(_event("PHONE_DETECTED", raised=False, confidence=0.9)) == "yellow"
    assert _level(_event("PHONE_DETECTED")) == "yellow"


@pytest.mark.parametrize("kind", ["TURN_AROUND", "HEAD_TURN_LEFT", "HEAD_TURN_RIGHT"])
def test_turn_red_when_fast_tier_agrees_within_3s(kind):
    near = {"seat_id": "s1", "video_offset_ms": 12900, "type": "HEAD_TURN_LEFT"}
    assert _level(_event(kind), [near]) == "red"
    assert _level(_event(kind), [{**near, "video_offset_ms": 13100}]) == "yellow"
    assert _level(_event(kind), [{**near, "seat_id": "s2"}]) == "yellow"
    assert _level(_event(kind), []) == "yellow"


def test_look_at_lap_always_yellow():
    assert _level(_event("LOOK_AT_LAP")) == "yellow"


def test_unclassified_always_yellow():
    (out,) = assign_levels([_event("UNCLASSIFIED_SUSPECT")], [], RULES)
    assert out["level"] == "yellow" and out["level_reason"] == "nghi vấn chưa rõ"


def test_unknown_type_is_yellow():
    assert _level(_event("SOMETHING_NEW")) == "yellow"


def test_merged_takes_level_of_its_alert_and_closed_is_dropped():
    first = _event("LOOK_AT_NEIGHBOR")
    merged = {**_event("LOOK_AT_NEIGHBOR", 12000), "action": "merged", "merged_into": first["id"]}
    orphan = {**_event("LOOK_AT_NEIGHBOR", 13000), "action": "merged", "merged_into": "gone"}
    out = assign_levels([merged, first, {"action": "closed", "id": first["id"]}, orphan], [], RULES)
    assert [(r["action"], r["level"]) for r in out] == [("merged", "red"), ("opened", "red"), ("merged", "yellow")]
    assert first.get("level") is None  # inputs are copied, not mutated
