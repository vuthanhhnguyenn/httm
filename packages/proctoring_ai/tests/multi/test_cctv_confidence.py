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


def test_merged_takes_level_of_its_alert_and_other_records_pass_through():
    first = _event("LOOK_AT_NEIGHBOR")
    merged = {**_event("LOOK_AT_NEIGHBOR", 12000), "action": "merged", "merged_into": first["id"]}
    orphan = {**_event("LOOK_AT_NEIGHBOR", 13000), "action": "merged", "merged_into": "gone"}
    out = assign_levels([merged, first, {"action": "closed", "id": first["id"]}, orphan], [], RULES)
    assert [r["action"] for r in out] == ["merged", "opened", "closed", "merged"]
    assert [r.get("level") for r in out] == ["red", "red", None, "yellow"]
    assert out[2] == {"action": "closed", "id": first["id"]}
    assert first.get("level") is None  # inputs are copied, not mutated


def test_boundaries_are_red():
    assert _level(_event("PHONE_DETECTED", raised=True, confidence=0.6)) == "red"
    assert _level(_event("PHONE_DETECTED", raised=True, confidence=None)) == "yellow"
    fast = {"seat_id": "s1", "video_offset_ms": 13000, "type": "TURN_AROUND"}
    assert _level(_event("TURN_AROUND"), [fast]) == "red"


def test_reasons():
    (lap,) = assign_levels([_event("LOOK_AT_LAP")], [], RULES)
    assert lap["level_reason"] == "tư thế không tách được, cần xem lại"
    (passing,) = assign_levels([_event("PASSING_OBJECT")], [], RULES)
    assert passing["level"] == "yellow" and passing["level_reason"] == "không rõ bằng chứng"


def test_red_if_conditions_combine_with_and():
    rules = load_confidence_text("X: {red_if: {evidence_kind: [a], fast_agrees_s: 3}}")
    near = [{"seat_id": "s1", "video_offset_ms": 10000}]
    ok = {**_event("X"), "evidence_kind": "a"}
    assert assign_levels([ok], near, rules)[0]["level"] == "red"
    assert assign_levels([ok], [], rules)[0]["level"] == "yellow"
    assert assign_levels([{**ok, "evidence_kind": "b"}], near, rules)[0]["level"] == "yellow"


def load_confidence_text(text):
    import tempfile

    path = Path(tempfile.mkdtemp()) / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return load_confidence(path)


@pytest.mark.parametrize("text", [
    "[1, 2]",
    "X: {}",
    "X: {always: red, red_if: {raised: true}}",
    "X: {always: green}",
    "X: {red_if: {}}",
    "X: {red_if: [raised]}",
    "X: {red_if: {bogus: 1}}",
    "X: {red_if: {evidence_kind: a}}",
    "X: {red_if: {evidence_kind: [1]}}",
    "X: {red_if: {raised: true, min_confidence: high}}",
    "X: {red_if: {fast_agrees_s: soon}}",
    "X: {red_if: {min_confidence: 0.6}}",
    "X: red",
])
def test_bad_config_raises(text):
    with pytest.raises(ValueError):
        load_confidence_text(text)




def _pair(kind: str, start: int, end: int, level: str = "red") -> list[dict]:
    return [{"action": "opened", "id": f"{kind}{start}", "type": kind, "video_offset_ms": start, "level": level,
             "level_reason": "x"},
            {"action": "closed", "id": f"{kind}{start}", "type": kind, "video_offset_ms": end}]


def test_require_sustained_drops_blips_and_demotes_short_reds() -> None:
    from proctoring_ai.multi.cctv.confidence import require_sustained

    records = (_pair("HEADS_TOGETHER", 0, 3000) + _pair("LOOK_AT_NEIGHBOR", 5000, 14000)
               + _pair("LOOK_AT_LAP", 20000, 20400, "yellow") + _pair("PHONE_DETECTED", 30000, 30000, "yellow")
               + [{"action": "merged", "id": "m", "merged_into": "LOOK_AT_LAP20000", "video_offset_ms": 21000}])
    out = require_sustained(records)
    opened = {r["type"]: r["level"] for r in out if r["action"] == "opened"}
    assert opened == {"HEADS_TOGETHER": "yellow", "LOOK_AT_NEIGHBOR": "red", "PHONE_DETECTED": "yellow"}
    assert not any(r.get("id") == "LOOK_AT_LAP20000" or r.get("merged_into") == "LOOK_AT_LAP20000" for r in out)
    demoted = next(r for r in out if r["type"] == "HEADS_TOGETHER" and r["action"] == "opened")
    assert "3 giây" in demoted["level_reason"]
