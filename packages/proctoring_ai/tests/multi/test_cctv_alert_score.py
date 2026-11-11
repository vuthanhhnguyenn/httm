from __future__ import annotations

from proctoring_ai.multi.cctv.alert_score import score_alerts
from proctoring_ai.multi.cctv.review_score import Verdict


def opened(event_id, kind, ms, head, level="yellow"):
    return {"action": "opened", "id": event_id, "type": kind, "video_offset_ms": ms, "head": list(head),
            "level": level}


def closed(event_id, ms):
    return {"action": "closed", "id": event_id, "video_offset_ms": ms}


def test_alert_covering_a_true_label_is_correct():
    v = [Verdict("LOOK_AT_LAP", 10.0, ((0.5, 0.5),), True), Verdict("PHONE_DETECTED", 30.0, ((0.2, 0.2),), False)]
    recs = [opened("a", "LOOK_AT_LAP", 9000, (0.52, 0.5)), closed("a", 12000),
            opened("b", "PHONE_DETECTED", 29000, (0.2, 0.21)), closed("b", 29500),
            opened("c", "TURN_AROUND", 50000, (0.9, 0.9))]
    s = score_alerts(recs, v)
    assert (s["caught"], s["true_total"], s["false_flagged"], s["false_total"]) == (1, 1, 1, 1)
    assert (s["alerts"], s["alerts_correct"], s["alerts_judged"]) == (3, 1, 2)


def test_head_farther_than_006_does_not_cover():
    v = [Verdict("LOOK_AT_LAP", 10.0, ((0.5, 0.5),), True)]
    assert score_alerts([opened("a", "LOOK_AT_LAP", 10000, (0.57, 0.5))], v)["caught"] == 0


def test_window_padding_two_seconds_after_close():
    v = [Verdict("LOOK_AT_LAP", 14.0, ((0.5, 0.5),), True)]
    recs = [opened("a", "LOOK_AT_LAP", 9000, (0.5, 0.5)), closed("a", 12000)]
    assert score_alerts(recs, v)["caught"] == 1


def test_ignored_types_and_missing_head_are_not_alerts_and_red_and_by_type_counted():
    v = [Verdict("LOOK_AT_LAP", 10.0, ((0.5, 0.5),), True)]
    recs = [opened("a", "UNCLASSIFIED_SUSPECT", 10000, (0.5, 0.5)), opened("b", "REVIEW_SKIPPED", 10000, (0.5, 0.5)),
            {"action": "opened", "id": "c", "type": "LOOK_AT_LAP", "video_offset_ms": 10000},
            opened("d", "LOOK_AT_LAP", 10000, (0.5, 0.5), level="red")]
    s = score_alerts(recs, v)
    assert (s["alerts"], s["red"], s["caught"]) == (1, 1, 1)
    assert s["by_type"] == {"LOOK_AT_LAP": {"alerts": 1, "correct": 1}}
