from __future__ import annotations

from pathlib import Path

import pytest
from proctoring_core.events.review_models import ReviewType
from proctoring_runtime.persistence import JsonlEventRepository
from proctoring_runtime.session_reader import build_views, latest_reviews, review_record

V1_OPENED = {"id": "e1", "action": "opened", "type": "PHONE_DETECTED", "severity": "HIGH", "confidence": 0.8,
             "occurred_at": "2026-10-16T10:00:00+07:00", "label_vi": "Phát hiện điện thoại", "metrics": {},
             "risk_score": 21.0}
V2_OPENED = {**V1_OPENED, "id": "e2", "schema_version": "2.0", "episode_id": "e2", "exam_mode": "computer",
             "supporting_signals": [], "object_trace": {"frame_id": 7, "bbox": {"x": 0.5, "y": 0.6, "width": 0.1,
                                                                                    "height": 0.1}}}


def test_v1_and_v2_lines_build_episode_views(tmp_path: Path) -> None:
    repository = JsonlEventRepository(tmp_path / "events.jsonl")
    for record in (V1_OPENED, V2_OPENED,
                   {"id": "e2", "action": "updated", "metrics": {"duration_ms": 1500}},
                   {"action": "evidence", "event_id": "e2", "status": "available", "relative_path": "sessions/x/e2.jpg"},
                   {"id": "e2", "action": "closed", "occurred_at": "2026-10-16T10:00:05+07:00", "metrics": {}}):
        repository.append(record)

    old, new = build_views(repository.records())

    assert old.exam_mode is None and old.object_trace is None and old.schema_version == "1.0"
    assert new.exam_mode == "computer" and new.object_trace and new.object_trace["frame_id"] == 7
    assert new.duration_ms == 1500 and new.closed_at is not None
    assert new.evidence_path == "sessions/x/e2.jpg"


def test_reviews_are_appended_and_latest_wins_without_touching_event_lines(tmp_path: Path) -> None:
    repository = JsonlEventRepository(tmp_path / "events.jsonl")
    repository.append(V2_OPENED)
    before = repository.path.read_text(encoding="utf-8")
    repository.append(review_record("e2", ReviewType.FALSE_POSITIVE, note="bộ bài", reviewed_at="t1"))
    repository.append(review_record("e2", ReviewType.INCONCLUSIVE, reviewed_at="t2"))

    assert repository.path.read_text(encoding="utf-8").startswith(before)
    assert latest_reviews(repository.records()) == {"e2": ReviewType.INCONCLUSIVE}
    view = build_views(repository.records())[0]
    assert view.reviewed and view.review_note == ""


def test_review_note_length_is_limited() -> None:
    with pytest.raises(ValueError):
        review_record("e1", ReviewType.VALID_SIGNAL, note="x" * 501, reviewed_at="t")
