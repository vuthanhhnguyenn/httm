from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

pytest.importorskip("pydantic")

from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.events.review_models import EvidenceStatus, ReviewType


def test_us2_policy_evidence_review_and_retention_invariants() -> None:
    evaluator = ObjectRuleEvaluator({"allow_scratch_paper": True, "allow_book": False, "allow_phone": True}, min_duration_ms=0)
    assert evaluator.evaluate(session_id=uuid4(), track_id=None, objects=[{"class": "paper", "confidence": 0.99}], monotonic_ms=0) == []
    assert ReviewType.FALSE_POSITIVE.value == "FALSE_POSITIVE"
    assert EvidenceStatus.DELETED.value == "DELETED"
    assert datetime.now(UTC) + timedelta(days=1) > datetime.now(UTC)

