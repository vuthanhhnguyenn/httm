from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("pydantic")

from proctoring_core.config.models import ConfigVersionStatus, ExamPolicy, SystemConfigVersion, canonical_sha256


def test_config_hash_is_canonical_and_policy_constraints_are_enforced() -> None:
    content = {"runtime": {"timezone": "UTC"}, "risk": {"maximum_score": 100}}
    version = SystemConfigVersion.create(content, version=1, status=ConfigVersionStatus.ACTIVE, created_by="admin")
    assert version.sha256 == canonical_sha256(content)
    assert len(version.sha256) == 64
    policy = ExamPolicy.create(
        {
            "name": "Allowed scratch paper",
            "allow_book": False,
            "allow_scratch_paper": True,
            "allow_phone": False,
            "allowed_materials": ["calculator"],
            "evidence_frame_enabled": True,
            "evidence_clip_enabled": False,
            "pre_event_seconds": 3,
            "post_event_seconds": 3,
            "retention_days": 2,
        },
        policy_id=uuid4(),
        created_by="admin",
    )
    assert policy.retention_days >= 1
    assert policy.pre_event_seconds <= 30


@pytest.mark.parametrize("field,value", [("name", ""), ("pre_event_seconds", 31), ("post_event_seconds", -1), ("retention_days", 0)])
def test_policy_rejects_invalid_constraints(field: str, value: object) -> None:
    payload = {
        "name": "Policy",
        "allow_book": False,
        "allow_scratch_paper": True,
        "allow_phone": False,
        "evidence_frame_enabled": True,
        "evidence_clip_enabled": False,
        "pre_event_seconds": 3,
        "post_event_seconds": 3,
        "retention_days": 30,
    }
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        ExamPolicy.create(payload)
