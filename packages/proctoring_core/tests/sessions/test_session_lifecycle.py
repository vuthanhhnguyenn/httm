from __future__ import annotations

from uuid import uuid4

import pytest

from proctoring_core.sessions.models import ExamSession, SessionStateError
from proctoring_core.types import SessionMode, SessionStatus


def make_session() -> ExamSession:
    return ExamSession.create(name="Practice", mode=SessionMode.SINGLE_PERSON, camera_source="fake", policy_id=uuid4(), policy_snapshot={}, config_version_id=uuid4(), config_snapshot={}, created_by="operator")


def test_session_lifecycle_and_idempotency() -> None:
    session = make_session()
    first = session.start("start-1", "request-1")
    assert first.status is SessionStatus.STARTING
    assert session.start("start-1", "request-2").replayed
    session.mark_running()
    stopped = session.stop("stop-1", "request-3")
    assert stopped.status is SessionStatus.STOPPING
    session.mark_stopped()
    assert session.status is SessionStatus.STOPPED
    with pytest.raises(SessionStateError) as error:
        session.start("start-2", "request-4")
    assert error.value.code == "SESSION_TERMINAL"

