from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest


@pytest.mark.contract
def test_websocket_envelope_has_monotonic_contract_fields() -> None:
    from proctoring_api.realtime.schemas import WebSocketEnvelope

    session_id = uuid4()
    envelope = WebSocketEnvelope(
        sequence=1,
        type="connection.ready",
        session_id=session_id,
        occurred_at=datetime.now(UTC),
        data={"status": "ready"},
    )
    assert envelope.schema_version == "1.0"
    assert envelope.sequence == 1
    assert envelope.session_id == session_id

