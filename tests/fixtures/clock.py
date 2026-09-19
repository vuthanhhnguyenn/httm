from __future__ import annotations

from datetime import UTC, datetime, timedelta


class FakeClock:
    """Deterministic UTC and monotonic clock for temporal rule tests."""

    def __init__(self, *, start: datetime | None = None) -> None:
        self._utc = (start or datetime(2026, 1, 1, tzinfo=UTC)).astimezone(UTC)
        self._monotonic_ms = 0

    def now_utc(self) -> datetime:
        return self._utc

    def monotonic_ms(self) -> int:
        return self._monotonic_ms

    def advance(self, milliseconds: int) -> None:
        if milliseconds < 0:
            raise ValueError("milliseconds must be non-negative")
        self._monotonic_ms += milliseconds
        self._utc += timedelta(milliseconds=milliseconds)

