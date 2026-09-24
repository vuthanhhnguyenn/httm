from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from ..types import RiskLevel, RiskScore, ensure_utc


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    session_id: UUID
    score_before: float
    score_after: float
    level_after: RiskLevel
    reason: str
    contributing_event_ids: tuple[UUID, ...] = ()
    reason_codes: tuple[str, ...] = ()
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: UUID = field(default_factory=uuid4)
    track_id: UUID | None = None
    cumulative_score: float = 0.0
    peak_score: float = 0.0
    event_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_before", RiskScore(self.score_before).value)
        object.__setattr__(self, "score_after", RiskScore(self.score_after).value)
        object.__setattr__(self, "level_after", RiskLevel(self.level_after))
        object.__setattr__(self, "occurred_at", ensure_utc(self.occurred_at))
