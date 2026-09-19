from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    async def update(self, observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(item, runtime_track_id=index + 1) for index, item in enumerate(observations)]

