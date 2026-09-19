from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class FakeDetector:
    def __init__(self, detections: Sequence[Mapping[str, Any]] = ()) -> None:
        self.detections = [dict(item) for item in detections]
        self.calls = 0

    async def detect(self, _frame: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(item) for item in self.detections]

