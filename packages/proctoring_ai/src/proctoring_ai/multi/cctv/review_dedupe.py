"""One deep-tier event per seat and behaviour every ``dedupe_ms``, across overlapping suspect windows.

Overlapping windows of one seat each see the same moment, and the live tier reviews them in priority
order, not video order; a repeat within ``dedupe_ms`` of an event already opened becomes
``{"action": "merged", "merged_into": <first id>}`` (no evidence picture) and its ``closed`` is dropped.
"""

from __future__ import annotations

from typing import Any


class ReviewDeduper:
    def __init__(self, dedupe_ms: int = 10000) -> None:
        self.dedupe_ms = dedupe_ms
        self._opened: dict[tuple[str, str], list[tuple[int, str]]] = {}  # (seat, type) -> [(ms, id)]
        self._merged_ids: set[str] = set()

    def admit(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in records:
            action = item.get("action")
            if action == "closed" and str(item.get("id")) in self._merged_ids:
                continue
            if action != "opened" or "type" not in item:
                out.append(item)
                continue
            key = (str(item.get("seat_id")), str(item["type"]))
            at = int(item.get("video_offset_ms", 0))
            earlier = [(ms, event_id) for ms, event_id in self._opened.get(key, []) if abs(ms - at) < self.dedupe_ms]
            if earlier:
                self._merged_ids.add(str(item.get("id")))
                merged = {k: v for k, v in item.items() if k not in ("evidence", "evidence_status")}
                out.append({**merged, "action": "merged", "merged_into": min(earlier)[1]})
                continue
            self._opened.setdefault(key, []).append((at, str(item.get("id"))))
            out.append(item)
        return out


__all__ = ["ReviewDeduper"]
