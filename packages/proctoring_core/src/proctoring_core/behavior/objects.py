from __future__ import annotations

from typing import Any

from ..types import EventType, Severity
from .temporal_buffer import ConditionState, TemporalCondition, TemporalSample


class ObjectRuleEvaluator:
    def __init__(
        self,
        policy: dict[str, Any] | None = None,
        *,
        min_duration_ms: int = 1000,
        confidence_threshold: float = 0.6,
        phone_confidence_threshold: float | None = None,
        document_confidence_threshold: float | None = None,
        phone_min_duration_ms: int | None = None,
        document_min_duration_ms: int | None = None,
        phone_release_confidence: float | None = None,
        max_gap_ms: int = 0,
        phone_fast_confidence: float = 0.8,
        phone_fast_duration_ms: int = 250,
        phone_fast_min_hits: int = 3,
        phone_max_sample_gap_ms: int = 1200,
        document_max_sample_gap_ms: int = 1200,
        phone_fast_max_sample_gap_ms: int = 350,
        phone_classes: tuple[str, ...] = ("cell phone", "phone", "mobile"),
        document_classes: tuple[str, ...] = ("book",),
        phone_episode_gap_ms: int = 0,
        document_episode_gap_ms: int = 0,
    ) -> None:
        if policy is not None and hasattr(policy, "content"):
            policy = policy.content()
        self.policy = dict(policy or {})
        self.confidence_threshold = confidence_threshold
        self.phone_confidence_threshold = phone_confidence_threshold if phone_confidence_threshold is not None else confidence_threshold
        self.document_confidence_threshold = document_confidence_threshold if document_confidence_threshold is not None else confidence_threshold
        self.phone_release_confidence = phone_release_confidence if phone_release_confidence is not None else self.phone_confidence_threshold
        self._conditions = {
            EventType.PHONE_DETECTED: TemporalCondition(min_duration_ms=phone_min_duration_ms if phone_min_duration_ms is not None else min_duration_ms, max_gap_ms=max_gap_ms, max_sample_gap_ms=phone_max_sample_gap_ms),
            EventType.DOCUMENT_DETECTED: TemporalCondition(min_duration_ms=document_min_duration_ms if document_min_duration_ms is not None else min_duration_ms, max_sample_gap_ms=document_max_sample_gap_ms),
        }
        self._phone_track_ids: set[int] = set()
        self.phone_fast_confidence = max(phone_fast_confidence, self.phone_confidence_threshold)
        self._phone_fast = TemporalCondition(min_duration_ms=phone_fast_duration_ms,
                                             max_sample_gap_ms=phone_fast_max_sample_gap_ms)
        self._classes = {
            EventType.PHONE_DETECTED: {name.lower() for name in phone_classes},
            EventType.DOCUMENT_DETECTED: {name.lower() for name in document_classes},
        }
        self.phone_fast_min_hits = max(2, phone_fast_min_hits)
        self._fast_hits = 0
        self._fast_latched = False
        self._fast_track_id: int | None = None
        # An episode, once active, survives detection dropouts and object-track switches
        # shorter than this gap: one phone in hand is one event (0 keeps the old behaviour).
        self._episode_gap_ms = {EventType.PHONE_DETECTED: max(0, phone_episode_gap_ms),
                                EventType.DOCUMENT_DETECTED: max(0, document_episode_gap_ms)}
        self._episodes: dict[EventType, dict[str, Any]] = {}

    def evaluate(self, *, session_id: Any, track_id: Any, objects: list[dict[str, Any]], monotonic_ms: int,
                 frame_id: int | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for event_type, classes, allowed_key, threshold in (
            (EventType.PHONE_DETECTED, self._classes[EventType.PHONE_DETECTED], "allow_phone", self.phone_confidence_threshold),
            (EventType.DOCUMENT_DETECTED, self._classes[EventType.DOCUMENT_DETECTED], "allow_book", self.document_confidence_threshold),
        ):
            matches = []
            candidate_matches = []
            for item in objects:
                if str(item.get("class", "")).lower() not in classes:
                    continue
                confirmed = item.get("track_confirmed") is not False
                confidence = float(item.get("confidence", 0))
                entry_threshold = self.phone_release_confidence if confirmed and event_type is EventType.PHONE_DETECTED else threshold
                if confidence >= entry_threshold and confirmed:
                    matches.append(item)
                elif (event_type is EventType.PHONE_DETECTED and not confirmed
                      and "object_track_id" in item and confidence >= threshold):
                    # Begin the timer on the first tracked detection, but never alert
                    # before the object tracker confirms the same object.
                    candidate_matches.append(item)
            observations = matches + candidate_matches
            if event_type is EventType.PHONE_DETECTED and observations:
                ids = {item["object_track_id"] for item in observations if "object_track_id" in item}
                if ids and self._phone_track_ids and not ids.intersection(self._phone_track_ids):
                    self._conditions[event_type].close()
                    self._phone_fast.close()
                    self._fast_hits = 0
                    self._fast_latched = False
                self._phone_track_ids = ids
            suppressed = bool(self.policy.get(allowed_key, False))
            if event_type is EventType.DOCUMENT_DETECTED:
                material_classes = {str(item.get("class", "")).lower() for item in matches}
                allowed_materials = {str(item).lower() for item in self.policy.get("allowed_materials", [])}
                suppressed = all((name == "paper" and bool(self.policy.get("allow_scratch_paper", True))) or (name != "paper" and (bool(self.policy.get("allow_book", False)) or name in allowed_materials)) for name in material_classes) if material_classes else False
            state = self._conditions[event_type].update(TemporalSample(
                monotonic_ms, bool(observations) and not suppressed,
                max((float(item.get("confidence", 0.0)) for item in observations), default=0.0),
            ))
            active = state is ConditionState.ACTIVE and bool(matches)
            if event_type is EventType.PHONE_DETECTED:
                strong_matches = [item for item in matches
                                  if float(item.get("confidence", 0)) >= self.phone_fast_confidence
                                  and item.get("track_confirmed") is True and "object_track_id" in item]
                chosen = next((item for item in strong_matches if item["object_track_id"] == self._fast_track_id),
                              max(strong_matches, key=lambda item: float(item["confidence"]), default=None))
                strong = chosen is not None and not suppressed
                if chosen is not None and chosen["object_track_id"] != self._fast_track_id:
                    self._phone_fast.close()
                    self._fast_hits = 0
                    self._fast_track_id = chosen["object_track_id"]
                previous_start = self._phone_fast.started_at_ms
                fast_state = self._phone_fast.update(TemporalSample(monotonic_ms, strong))
                if not strong:
                    self._fast_hits = 0
                elif previous_start is None or self._phone_fast.started_at_ms != previous_start:
                    self._fast_hits = 1
                else:
                    self._fast_hits += 1
                if state is ConditionState.INACTIVE or suppressed:
                    self._fast_latched = False
                if fast_state is ConditionState.ACTIVE and self._fast_hits >= self.phone_fast_min_hits:
                    self._fast_latched = True
                active = active or (self._fast_latched and bool(matches) and not suppressed)
            held, closing = False, False
            episode = self._episodes.get(event_type)
            gap = self._episode_gap_ms[event_type]
            if active and gap > 0:
                if episode is None:
                    best = max(matches, key=lambda item: float(item.get("confidence", 0.0)))
                    episode = {"trace": _trace(best, frame_id, monotonic_ms), "confidence": 0.0}
                    self._episodes[event_type] = episode
                episode["last_seen_ms"] = monotonic_ms
            elif episode is not None:
                if observations and not suppressed:
                    episode["last_seen_ms"] = monotonic_ms
                if not suppressed and monotonic_ms - int(episode["last_seen_ms"]) <= gap:
                    active, held = True, True
                else:
                    del self._episodes[event_type]
                    closing = True
            last_closed = self._conditions[event_type].last_closed
            if not matches and not last_closed and not held and not closing:
                continue
            if state is not ConditionState.INACTIVE or last_closed or held or closing:
                confidence = max((float(item.get("confidence", 0.0)) for item in matches), default=0.0)
                if episode is not None:
                    confidence = confidence or float(episode["confidence"])
                    episode["confidence"] = confidence
                result: dict[str, Any] = {
                    "session_id": session_id, "track_id": track_id, "event_type": event_type,
                    "severity": Severity.MEDIUM if event_type is EventType.DOCUMENT_DETECTED else Severity.HIGH,
                    "confidence": confidence, "active": active, "state": state.value,
                    "metrics": {"object_count": len(matches), "suppressed_by_policy": suppressed,
                                "fast_confirmation": event_type is EventType.PHONE_DETECTED and self._fast_latched,
                                "episode_hold": held, **self._conditions[event_type].current_metrics()},
                    "reason_codes": [f"{event_type.value}_CONFIDENCE"],
                }
                if episode is not None:
                    result["object_trace"] = dict(episode["trace"])
                results.append(result)
        return results


def _trace(item: dict[str, Any], frame_id: int | None, monotonic_ms: int) -> dict[str, Any]:
    """Where the alert came from: the analysed frame and the detection that opened it."""

    return {
        "frame_id": frame_id,
        "monotonic_ms": monotonic_ms,
        "label": str(item.get("class", "")),
        "confidence": float(item.get("confidence", 0.0)),
        "bbox": dict(item["bbox"]) if isinstance(item.get("bbox"), dict) else None,
        "object_track_id": item.get("object_track_id"),
        "model_name": item.get("model_name"),
        "model_version": item.get("model_version"),
    }
