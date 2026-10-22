"""Test helper: run the behaviour engine on an ObservationBundle and return mappings."""

from __future__ import annotations

from typing import Any

from proctoring_core.behavior.engine import BehaviorEngine
from proctoring_core.config.schema import AppConfig
from proctoring_core.events.models import ObservationBundle
from proctoring_core.observations import Observation


class BundleRules:
    def __init__(self, session_id: Any, policy: dict[str, Any], config: dict[str, Any]) -> None:
        self.engine = BehaviorEngine(session_id, policy, AppConfig.model_validate(config).behavior)

    def evaluate(self, bundle: ObservationBundle) -> list[dict[str, Any]]:
        return [signal.as_mapping() for signal in self.engine.evaluate(Observation.from_bundle(bundle))]
