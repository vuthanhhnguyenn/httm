from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from proctoring_core.types import ExamMode
from proctoring_runtime.persistence import write_session_manifest

MODELS = {
    "object_detector": {"name": "yolo11n", "version": "8.3.0", "sha256": "a" * 64},
    "face": {"name": "mediapipe-face-landmarker", "version": "0.10", "sha256": "b" * 64},
}


def _settings() -> SimpleNamespace:
    return SimpleNamespace(raw_config={"camera": {"device_index": 0}},
                           raw_policy={"allow_phone": False})


def test_session_manifest_records_models_and_calibration(tmp_path: Path) -> None:
    session = SimpleNamespace(id=uuid4())
    target = write_session_manifest(tmp_path, session, ExamMode.COMPUTER, _settings(),
                                    model_versions=MODELS, calibration_profile_id="gaze-42")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model_versions"] == MODELS
    assert payload["calibration_profile_id"] == "gaze-42"
    assert payload["exam_mode"] == "computer"
    assert len(payload["config_sha256"]) == 64 and len(payload["policy_sha256"]) == 64


def test_session_manifest_defaults_to_empty_models(tmp_path: Path) -> None:
    session = SimpleNamespace(id=uuid4())
    target = write_session_manifest(tmp_path, session, ExamMode.PAPER, _settings())

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model_versions"] == {}
    assert payload["calibration_profile_id"] is None
