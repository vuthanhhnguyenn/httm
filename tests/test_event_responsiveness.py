from __future__ import annotations

from uuid import uuid4

from proctoring_ai.face.head_pose import HeadPoseEstimator
from proctoring_ai.tracking.object_tracks import ObjectEvidenceTracker
from proctoring_core.behavior.head import HeadRuleEvaluator
from proctoring_core.behavior.objects import ObjectRuleEvaluator
from proctoring_core.types import EventType


def test_phone_fast_path_requires_repeated_strong_same_object_observations():
    tracker = ObjectEvidenceTracker()
    rules = ObjectRuleEvaluator(phone_confidence_threshold=0.55, phone_release_confidence=0.35, max_gap_ms=600)
    session = uuid4()
    phone = {"class": "cell phone", "confidence": 0.9,
             "bbox": {"x": 0.3, "y": 0.2, "width": 0.1, "height": 0.2}}
    for timestamp in (0, 100, 200, 300):
        result = rules.evaluate(session_id=session, track_id=None,
                                objects=tracker.update([phone], timestamp), monotonic_ms=timestamp)
        assert not any(item["active"] for item in result)
    result = rules.evaluate(session_id=session, track_id=None,
                            objects=tracker.update([phone], 400), monotonic_ms=400)
    assert result[0]["active"] and result[0]["metrics"]["fast_confirmation"]
    # A lower-confidence continuation must not close/reopen an already confirmed alert.
    result = rules.evaluate(session_id=session, track_id=None,
                            objects=tracker.update([{**phone, "confidence": 0.4}], 500), monotonic_ms=500)
    assert result[0]["active"]
    result = rules.evaluate(session_id=session, track_id=None, objects=[], monotonic_ms=1200)
    assert result[0]["active"] is False


def test_desktop_phone_timing_starts_with_first_tracked_detection():
    tracker = ObjectEvidenceTracker()
    rules = ObjectRuleEvaluator(
        phone_confidence_threshold=0.55,
        phone_min_duration_ms=600,
        phone_fast_confidence=0.85,
        phone_fast_duration_ms=200,
        phone_fast_min_hits=2,
    )
    session = uuid4()
    phone = {"class": "cell phone", "confidence": 0.9,
             "bbox": {"x": 0.3, "y": 0.2, "width": 0.1, "height": 0.2}}
    first = rules.evaluate(session_id=session, track_id=None,
                           objects=tracker.update([phone], 0), monotonic_ms=0)
    assert not any(item["active"] for item in first)
    second = rules.evaluate(session_id=session, track_id=None,
                            objects=tracker.update([phone], 200), monotonic_ms=200)
    assert not any(item["active"] for item in second)
    third = rules.evaluate(session_id=session, track_id=None,
                           objects=tracker.update([phone], 400), monotonic_ms=400)
    assert third[0]["active"] and third[0]["metrics"]["fast_confirmation"]


def test_desktop_phone_normal_path_uses_sustained_same_track_evidence():
    tracker = ObjectEvidenceTracker()
    rules = ObjectRuleEvaluator(phone_confidence_threshold=0.55, phone_min_duration_ms=600,
                                phone_fast_confidence=0.85, phone_fast_min_hits=2)
    session = uuid4()
    phone = {"class": "cell phone", "confidence": 0.7,
             "bbox": {"x": 0.3, "y": 0.2, "width": 0.1, "height": 0.2}}
    for timestamp in (0, 200, 400):
        result = rules.evaluate(session_id=session, track_id=None,
                                objects=tracker.update([phone], timestamp), monotonic_ms=timestamp)
        assert not any(item["active"] for item in result)
    result = rules.evaluate(session_id=session, track_id=None,
                            objects=tracker.update([phone], 600), monotonic_ms=600)
    assert result[0]["active"] and not result[0]["metrics"]["fast_confirmation"]


def test_phone_fast_path_cannot_mix_tracks_or_override_policy():
    for policy in ({}, {"allow_phone": True}):
        rules = ObjectRuleEvaluator(policy)
        session = uuid4()
        for timestamp in range(0, 1000, 100):
            # Two stable tracks, but neither has consecutive high-confidence detections.
            objects = [{"class": "cell phone", "track_confirmed": True, "object_track_id": index,
                        "confidence": 0.9 if (timestamp // 100) % 2 == index else 0.6} for index in (0, 1)]
            result = rules.evaluate(session_id=session, track_id=None, objects=objects, monotonic_ms=timestamp)
            assert not any(item["active"] for item in result)


def test_phone_single_high_hit_and_large_sample_gap_do_not_fast_confirm():
    rules = ObjectRuleEvaluator()
    session = uuid4()
    phone = {"class": "cell phone", "confidence": 0.9, "track_confirmed": True, "object_track_id": 1}
    for timestamp in (0, 400, 800):
        result = rules.evaluate(session_id=session, track_id=None, objects=[phone], monotonic_ms=timestamp)
        assert not any(item["active"] for item in result)


def test_head_rules_require_calibration_and_ignore_roll():
    rules = HeadRuleEvaluator({"head_turn": {"require_calibration": True, "minimum_duration_seconds": 0.8}})
    session = uuid4()
    def sample(timestamp, **values):
        return rules.evaluate(session_id=session, track_id=None,
                              sample={"monotonic_ms": timestamp, "quality": 0.8, **values})
    for timestamp in (0, 400, 800, 1200):
        assert not any(item["active"] for item in sample(timestamp, yaw=45, calibration_status="uncalibrated"))
    assert not any(item["active"] for item in sample(1400, yaw=0, roll=50, calibration_status="calibrated"))
    sample(1600, yaw=45, calibration_status="calibrated")
    result = sample(1900, yaw=45, calibration_status="calibrated")
    assert any(item["active"] and item["event_type"] is EventType.HEAD_TURN_RIGHT for item in result)


def test_strong_head_turn_alerts_after_short_confirmation_without_single_frame_alarm():
    rules = HeadRuleEvaluator({"head_turn": {"yaw_degrees": 22, "minimum_duration_seconds": 0.45,
                                               "fast_yaw_degrees": 42, "fast_duration_seconds": 0.25,
                                               "require_calibration": True}})
    session = uuid4()

    def sample(timestamp, yaw):
        return rules.evaluate(session_id=session, track_id=None,
                              sample={"monotonic_ms": timestamp, "yaw": yaw, "quality": 0.8,
                                      "calibration_status": "calibrated"})

    assert not any(item["active"] for item in sample(0, 55))
    assert not any(item["active"] for item in sample(150, 0))
    assert not any(item["active"] for item in sample(300, 55))
    result = sample(600, 55)
    assert any(item["active"] and item["event_type"] is EventType.HEAD_TURN_RIGHT
               and item["metrics"]["fast_confirmation"] for item in result)


def test_filtered_head_pose_reaches_fast_alert_on_sustained_turn():
    estimator = HeadPoseEstimator()
    estimator.calibrate([{"yaw": 0, "pitch": 0, "roll": 0}])
    rules = HeadRuleEvaluator({"head_turn": {"yaw_degrees": 22, "minimum_duration_seconds": 0.45,
                                               "fast_yaw_degrees": 42, "fast_duration_seconds": 0.25,
                                               "require_calibration": True}})
    session = uuid4()
    for timestamp in (0, 100, 200):
        face = estimator.update({"yaw": 0, "pitch": 0, "roll": 0, "quality": 0.8}, timestamp)
        rules.evaluate(session_id=session, track_id=None, sample={**face, "monotonic_ms": timestamp})
    for timestamp in (300, 450, 600):
        face = estimator.update({"yaw": 55, "pitch": 0, "roll": 0, "quality": 0.8}, timestamp)
        result = rules.evaluate(session_id=session, track_id=None,
                                sample={**face, "monotonic_ms": timestamp})
        assert not any(item["active"] for item in result)
    face = estimator.update({"yaw": 55, "pitch": 0, "roll": 0, "quality": 0.8}, 750)
    result = rules.evaluate(session_id=session, track_id=None, sample={**face, "monotonic_ms": 750})
    assert any(item["active"] and item["event_type"] is EventType.HEAD_TURN_RIGHT for item in result)


def test_face_loss_and_threshold_jitter_cannot_count_as_repeated_turns():
    rules = HeadRuleEvaluator({"min_duration_ms": 100, "abnormal_head_movement": {"minimum_turn_count": 2}})
    session = uuid4()
    for timestamp, yaw, quality in ((0, 45, 0.8), (100, 45, 0.8), (200, 0, 0),
                                    (300, 45, 0.8), (400, 45, 0.8), (500, 29, 0.8),
                                    (600, 45, 0.8), (700, 45, 0.8)):
        result = rules.evaluate(session_id=session, track_id=None,
                                sample={"monotonic_ms": timestamp, "yaw": yaw, "quality": quality})
        assert not any(item["event_type"] is EventType.ABNORMAL_HEAD_MOVEMENT and item["active"] for item in result)


def test_head_angle_filter_rejects_an_isolated_spike_but_accepts_sustained_turn():
    estimator = HeadPoseEstimator()
    for timestamp in (0, 100, 200):
        estimator.update({"yaw": 0, "quality": 0.8}, timestamp)
    assert abs(estimator.update({"yaw": 70, "quality": 0.8}, 300)["yaw"]) < 1
    assert abs(estimator.update({"yaw": 0, "quality": 0.8}, 400)["yaw"]) < 1
    for timestamp in (500, 600, 700, 800):
        result = estimator.update({"yaw": 45, "quality": 0.8}, timestamp)
    assert result["yaw"] > 35


def test_allowed_phone_never_alerts_even_after_long_strong_detection():
    rules = ObjectRuleEvaluator({"allow_phone": True})
    for timestamp in range(0, 2100, 100):
        result = rules.evaluate(session_id=uuid4(), track_id=None, monotonic_ms=timestamp,
                                objects=[{"class": "cell phone", "confidence": 0.95,
                                          "track_confirmed": True, "object_track_id": 1}])
        assert not any(item["active"] for item in result)


def test_head_calibration_does_not_bridge_a_camera_stall():
    estimator = HeadPoseEstimator()
    estimator.begin_calibration()
    for timestamp in range(0, 800, 100):
        estimator.update({"yaw": 0, "quality": 0.8}, timestamp)
    result = estimator.update({"yaw": 0, "quality": 0.8}, 2000)
    assert result["calibration_status"] == "collecting"


def test_default_detection_configuration_is_valid_and_calibration_is_required():
    from pathlib import Path

    import yaml
    from proctoring_core.config.schema import AppConfig

    root = Path(__file__).resolve().parents[1]
    config = AppConfig.model_validate(yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8")))
    assert config.behavior.head_turn.require_calibration
    assert config.behavior.phone.fast_confidence == 0.85
    assert config.behavior.phone.fast_min_hits == 2
    example = AppConfig.model_validate(yaml.safe_load(
        (root / "configs/default.example.yaml").read_text(encoding="utf-8")))
    assert config.behavior.head_turn == example.behavior.head_turn
    assert config.behavior.look_down == example.behavior.look_down


def test_moderate_down_uses_normal_confirmation():
    from proctoring_core.config.schema import BehaviorConfig

    rules = HeadRuleEvaluator(BehaviorConfig().model_dump())
    for timestamp in (0, 200, 400, 600):
        result = rules.evaluate(session_id=None, track_id=None, sample={
            "monotonic_ms": timestamp, "pitch": 22, "quality": 0.8,
            "calibration_status": "calibrated"})
        down = next(item for item in result if item["event_type"] is EventType.LOOK_DOWN)
        assert down["active"] is (timestamp >= 400)
        assert not down["metrics"]["fast_confirmation"]


def test_default_head_rules_report_turn_and_deep_down_within_600ms_after_filtering():
    from proctoring_core.config.schema import BehaviorConfig

    for yaw, pitch, expected in ((40, 0, EventType.HEAD_TURN_RIGHT),
                                 (-40, 0, EventType.HEAD_TURN_LEFT),
                                 (0, 40, EventType.LOOK_DOWN),
                                 (35, 40, EventType.LOOK_DOWN)):
        estimator = HeadPoseEstimator()
        estimator.calibrate([{"yaw": 0, "pitch": 0, "roll": 0}])
        rules = HeadRuleEvaluator(BehaviorConfig().model_dump())
        session = uuid4()
        active_at = None
        for timestamp in range(0, 1001, 100):
            face = estimator.update({"yaw": yaw if timestamp >= 400 else 0,
                                     "pitch": pitch if timestamp >= 400 else 0,
                                     "quality": 0.8}, timestamp)
            results = rules.evaluate(session_id=session, track_id=None,
                                     sample={**face, "monotonic_ms": timestamp})
            if any(item["event_type"] is expected and item["active"] for item in results):
                active_at = timestamp
                break
        assert active_at is not None and 400 < active_at <= 1000


def test_down_confirmation_rejects_spikes_invalid_pose_and_upward_or_roll_only():
    from proctoring_core.config.schema import BehaviorConfig

    for overrides in ({"pitch": -45}, {"pitch": 0, "roll": 45},
                      {"pose_valid": False}, {"quality": 0.2},
                      {"calibration_status": "uncalibrated"}):
        rules = HeadRuleEvaluator(BehaviorConfig().model_dump())
        for timestamp in (0, 300, 600, 900):
            results = rules.evaluate(session_id=None, track_id=None, sample={
                "monotonic_ms": timestamp, "pitch": 40, "quality": 0.8,
                "calibration_status": "calibrated", **overrides})
            assert not any(item["active"] for item in results)
    rules = HeadRuleEvaluator(BehaviorConfig().model_dump())
    for timestamp, pitch in ((0, 40), (100, 0), (200, 40), (300, 0)):
        results = rules.evaluate(session_id=None, track_id=None, sample={
            "monotonic_ms": timestamp, "pitch": pitch, "quality": 0.8,
            "calibration_status": "calibrated"})
        assert not any(item["active"] for item in results)


def test_deep_down_holds_one_event_and_requires_new_confirmation_after_stall():
    from proctoring_core.config.schema import BehaviorConfig

    rules = HeadRuleEvaluator(BehaviorConfig().model_dump())
    for timestamp, pitch, expected in ((0, 40, False), (300, 40, True),
                                       (400, 20, True), (500, 9, False),
                                       (600, 40, False), (900, 40, True),
                                       (1600, 40, False), (1900, 40, True)):
        results = rules.evaluate(session_id=None, track_id=None, sample={
            "monotonic_ms": timestamp, "pitch": pitch, "quality": 0.8,
            "calibration_status": "calibrated"})
        assert any(item["event_type"] is EventType.LOOK_DOWN and item["active"]
                   for item in results) is expected
