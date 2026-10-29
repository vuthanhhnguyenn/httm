from __future__ import annotations

from proctoring_ai.face.head_pose import HeadPoseEstimator


def face(yaw: float, pitch: float = 8.0) -> dict[str, float | bool]:
    return {"yaw": yaw, "pitch": pitch, "roll": 0.0, "quality": 0.9, "pose_valid": True}


def feed(estimator: HeadPoseEstimator, yaws: list[float], start: int = 0, pitch: float = 8.0) -> str:
    status = ""
    for index, yaw in enumerate(yaws):
        status = str(estimator.update(face(yaw, pitch), start + index * 125)["calibration_status"])
    return status


def test_steady_head_is_suggested_but_not_applied_until_accepted() -> None:
    estimator = HeadPoseEstimator()
    assert feed(estimator, [3.0 + (index % 2) for index in range(10)]) == "uncalibrated"
    assert feed(estimator, [3.0 + (index % 2) for index in range(10)], start=1250) == "suggested"
    assert estimator.calibration.yaw_offset == 0.0  # nothing applied silently

    estimator.begin_calibration()  # operator accepts
    assert estimator.calibration_status == "calibrated"
    assert 3.0 <= estimator.calibration.yaw_offset <= 4.0


def test_moving_head_withdraws_suggestion() -> None:
    estimator = HeadPoseEstimator()
    feed(estimator, [2.0] * 20)
    assert estimator.calibration_status == "suggested"
    assert feed(estimator, [25.0, -20.0, 30.0], start=3000) == "uncalibrated"


def test_off_axis_baseline_is_rejected() -> None:
    estimator = HeadPoseEstimator(max_baseline_degrees=35.0)
    assert feed(estimator, [50.0] * 30) == "uncalibrated"  # never suggested
    estimator.begin_calibration()
    status = feed(estimator, [50.0] * 20, start=5000)
    assert status == "failed" and estimator.calibration_reason == "BASELINE_OFF_AXIS"


def test_auto_calibrate_accepts_suggestion_after_3_s() -> None:
    from proctoring_ai.face.landmarks import MediaPipeFaceLandmarker

    landmarker = MediaPipeFaceLandmarker(None, suggest_after_ms=3000)
    estimator = landmarker._pose_estimator
    feed(estimator, [3.0 + (index % 2) for index in range(17)])  # 0 … 2000 ms
    assert landmarker.auto_calibrate() == "uncalibrated"
    feed(estimator, [3.0 + (index % 2) for index in range(9)], start=2125)  # … 3125 ms
    assert landmarker.auto_calibrate() == "calibrated"
    assert 3.0 <= estimator.calibration.yaw_offset <= 4.0


def test_rejected_face_withdraws_pending_suggestion() -> None:
    from proctoring_ai.face.landmarks import MediaPipeFaceLandmarker

    landmarker = MediaPipeFaceLandmarker(None, suggest_after_ms=3000)
    feed(landmarker._pose_estimator, [3.0 + (index % 2) for index in range(26)])  # … 3125 ms
    assert landmarker._pose_estimator.calibration_status == "suggested"
    landmarker.reject_last_face()  # it belonged to someone else
    assert landmarker.auto_calibrate() == "uncalibrated"
