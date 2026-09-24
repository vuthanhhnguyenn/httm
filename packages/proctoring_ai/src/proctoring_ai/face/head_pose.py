from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, degrees, exp, hypot, isfinite
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class HeadPoseCalibration:
    yaw_offset: float = 0.0
    pitch_offset: float = 0.0
    roll_offset: float = 0.0

    @classmethod
    def from_samples(cls, samples: list[dict[str, float]]) -> HeadPoseCalibration:
        if not samples:
            return cls()
        return cls(
            yaw_offset=median(item.get("yaw", 0.0) for item in samples),
            pitch_offset=median(item.get("pitch", 0.0) for item in samples),
            roll_offset=median(item.get("roll", 0.0) for item in samples),
        )


class HeadPoseEstimator:
    def __init__(self, calibration: HeadPoseCalibration | None = None) -> None:
        self.calibration = calibration or HeadPoseCalibration()
        self._previous: dict[str, float] | None = None
        self._last_ms: int | None = None
        self._samples: list[dict[str, float]] = []
        self._calibration_start: int | None = None
        self.calibration_status = "uncalibrated"
        self._angle_window: deque[dict[str, float]] = deque(maxlen=3)

    def begin_calibration(self) -> None:
        self._samples.clear()
        self._calibration_start = None
        self.calibration_status = "collecting"
        self._angle_window.clear()

    def calibrate(self, samples: list[dict[str, float]]) -> None:
        self.calibration = HeadPoseCalibration.from_samples(samples)
        self._previous = None
        self.calibration_status = "calibrated"

    def update(self, face: dict[str, Any], timestamp_ms: int) -> dict[str, Any]:
        """Smooth angles in time; calibrate only after an explicit operator request."""
        raw = {axis: float(face.get(axis, 0)) for axis in ("yaw", "pitch", "roll")}
        quality = float(face.get("quality", 0))
        valid = bool(face.get("pose_valid", True)) and all(isfinite(v) for v in raw.values())
        if not valid or quality < 0.35:
            self._previous = None
            self._last_ms = None
            self._samples.clear()
            self._angle_window.clear()
            return {**face, "quality": 0.0, "pose_valid": False,
                    "calibration_status": self.calibration_status}
        if self.calibration_status == "collecting":
            if self._calibration_start is None:
                self._calibration_start = timestamp_ms
            if self._samples and timestamp_ms - self._samples[-1]["timestamp_ms"] > 600:
                self._samples.clear()
            self._samples.append({**raw, "timestamp_ms": timestamp_ms})
            # A moving head must not become the neutral reference.
            if any(max(s[a] for s in self._samples) - min(s[a] for s in self._samples) > 8
                   for a in raw):
                self._samples = [{**raw, "timestamp_ms": timestamp_ms}]
            if len(self._samples) >= 8 and timestamp_ms - self._samples[0]["timestamp_ms"] >= 1500:
                self.calibrate(self._samples)
            elif timestamp_ms - self._calibration_start > 8000:
                self.calibration_status = "failed"
            else:
                return {**face, "quality": 0.0, "calibration_status": "collecting"}
        dt = timestamp_ms - self._last_ms if self._last_ms is not None else 0
        if dt <= 0 or dt > 600:
            self._angle_window.clear()
        self._angle_window.append(raw)
        filtered = {axis: median(item[axis] for item in self._angle_window) for axis in raw}
        adjusted = self.estimate({**filtered, "quality": quality})
        alpha = 1.0 - exp(-dt / 100.0) if 0 < dt <= 600 else 1.0
        angles = {}
        for axis in raw:
            current = float(adjusted[axis])
            previous = self._previous[axis] if self._previous else current
            delta = (current - previous + 180.0) % 360.0 - 180.0
            angles[axis] = (previous + alpha * delta + 180.0) % 360.0 - 180.0
        self._previous, self._last_ms = angles, timestamp_ms
        return {**face, **angles, "calibration_status": self.calibration_status}

    def estimate(self, face: dict[str, Any]) -> dict[str, float | str]:
        yaw = float(face.get("yaw", face.get("head_yaw", 0.0))) - self.calibration.yaw_offset
        pitch = float(face.get("pitch", face.get("head_pitch", 0.0))) - self.calibration.pitch_offset
        roll = float(face.get("roll", face.get("head_roll", 0.0))) - self.calibration.roll_offset
        quality = max(0.0, min(1.0, float(face.get("quality", 1.0))))
        return {"yaw": yaw, "pitch": pitch, "roll": roll, "quality": quality}


def pose_from_matrix(transform: Any) -> dict[str, float] | None:
    """MediaPipe canonical-to-camera rotation: Rz(roll) Ry(yaw) Rx(pitch).

    Camera coordinates are right-handed, +Y up. Positive pitch looks down;
    positive yaw points towards the right of the unmirrored image.
    """
    import numpy as np

    try:
        matrix = np.asarray(transform, dtype=float).reshape(4, 4)[:3, :3]
        if not np.isfinite(matrix).all():
            return None
        u, scale, vt = np.linalg.svd(matrix)
        if min(scale) < 1e-6 or np.linalg.det(matrix) <= 0:
            return None
        rotation = u @ vt  # Remove scale/shear before extracting rotation.
        sy = hypot(float(rotation[0, 0]), float(rotation[1, 0]))
        yaw = atan2(-float(rotation[2, 0]), sy)
        if sy < 1e-6:
            pitch = atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
            roll = 0.0
        else:
            pitch = atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            roll = atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        return {"yaw": degrees(yaw), "pitch": degrees(pitch), "roll": degrees(roll)}
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return None
