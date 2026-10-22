"""Per-candidate gaze calibration from looking at known targets.

The candidate looks at a few targets (screen centre, the four screen corners for a
computer exam, or the answer sheet for a paper exam). For each axis we fit
``screen = c0 + c1 * eye_ratio + c2 * head_angle / 45`` by least squares, so head
and eye movement both count. With only the centre target the profile still gives a
neutral eye position, which is enough for the "sideways glance" rule but not for
"off screen".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from statistics import median

from ..observations import FaceObs


class GazeStatus(StrEnum):
    UNCALIBRATED = "uncalibrated"
    COLLECTING = "collecting"
    CALIBRATED = "calibrated"
    FAILED = "failed"


# Normalized screen positions of the calibration targets.
TARGETS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.5),
    "top_left": (0.0, 0.0),
    "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0),
    "bottom_right": (1.0, 1.0),
}
PAPER_TARGET = "paper"
SCREEN_TARGETS = ("center", "top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True, slots=True)
class GazeSample:
    h: float
    v: float
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class AxisFit:
    c0: float
    c_eye: float
    c_head: float

    def __call__(self, eye: float, head: float) -> float:
        return self.c0 + self.c_eye * eye + self.c_head * head / 45.0


@dataclass(frozen=True, slots=True)
class GazeProfile:
    center: GazeSample
    x_fit: AxisFit | None = None
    y_fit: AxisFit | None = None
    paper: GazeSample | None = None
    residual: float = 0.0

    @property
    def has_screen(self) -> bool:
        return self.x_fit is not None and self.y_fit is not None

    def screen_point(self, sample: GazeSample) -> tuple[float, float] | None:
        if self.x_fit is None or self.y_fit is None:
            return None
        return self.x_fit(sample.h, sample.yaw), self.y_fit(sample.v, sample.pitch)


def _solve3(rows: list[tuple[float, float, float]], values: list[float], ridge: float = 1e-3) -> AxisFit | None:
    """Least squares for three coefficients via ridge-regularized normal equations."""

    ata = [[sum(row[i] * row[j] for row in rows) + (ridge if i == j and i > 0 else 0.0) for j in range(3)]
           for i in range(3)]
    atb = [sum(row[i] * value for row, value in zip(rows, values, strict=True)) for i in range(3)]
    # Gauss–Jordan elimination with partial pivoting on the 3×3 system.
    matrix = [ata[i] + [atb[i]] for i in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-9:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(3):
            if row != column:
                factor = matrix[row][column]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[column], strict=True)]
    return AxisFit(matrix[0][3], matrix[1][3], matrix[2][3])


@dataclass(slots=True)
class GazeCalibrator:
    """Collect samples per target while the UI shows it, then fit a :class:`GazeProfile`."""

    min_quality: float = 0.35
    min_samples_per_target: int = 4
    max_residual: float = 0.3
    status: GazeStatus = GazeStatus.UNCALIBRATED
    reason: str | None = None
    profile: GazeProfile | None = None
    _target: str | None = None
    _samples: dict[str, list[GazeSample]] = field(default_factory=dict)

    def begin(self) -> None:
        self.status, self.reason, self._target = GazeStatus.COLLECTING, None, None
        self._samples = {}

    def set_target(self, name: str | None) -> None:
        if name is not None and name not in TARGETS and name != PAPER_TARGET:
            raise ValueError(f"unknown gaze target {name!r}")
        self._target = name

    def add(self, face: FaceObs | None, quality: float) -> None:
        if self.status is not GazeStatus.COLLECTING or self._target is None or face is None or face.gaze is None:
            return
        if quality < self.min_quality:
            return
        self._samples.setdefault(self._target, []).append(
            GazeSample(face.gaze.h, face.gaze.v, face.yaw, face.pitch))

    def finish(self) -> GazeStatus:
        """Fit the profile from what was collected; refuse an unreliable one."""

        self._target = None
        robust = {name: GazeSample(median(s.h for s in items), median(s.v for s in items),
                                   median(s.yaw for s in items), median(s.pitch for s in items))
                  for name, items in self._samples.items() if len(items) >= self.min_samples_per_target}
        if "center" not in robust:
            return self._fail("CENTER_NOT_OBSERVED")
        screen = [name for name in SCREEN_TARGETS if name in robust]
        x_fit = y_fit = None
        residual = 0.0
        if len(screen) >= 4:
            # Same feature scaling as AxisFit.__call__: head angles enter as degrees / 45.
            x_fit = _solve3([(1.0, robust[n].h, robust[n].yaw / 45.0) for n in screen],
                            [TARGETS[n][0] for n in screen])
            y_fit = _solve3([(1.0, robust[n].v, robust[n].pitch / 45.0) for n in screen],
                            [TARGETS[n][1] for n in screen])
            if x_fit is None or y_fit is None:
                return self._fail("TARGETS_NOT_DISTINCT")
            errors = [(x_fit(robust[n].h, robust[n].yaw) - TARGETS[n][0]) ** 2
                      + (y_fit(robust[n].v, robust[n].pitch) - TARGETS[n][1]) ** 2 for n in screen]
            residual = (sum(errors) / len(errors)) ** 0.5
            if residual > self.max_residual:
                return self._fail("FIT_RESIDUAL_TOO_HIGH")
        self.profile = GazeProfile(robust["center"], x_fit, y_fit, robust.get(PAPER_TARGET), residual)
        self.status, self.reason = GazeStatus.CALIBRATED, None
        return self.status

    def calibrate_center_only(self, sample: GazeSample) -> None:
        """Neutral eye position without screen bounds (used by replay and quick calibration)."""

        self.profile = GazeProfile(sample)
        self.status, self.reason = GazeStatus.CALIBRATED, None

    def _fail(self, reason: str) -> GazeStatus:
        self.status, self.reason, self.profile = GazeStatus.FAILED, reason, None
        return self.status


__all__ = [
    "PAPER_TARGET",
    "SCREEN_TARGETS",
    "TARGETS",
    "AxisFit",
    "GazeCalibrator",
    "GazeProfile",
    "GazeSample",
    "GazeStatus",
]
