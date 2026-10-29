"""Collect per-frame timings into the ``timing.json`` summary of a multi-subject run."""

from __future__ import annotations

from math import ceil
from typing import Any

from .pipeline import MultiFrameResult


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; 0.0 for no samples."""

    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, ceil(fraction * len(ordered)) - 1)])


def _spread(values: list[float]) -> dict[str, float]:
    return {"p50": round(percentile(values, 0.5), 3), "p95": round(percentile(values, 0.95), 3)}


def _pose_stats(samples: list[tuple[float, float] | None]) -> dict[str, float]:
    """Why a seat never calibrated: a pitch far from 0 or a jittery pose blocks the neutral suggestion."""

    valid = [sample for sample in samples if sample is not None]
    jitter = [max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in zip(valid, valid[1:], strict=False)]
    return {"samples": len(samples),
            "valid_ratio": round(len(valid) / len(samples), 3) if samples else 0.0,
            "yaw_p50": round(percentile([yaw for yaw, _ in valid], 0.5), 1),
            "pitch_p50": round(percentile([pitch for _, pitch in valid], 0.5), 1),
            "jitter_p50": round(percentile(jitter, 0.5), 1)}


def _rate(timestamps: list[int]) -> float:
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        return 0.0
    return (len(timestamps) - 1) * 1000 / (timestamps[-1] - timestamps[0])


class TimingRecorder:
    def __init__(self) -> None:
        self._series: dict[str, list[float]] = {name: [] for name in
                                                ("detect", "track", "face_frame", "face_each", "focus", "total")}
        self._latency: list[float] = []
        self._frames = 0
        self._dropped = 0
        self._detector_reused = 0
        self._fresh_faces = 0
        self._reused_faces = 0
        self._seat_faces: dict[int, list[int]] = {}
        self._seat_fresh: dict[int, list[int]] = {}
        self._calibration: dict[int, str] = {}
        self._seat_poses: dict[int, list[tuple[float, float] | None]] = {}
        self._created = 0
        self._max_visible = 0
        self._over_limit_frames = 0

    def add(self, result: MultiFrameResult, *, latency_ms: float) -> None:
        timing = result.timing
        self._frames += 1
        self._latency.append(latency_ms)
        for name, value in (("detect", timing.detect_ms), ("track", timing.track_ms), ("face_frame", timing.face_ms),
                            ("focus", timing.focus_ms), ("total", timing.total_ms)):
            self._series[name].append(value)
        self._series["face_each"].extend(timing.face_each_ms.values())
        self._detector_reused += int(timing.detector_reused)
        runtime_ids = {seat.seat_id: seat.runtime_track_id for seat in result.tracker.analyzed}
        for seat_id, face in result.faces.items():
            runtime_id = runtime_ids.get(seat_id)
            if runtime_id is None or face.status == "not_run":
                continue
            self._seat_faces.setdefault(runtime_id, []).append(result.monotonic_ms)
            if face.reused:
                self._reused_faces += 1
            else:
                self._fresh_faces += 1
                self._seat_fresh.setdefault(runtime_id, []).append(result.monotonic_ms)
                if face.face is not None:
                    valid = bool(face.face.get("pose_valid", False))
                    self._seat_poses.setdefault(runtime_id, []).append(
                        (float(face.face.get("yaw", 0.0)), float(face.face.get("pitch", 0.0))) if valid else None)
            self._calibration[runtime_id] = face.calibration_status
        self._created += result.tracker.created
        self._max_visible = max(self._max_visible, len(result.tracker.analyzed) + len(result.tracker.over_limit))
        self._over_limit_frames += int(bool(result.tracker.over_limit))

    def add_dropped(self) -> None:
        self._dropped += 1

    def set_calibration(self, runtime_track_id: int, status: str) -> None:
        self._calibration[runtime_track_id] = status

    def summary(self, *, wall_s: float) -> dict[str, Any]:
        faces = self._fresh_faces + self._reused_faces
        seats = sorted(set(self._seat_faces) | set(self._calibration))
        return {
            "frames": self._frames,
            "dropped_frames": self._dropped,
            "ai_fps": round(self._frames / wall_s, 3) if wall_s > 0 else 0.0,
            "latency_ms": _spread(self._latency),
            "models": {name: _spread(values) for name, values in self._series.items()},
            "detector_reuse_ratio": round(self._detector_reused / self._frames, 3) if self._frames else 0.0,
            "face_reuse_ratio": round(self._reused_faces / faces, 3) if faces else 0.0,
            "seats": {seat: {"face_hz": round(_rate(self._seat_faces.get(seat, [])), 3),
                             "fresh_face_hz": round(_rate(self._seat_fresh.get(seat, [])), 3),
                             "calibration_status": self._calibration.get(seat, "uncalibrated"),
                             "pose": _pose_stats(self._seat_poses.get(seat, []))}
                      for seat in seats},
            "seats_created": self._created,
            "max_visible": self._max_visible,
            "id_switches": max(0, self._created - self._max_visible),
            "over_limit_frames": self._over_limit_frames,
        }


__all__ = ["TimingRecorder", "percentile"]
