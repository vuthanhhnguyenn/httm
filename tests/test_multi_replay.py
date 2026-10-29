"""Multi-subject replay: per-seat rule processors, outputs and CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import yaml
from proctoring_ai.capture.video_file_source import REPLAY_EPOCH_UTC as EPOCH
from proctoring_core.behavior.signals import RuleSignal, SignalState
from proctoring_core.events.models import ObservationBundle, ObservationHealth
from proctoring_core.types import EventType, Severity
from proctoring_runtime.replay import create_replay_processor

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from proctoring_ai.multi.crop import SubjectCropper  # noqa: E402
from proctoring_ai.multi.face_pool import PerSubjectFaceAnalyzer  # noqa: E402
from proctoring_ai.multi.focus import FocusScheduler  # noqa: E402
from proctoring_ai.multi.focus_analyzers import FocusAnalyzers  # noqa: E402
from proctoring_ai.multi.pipeline import MultiSubjectPipeline  # noqa: E402
from proctoring_ai.multi.tracker import MultiSubjectTracker  # noqa: E402
from proctoring_runtime.multi_replay import run_multi_replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load((ROOT / "configs" / "exam_policy.yaml").read_text(encoding="utf-8"))
    return config, dict(policy.get("exam_policy", policy))


def empty_bundle(session_id: UUID, *, frame_id: int, ms: int) -> ObservationBundle:
    return ObservationBundle(frame_id=frame_id, session_id=session_id,
                             captured_at_utc=EPOCH + timedelta(milliseconds=ms),
                             captured_at_monotonic_ms=ms, frame_size=(640, 480),
                             analyzer_health={"object_detector": ObservationHealth("ready")})


def neighbor_signal(session_id: UUID, seat: str, state: SignalState) -> RuleSignal:
    return RuleSignal(EventType.LOOK_AT_NEIGHBOR, state, session_id, seat, 0.8, Severity.LOW,
                      ("HEAD_TOWARDS_NEIGHBOR",), {"risk_cap": 25.0}, rule_key="LOOK_AT_NEIGHBOR")


async def test_extra_signal_opens_event_through_aggregator() -> None:
    config, policy = _config()
    session_id = uuid4()
    processor = create_replay_processor(session_id, policy, config)
    signal = neighbor_signal(session_id, str(uuid4()), SignalState.ACTIVE)
    items = await processor.process(empty_bundle(session_id, frame_id=1, ms=100), extra_signals=[signal])
    assert [(i["type"], i["action"]) for i in items if i["type"] == "LOOK_AT_NEIGHBOR"] == [
        ("LOOK_AT_NEIGHBOR", "opened")]
    opened = next(i for i in items if i["type"] == "LOOK_AT_NEIGHBOR")
    assert opened["label_vi"] == "Quay đầu về phía thí sinh bên cạnh"


async def test_last_signals_include_candidates_and_extra() -> None:
    config, policy = _config()
    session_id = uuid4()
    processor = create_replay_processor(session_id, policy, config)
    assert processor.last_signals == ()
    candidate = neighbor_signal(session_id, str(uuid4()), SignalState.CANDIDATE)
    await processor.process(empty_bundle(session_id, frame_id=1, ms=100), extra_signals=[candidate])
    assert candidate in processor.last_signals


# --- run_multi_replay -------------------------------------------------------------------------



def person(x: float) -> dict[str, Any]:
    return {"class": "person", "confidence": 0.9, "bbox": {"x": x, "y": 0.2, "width": 0.2, "height": 0.6}}


class ListSource:
    def __init__(self, count: int, step_ms: int = 100) -> None:
        self.count, self.step_ms, self.index = count, step_ms, 0

    async def open(self) -> None:
        return None

    async def read(self) -> dict[str, Any] | None:
        if self.index >= self.count:
            return None
        offset = self.index * self.step_ms
        self.index += 1
        return {"frame_id": self.index, "image": np.full((480, 640, 3), 90, np.uint8),
                "captured_at_utc": EPOCH + timedelta(milliseconds=offset), "captured_at_monotonic_ms": offset + 1,
                "video_offset_ms": offset, "frame_size": (640, 480)}

    async def close(self) -> None:
        return None


class ScriptedDetector:
    """People per frame from ``people(frame_index)``; ``on_detect`` lets a test burn fake time."""

    def __init__(self, people: Callable[[int], list[dict[str, Any]]], on_detect: Callable[[], None] | None = None):
        self.people, self.on_detect, self.calls = people, on_detect, 0

    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        if self.on_detect is not None:
            self.on_detect()
        return self.people(int(frame["frame_id"]) - 1)


class YawFace:
    """Face whose yaw depends on which seat (crop x) it analyses and on time."""

    def __init__(self, yaw: Callable[[int], float], calibration: str) -> None:
        self.yaw, self.calibration = yaw, calibration

    async def initialize(self) -> dict[str, Any]:
        return {"status": "ready"}

    async def analyze(self, frame: dict[str, Any]) -> dict[str, Any]:
        yaw = self.yaw(int(frame["captured_at_monotonic_ms"]))
        return {"status": "ready", "faces": [{
            "bbox": {"x": 0.4, "y": 0.3, "width": 0.2, "height": 0.3}, "yaw": yaw, "pitch": 0.0, "roll": 0.0,
            "quality": 0.9, "pose_valid": True, "face_width": 0.2, "calibration_status": self.calibration}]}

    def auto_calibrate(self) -> str:
        return self.calibration

    def close(self) -> None:
        return None


def multi_pipeline(people: Callable[[int], list[dict[str, Any]]], yaws: list[Callable[[int], float]], *,
                   calibration: str = "calibrated", on_detect: Callable[[], None] | None = None) -> MultiSubjectPipeline:
    queue = list(yaws)
    return MultiSubjectPipeline(
        detector=ScriptedDetector(people, on_detect), tracker=MultiSubjectTracker(),
        cropper=SubjectCropper(min_side=64),
        faces=PerSubjectFaceAnalyzer(lambda: YawFace(queue.pop(0) if queue else (lambda _ms: 0.0), calibration)),
        focus_analyzers=FocusAnalyzers(None, None), scheduler=FocusScheduler(frame_budget_ms=10_000))


def two_people(_index: int) -> list[dict[str, Any]]:
    return [person(0.1), person(0.35)]


def read_events(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]


async def run(pipeline: MultiSubjectPipeline, frames: int, out: Path, **kwargs: Any) -> Any:
    config, policy = _config()
    return await run_multi_replay(ListSource(frames), pipeline, policy=policy, raw_config=config, out_dir=out,
                                  **kwargs)


async def test_events_carry_runtime_track_id_and_are_written(tmp_path: Path) -> None:
    pipeline = multi_pipeline(two_people, [lambda _ms: 40.0, lambda _ms: 0.0])
    result = await run(pipeline, 30, tmp_path)
    events = read_events(tmp_path)
    assert events and all("runtime_track_id" in item for item in events)
    turns = [e for e in events if e["type"] == "HEAD_TURN_RIGHT" and e["action"] == "opened"]
    assert [e["runtime_track_id"] for e in turns] == [1]
    assert turns[0]["evidence_status"] == "saved"
    assert (tmp_path / "evidence" / f"{turns[0]['id']}.jpg").is_file()
    assert (tmp_path / "annotated.mp4").stat().st_size > 0
    timing = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert timing["frames"] == 30 == result.frames_analyzed
    assert set(timing["seats"]) == {"1", "2"}


async def test_neighbor_look_event_opens_for_turning_seat(tmp_path: Path) -> None:
    pipeline = multi_pipeline(two_people, [lambda _ms: 30.0, lambda _ms: 0.0])
    await run(pipeline, 25, tmp_path, write_video=False)
    opened = [e for e in read_events(tmp_path) if e["type"] == "LOOK_AT_NEIGHBOR" and e["action"] == "opened"]
    assert [e["runtime_track_id"] for e in opened] == [1]
    assert not (tmp_path / "annotated.mp4").exists()


async def test_expired_seat_closes_its_open_events(tmp_path: Path) -> None:
    def people(index: int) -> list[dict[str, Any]]:
        return [person(0.1), person(0.35)] if index < 10 else [person(0.35)]

    pipeline = multi_pipeline(people, [lambda _ms: 0.0, lambda _ms: 0.0])
    await run(pipeline, 140, tmp_path, write_video=False)
    events = read_events(tmp_path)
    missing = [e for e in events if e["type"] == "PERSON_MISSING" and e["runtime_track_id"] == 1]
    assert [e["action"] for e in missing][0] == "opened"
    closed = [e for e in missing if e["action"] == "closed"]
    assert closed and closed[0]["video_offset_ms"] < 13_900


async def test_short_video_never_calibrates_and_reports_status(tmp_path: Path) -> None:
    pipeline = multi_pipeline(two_people, [lambda _ms: 40.0, lambda _ms: 40.0], calibration="uncalibrated")
    result = await run(pipeline, 20, tmp_path, write_video=False)
    timing = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert timing["seats"]["1"]["calibration_status"] == "uncalibrated"
    assert not any(e["type"].startswith("HEAD_TURN") or e["type"] == "LOOK_AT_NEIGHBOR" for e in result.events)


async def test_realtime_mode_drops_frames_when_slow(tmp_path: Path) -> None:
    clock = {"t": 0.0}

    def burn() -> None:
        clock["t"] += 0.3

    async def fake_sleep(seconds: float) -> None:
        clock["t"] += max(0.0, seconds)

    pipeline = multi_pipeline(two_people, [lambda _ms: 0.0, lambda _ms: 0.0], on_detect=burn)
    config, policy = _config()
    result = await run_multi_replay(ListSource(30, step_ms=33), pipeline, policy=policy, raw_config=config,
                                    out_dir=tmp_path, mode="realtime", write_video=False,
                                    clock=lambda: clock["t"], sleep=fake_sleep)
    assert result.timing["dropped_frames"] > 0
    assert result.frames_analyzed + result.timing["dropped_frames"] == 30
    assert result.timing["latency_ms"]["p95"] >= 300.0


async def test_candidate_signal_marks_focus(tmp_path: Path) -> None:
    pipeline = multi_pipeline(two_people, [lambda _ms: 40.0, lambda _ms: 0.0])
    marked: list[UUID] = []
    original = pipeline.mark_focus

    def spy(seat_id: UUID, monotonic_ms: int) -> None:
        marked.append(seat_id)
        original(seat_id, monotonic_ms)

    pipeline.mark_focus = spy  # type: ignore[method-assign]
    await run(pipeline, 10, tmp_path, write_video=False)
    seat_one = next(seat for seat in pipeline.tracker.seats.values() if seat.runtime_track_id == 1)
    assert set(marked) == {seat_one.seat_id}


def test_cli_refuses_existing_out_dir_without_overwrite(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze_video_multi", ROOT / "scripts" / "analyze_video_multi.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "old.txt").write_text("x", encoding="utf-8")
    assert module.main([str(tmp_path / "video.mp4"), "--out", str(tmp_path)]) == 2
    assert module.main([str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "new")]) == 1
