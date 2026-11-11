"""End-to-end ``scripts/process_video.py`` on CPU: a synthetic video, a fake fast tier and fake deep-tier models."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from proctoring_ai.multi.cctv.review_detectors import ModelSpec
from proctoring_ai.multi.cctv.suspects import SuspectWindow
from proctoring_core.observations import Box

ROOT = Path(__file__).resolve().parents[1]
SIZE = (320, 180)
FPS = 10.0
SEAT_A, SEAT_B = uuid4(), uuid4()
BOX_A, BOX_B = Box(0.2, 0.3, 0.2, 0.4), Box(0.5, 0.3, 0.2, 0.4)


def _module():
    spec = importlib.util.spec_from_file_location("process_video", ROOT / "scripts" / "process_video.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(path: Path, seconds: float = 4.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)  # type: ignore[attr-defined]
    for index in range(int(seconds * FPS)):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 60 + index % 7, np.uint8))
    writer.release()
    return path


def _window(start: int = 1000, end: int = 3000, reasons: tuple[str, ...] = ("RULE_EVENT",)) -> SuspectWindow:
    return SuspectWindow(SEAT_A, 1, (SEAT_B,), start, end, reasons, BOX_A, {SEAT_A: BOX_A, SEAT_B: BOX_B}, 2000)


class _FastRunner:
    """Writes what the fast tier writes: suspects.jsonl and events.jsonl in its folder, and tracks.jsonl."""

    def __init__(self, windows: list[SuspectWindow]) -> None:
        self.windows = windows
        self.calls = 0

    def __call__(self, video: Path, fast_dir: Path, tracks_path: Path) -> None:
        self.calls += 1
        fast_dir.mkdir(parents=True, exist_ok=True)
        (fast_dir / "suspects.jsonl").write_text(
            "".join(json.dumps(window.to_json()) + "\n" for window in self.windows), encoding="utf-8")
        (fast_dir / "events.jsonl").write_text(json.dumps(
            {"action": "opened", "id": "f1", "type": "HEAD_TURN_LEFT", "seat_id": str(SEAT_A),
             "video_offset_ms": 1800}) + "\n", encoding="utf-8")
        with tracks_path.open("w", encoding="utf-8") as handle:
            for t in range(0, 4000, 200):
                handle.write(json.dumps({"t": t, "seats": {str(SEAT_A): [*_xywh(BOX_A), "candidate"],
                                                           str(SEAT_B): [*_xywh(BOX_B), "candidate"]}}) + "\n")


def _xywh(box: Box) -> list[float]:
    return [box.x, box.y, box.width, box.height]


class _Detectors:
    """Sees nobody and nothing: a window with a fast-tier rule event becomes UNCLASSIFIED_SUSPECT."""

    def __init__(self, fail: bool = False, pose_ok: bool = True) -> None:
        self.fail = fail
        self.status = {"fake-pose": "ok" if pose_ok else "bỏ qua: không có file", "fake-objects": "ok"}
        self.kinds = {"fake-objects": "objects", **({"fake-pose": "pose"} if pose_ok else {})}
        self.devices = {name: "cpu" for name in self.kinds}
        self.reasons = {name: "cpu: không có CUDA" for name in self.kinds}
        self.calls = 0

    def pose_many(self, crops, transforms):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model hỏng")
        return [[] for _ in crops]

    def objects_many(self, crops, transforms):
        return [[] for _ in crops]


def _events(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_end_to_end_writes_every_output(tmp_path: Path) -> None:
    module = _module()
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "ket-qua"
    runner, detectors = _FastRunner([_window()]), _Detectors()
    timing = module.process(video, out, detectors=detectors, fast_runner=runner)

    assert runner.calls == 1 and detectors.calls >= 1
    for name in ("annotated.mp4", "report.html", "events.csv", "events.jsonl", "review_cache.json.gz",
                 "timing.json", "suspects.jsonl", "tracks.jsonl"):
        assert (out / name).is_file(), name
    saved = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    for key in ("video_s", "fast_s", "review_s", "render_s", "report_s", "total_s", "ratio", "over_budget",
                "devices", "device_reasons", "deep_stats", "encoder"):
        assert key in saved, key
    assert saved["ratio"] == timing["ratio"] and abs(saved["video_s"] - 4.0) < 0.2
    assert saved["over_budget"] == (saved["ratio"] > 2)
    assert saved["devices"] == {"fake-objects": "cpu", "fake-pose": "cpu"}

    opened = [item for item in _events(out) if item["action"] == "opened"]
    assert [item["type"] for item in opened] == ["UNCLASSIFIED_SUSPECT"]
    event = opened[0]
    assert event["level"] == "yellow" and event["level_reason"] and event["label_vi"]
    assert (out / f"evidence/{event['id']}.jpg").is_file()
    assert event["evidence"] == f"evidence/{event['id']}.jpg"
    assert event["id"] in (out / "report.html").read_text(encoding="utf-8") or \
        f"evidence/{event['id']}.jpg" in (out / "report.html").read_text(encoding="utf-8")
    with gzip.open(out / "review_cache.json.gz", "rt", encoding="utf-8") as handle:
        assert len([line for line in handle if line.strip()]) == 1
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()


def test_second_run_replaces_cache(tmp_path: Path) -> None:
    module = _module()
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "ket-qua"
    for _ in range(2):
        module.process(video, out, detectors=_Detectors(), fast_runner=_FastRunner([_window()]))
    with gzip.open(out / "review_cache.json.gz", "rt", encoding="utf-8") as handle:
        assert len([line for line in handle if line.strip()]) == 1


def test_no_suspect_windows_still_gives_video_and_report(tmp_path: Path) -> None:
    module = _module()
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "ket-qua"
    detectors = _Detectors()
    timing = module.process(video, out, detectors=detectors, fast_runner=_FastRunner([]))
    assert detectors.calls == 0
    assert (out / "annotated.mp4").is_file() and (out / "report.html").is_file()
    assert [item for item in _events(out) if item["action"] == "opened"] == []
    assert timing["deep_stats"]["windows"] == 0


def test_window_error_becomes_review_skipped(tmp_path: Path) -> None:
    module = _module()
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "ket-qua"
    windows = [_window(), _window(500, 3500, ("OBJECT_HINT",))]
    module.process(video, out, detectors=_Detectors(fail=True), fast_runner=_FastRunner(windows))
    opened = [item for item in _events(out) if item["action"] == "opened"]
    assert [item["type"] for item in opened] == ["REVIEW_SKIPPED"]  # same seat within 10 s: one alert, one merged
    assert opened[0]["level"] == "yellow" and opened[0]["skip_reason"] == "error"
    assert (out / "report.html").is_file() and (out / "annotated.mp4").is_file()


def test_one_bad_window_does_not_lose_the_others(tmp_path: Path) -> None:
    module = _module()

    class _Picky(_Detectors):
        def pose_many(self, crops, transforms):
            self.calls += 1
            if any(crop.shape[1] > crop.shape[0] for crop in crops):  # only seat A's window (A+B) is wide
                raise RuntimeError("model hỏng")
            return [[] for _ in crops]

    other = SuspectWindow(SEAT_B, 2, (), 1000, 3000, ("RULE_EVENT",), BOX_B, {SEAT_B: BOX_B}, 2000)
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "ket-qua"
    detectors = _Picky()
    module.process(video, out, detectors=detectors, fast_runner=_FastRunner([_window(), other]))
    opened = {item["seat_id"]: item["type"] for item in _events(out) if item["action"] == "opened"}
    assert opened == {str(SEAT_A): "REVIEW_SKIPPED", str(SEAT_B): "UNCLASSIFIED_SUSPECT"}


def test_no_pose_model_marks_windows_skipped(tmp_path: Path) -> None:
    module = _module()
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "ket-qua"
    detectors = _Detectors(pose_ok=False)
    timing = module.process(video, out, detectors=detectors, fast_runner=_FastRunner([_window()]))
    assert detectors.calls == 0
    opened = [item for item in _events(out) if item["action"] == "opened"]
    assert [item["type"] for item in opened] == ["REVIEW_SKIPPED"]
    assert timing["deep_stats"]["skipped"] == 1


def test_fast_events_confirm_head_turns(tmp_path: Path) -> None:
    """A deep HEAD_TURN_LEFT is red only when the fast tier's own rule fired on the same seat within 3 s."""

    module = _module()
    from proctoring_ai.multi.cctv.confidence import load_confidence

    window = _window()
    record = {"action": "opened", "id": "d1", "type": "HEAD_TURN_LEFT", "seat_id": str(SEAT_A),
              "video_offset_ms": 2000}
    rules = load_confidence(ROOT / "configs/confidence.yaml")
    fast = [{"action": "opened", "type": "HEAD_TURN_LEFT", "seat_id": str(SEAT_A), "video_offset_ms": 1800}]
    red = module.finish_records([(window, [dict(record)])], fast, rules)
    yellow = module.finish_records([(window, [dict(record)])], [], rules)
    assert red[0]["level"] == "red" and yellow[0]["level"] == "yellow"
    assert red[0]["label_vi"] == "Quay đầu sang trái"


def test_unopenable_video_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    bad = tmp_path / "hong.mp4"
    bad.write_bytes(b"not a video")
    runner = _FastRunner([_window()])
    assert module.main([str(bad), "--out", str(tmp_path / "out")], fast_runner=runner) == 1
    assert runner.calls == 0
    assert "video" in capsys.readouterr().err.lower()
    assert module.main([str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "out")], fast_runner=runner) == 1


def test_device_overrides_put_objects_on_second_card() -> None:
    module = _module()
    specs = [ModelSpec("yolo11x-pose", "pose", Path("p.pt"), {}, 0.25, None, "cuda"),
             ModelSpec("yolo11x", "objects", Path("o.pt"), {}, 0.35, None, "intel:gpu")]
    assert module.device_overrides(specs, None, None) == {"yolo11x-pose": "cuda", "yolo11x": "cuda"}
    assert module.device_overrides(specs, "0", "1") == {"yolo11x-pose": "0", "yolo11x": "1"}
    assert module.device_overrides(specs, "0", None) == {"yolo11x-pose": "0", "yolo11x": "0"}


def test_quality_sets_crop_size() -> None:
    module = _module()
    assert module.QUALITY_IMGSZ == {"max": 960, "balanced": 640}


def test_fp32_and_share_iou_reach_the_deep_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnosis knobs: --fp32 builds the models without FP16, --share-iou reaches review_offline."""

    module = _module()
    built: dict = {}
    seen: dict = {}
    monkeypatch.setattr(module, "build_detectors",
                        lambda quality, device, device2, batch, half=True: built.update(half=half) or _Detectors())
    real = module.review_offline
    monkeypatch.setattr(module, "review_offline",
                        lambda *a, **kw: seen.update(share_iou=kw.get("share_iou")) or real(*a, **kw))
    video = _video(tmp_path / "in.mp4")
    assert module.main([str(video), "--out", str(tmp_path / "o"), "--fp32", "--share-iou", "1.1"],
                       fast_runner=_FastRunner([_window()])) == 0
    assert built == {"half": False} and seen == {"share_iou": 1.1}
    saved = json.loads((tmp_path / "o" / "timing.json").read_text(encoding="utf-8"))
    assert saved["config"] == {"quality": "max", "half": False, "share_iou": 1.1, "core": "rules", "all_seats": False,
                               "pose_flip": False}


def test_pose_flip_reaches_the_deep_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    seen: list[bool] = []
    monkeypatch.setattr(module, "build_detectors", lambda *a, **kw: _Detectors())
    real = module.review_offline
    monkeypatch.setattr(module, "review_offline",
                        lambda *a, **kw: seen.append(kw.get("pose_flip")) or real(*a, **kw))
    video = _video(tmp_path / "in.mp4")
    assert module.main([str(video), "--out", str(tmp_path / "o"), "--pose-flip"],
                       fast_runner=_FastRunner([_window()])) == 0
    assert seen == [True]
    saved = json.loads((tmp_path / "o" / "timing.json").read_text(encoding="utf-8"))
    assert saved["config"]["pose_flip"] is True


def test_reuse_fast_skips_the_fast_tier(tmp_path: Path) -> None:
    module = _module()
    video, out = _video(tmp_path / "in.mp4"), tmp_path / "o"
    runner = _FastRunner([_window()])
    module.process(video, out, detectors=_Detectors(), fast_runner=runner)
    module.process(video, out, detectors=_Detectors(), fast_runner=runner, reuse_fast=True)
    assert runner.calls == 1
    assert [e["type"] for e in _events(out) if e["action"] == "opened"] == ["UNCLASSIFIED_SUSPECT"]


def _person(cx: float) -> list[list[float]]:
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[0] = [cx, 0.36, 0.9]
    kps[1], kps[2] = [cx - 0.01, 0.35, 0.9], [cx + 0.01, 0.35, 0.9]
    kps[5], kps[6] = [cx - 0.04, 0.42, 0.9], [cx + 0.04, 0.42, 0.9]
    kps[9], kps[10] = [cx - 0.02, 0.55, 0.9], [cx + 0.02, 0.55, 0.9]
    return kps


class _SeeingDetectors(_Detectors):
    """Both candidates sit still in every crop."""

    def pose_many(self, crops, transforms):
        self.calls += 1
        people = [{"class": "person", "confidence": 0.9, "keypoints": _person(box.center[0])} for box in (BOX_A, BOX_B)]
        return [list(people) for _ in crops]


class _AlwaysLooking:
    """Stands in for a trained model: every window of every seat looks at the neighbour."""

    def __init__(self) -> None:
        from proctoring_ai.multi.cctv.learned import LearnedCore

        self.core = LearnedCore(None, ("NORMAL", "LOOK_AT_NEIGHBOR"), {"LOOK_AT_NEIGHBOR": (0.8, 0.5)}, {})
        self.core.predict = lambda rows: [{"NORMAL": 0.1, "LOOK_AT_NEIGHBOR": 0.9} for _ in rows]  # type: ignore[method-assign]


def test_learned_core_requires_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _module().main([str(_video(tmp_path / "in.mp4")), "--out", str(tmp_path / "o"), "--core", "learned"])
    assert code == 1 and "--model" in capsys.readouterr().err


def test_learned_core_missing_model_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _module().main([str(_video(tmp_path / "in.mp4")), "--out", str(tmp_path / "o"), "--core", "learned",
                           "--model", str(tmp_path / "khong-co")])
    assert code == 1 and "model.txt" in capsys.readouterr().err


def test_learned_core_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "o"
    timing = _module().process(_video(tmp_path / "in.mp4", seconds=6.0), out, detectors=_SeeingDetectors(),
                               fast_runner=_FastRunner([]), core="learned", learned=_AlwaysLooking().core)
    opened = [item for item in _events(out) if item["action"] == "opened"]
    assert opened and all(item["stage"] == "learned" and item["level"] == "red" for item in opened)
    assert {item["seat_id"] for item in opened} == {str(SEAT_A), str(SEAT_B)}
    assert all(item.get("label_vi") for item in opened)
    assert (out / "features.csv").is_file() and (out / "timeline.json.gz").is_file()
    assert (out / "annotated.mp4").is_file()
    assert timing["config"]["core"] == "learned"


def test_learned_core_short_video(tmp_path: Path) -> None:
    out = tmp_path / "o"
    timing = _module().process(_video(tmp_path / "in.mp4", seconds=2.0), out, detectors=_SeeingDetectors(),
                               fast_runner=_FastRunner([]), core="learned", learned=_AlwaysLooking().core)
    assert [item for item in _events(out) if item["action"] == "opened"] == []
    assert timing["video_s"] > 0


def test_learned_core_refuses_without_pose_model(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.ProcessError, match="khung xương"):
        module.process(_video(tmp_path / "in.mp4", seconds=6.0), tmp_path / "o", detectors=_Detectors(pose_ok=False),
                       fast_runner=_FastRunner([]), core="learned", learned=_AlwaysLooking().core)


def test_learned_core_report_shows_domain_warning_and_empty_note(tmp_path: Path) -> None:
    core = _AlwaysLooking().core
    core.meta = {"seat_height_median": 0.05}
    out = tmp_path / "o"
    _module().process(_video(tmp_path / "in.mp4", seconds=2.0), out, detectors=_SeeingDetectors(),
                      fast_runner=_FastRunner([]), core="learned", learned=core)
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "Cảnh báo" in report and "khác xa lúc train" in report
    assert "3 giây" in report


def test_all_seats_looks_at_every_seat_without_suspects(tmp_path: Path) -> None:
    out, detectors = tmp_path / "o", _SeeingDetectors()
    timing = _module().process(_video(tmp_path / "in.mp4", seconds=4.0), out, detectors=detectors,
                               fast_runner=_FastRunner([]), all_seats=True)
    assert detectors.calls > 0 and (out / "timeline.json.gz").is_file()
    assert timing["config"]["all_seats"] is True
    assert (out / "report.html").is_file() and (out / "annotated.mp4").is_file()


def test_all_seats_refuses_without_pose_model(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.ProcessError, match="khung xương"):
        module.process(_video(tmp_path / "in.mp4"), tmp_path / "o", detectors=_Detectors(pose_ok=False),
                       fast_runner=_FastRunner([]), all_seats=True)


def test_all_seats_reuse_fast_keeps_the_timeline(tmp_path: Path) -> None:
    module, video, out = _module(), _video(tmp_path / "in.mp4", seconds=4.0), tmp_path / "o"
    module.process(video, out, detectors=_SeeingDetectors(), fast_runner=_FastRunner([]), all_seats=True)
    again = _SeeingDetectors()
    module.process(video, out, detectors=again, fast_runner=_FastRunner([]), all_seats=True, reuse_fast=True)
    assert again.calls == 0


def _blip_records(*_args, **_kwargs):
    return [{"action": "opened", "id": "blip", "type": "LOOK_AT_LAP", "seat_id": str(SEAT_A), "video_offset_ms": 1000,
             "head": [0.3, 0.35], "level": "yellow", "stage": "review"},
            {"action": "closed", "id": "blip", "video_offset_ms": 1500}]


def test_keep_blips_skips_require_sustained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "finish_records", _blip_records)
    video = _video(tmp_path / "in.mp4", seconds=4.0)
    module.process(video, tmp_path / "default", detectors=_SeeingDetectors(), fast_runner=_FastRunner([]),
                   all_seats=True)
    assert [e for e in _events(tmp_path / "default") if e["action"] == "opened"] == []
    timing = module.process(video, tmp_path / "loose", detectors=_SeeingDetectors(), fast_runner=_FastRunner([]),
                            all_seats=True, keep_blips=True)
    assert [e["id"] for e in _events(tmp_path / "loose") if e["action"] == "opened"] == ["blip"]
    assert timing["config"]["keep_blips"] is True


def _lap(event_id: str, opened: int, closed: int) -> list[dict]:
    base = {"id": event_id, "type": "LOOK_AT_LAP", "seat_id": str(SEAT_A), "stage": "review"}
    return [{"action": "opened", "video_offset_ms": opened, "head": [0.3, 0.35], **base},
            {"action": "closed", "video_offset_ms": closed, **base}]


def test_dropped_blip_does_not_swallow_a_later_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip 1 (Task 3b): a 0 ms LOOK_AT_LAP at 52.4 s took the 60.0-65.4 s lap of the same seat as a merged
    repeat, then require_sustained dropped the blip and its repeat with it. Blips go before the deduper."""

    module = _module()
    window = SuspectWindow(SEAT_A, None, (), 0, 70000, ("ALL_SEATS",), BOX_A, {SEAT_A: BOX_A})
    monkeypatch.setattr(module, "review_window",
                        lambda *_args, **_kwargs: _lap("blip", 52400, 52400) + _lap("lap", 60000, 65400))
    frames = {module.window_key(window): [module.ReviewFrame(1000, {SEAT_A: BOX_A}, {}, [])]}
    found = module.review_records([window], frames, sustained_ms=1000)[0][1]
    assert [(r["action"], r["id"]) for r in found] == [("opened", "lap"), ("closed", "lap")]


def test_sustained_minimum_follows_a_short_turn_hold() -> None:
    module = _module()
    turns = {"LOOK_AT_NEIGHBOR", "HEAD_TURN_LEFT", "HEAD_TURN_RIGHT", "TURN_AROUND"}
    assert module.sustained_by_type(750) == dict.fromkeys(turns, 750)
    assert module.sustained_by_type(1500) == dict.fromkeys(turns, 1000)


def test_short_turn_kept_when_turn_hold_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    window = SuspectWindow(SEAT_A, None, (), 0, 70000, ("ALL_SEATS",), BOX_A, {SEAT_A: BOX_A})
    base = {"id": "look", "type": "LOOK_AT_NEIGHBOR", "seat_id": str(SEAT_A), "stage": "review"}
    look = [{"action": "opened", "video_offset_ms": 5000, "head": [0.3, 0.35], **base},
            {"action": "closed", "video_offset_ms": 5800, **base}]
    monkeypatch.setattr(module, "review_window", lambda *_args, **_kwargs: [dict(r) for r in look])
    frames = {module.window_key(window): [module.ReviewFrame(1000, {SEAT_A: BOX_A}, {}, [])]}
    assert module.review_records([window], frames, sustained_ms=1000)[0][1] == []  # default hold 1.5 s
    short = module.ReviewRuleConfig(turn_hold_ms=750)
    kept = module.review_records([window], frames, config=short, sustained_ms=1000)[0][1]
    assert [r["action"] for r in kept] == ["opened", "closed"]


def test_final_blip_filter_follows_the_review_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    base = {"id": "look", "type": "LOOK_AT_NEIGHBOR", "seat_id": str(SEAT_A), "stage": "review"}
    look = [{"action": "opened", "video_offset_ms": 1000, "head": [0.3, 0.35], "level": "yellow", **base},
            {"action": "closed", "video_offset_ms": 1800, **base}]
    monkeypatch.setattr(module, "finish_records", lambda *_args, **_kwargs: [dict(r) for r in look])
    video = _video(tmp_path / "in.mp4", seconds=4.0)
    monkeypatch.setattr(module, "REVIEW_CONFIG", module.ReviewRuleConfig(turn_hold_ms=1500))
    module.process(video, tmp_path / "default", detectors=_SeeingDetectors(), fast_runner=_FastRunner([]),
                   all_seats=True)
    assert [e for e in _events(tmp_path / "default") if e["action"] == "opened"] == []
    monkeypatch.setattr(module, "REVIEW_CONFIG", module.ReviewRuleConfig(turn_hold_ms=750))
    module.process(video, tmp_path / "short", detectors=_SeeingDetectors(), fast_runner=_FastRunner([]),
                   all_seats=True)
    assert [e["id"] for e in _events(tmp_path / "short") if e["action"] == "opened"] == ["look"]
