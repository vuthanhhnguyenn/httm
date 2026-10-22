"""Replay labelled videos through the live pipeline and score events per type.

See specs/002-phase1-hardening/contracts/evaluation.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
for source in (
    ROOT / "packages" / "proctoring_core" / "src",
    ROOT / "packages" / "proctoring_ai" / "src",
    ROOT / "packages" / "proctoring_runtime" / "src",
    ROOT / "apps" / "desktop" / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import yaml  # noqa: E402
from proctoring_ai.analyzer_factory import AnalyzerFactory  # noqa: E402
from proctoring_ai.capture.video_file_source import VideoFileFrameSource  # noqa: E402
from proctoring_core.config.models import canonical_sha256  # noqa: E402
from proctoring_core.evaluation import (  # noqa: E402
    CaseResult,
    EvaluationReport,
    LabelError,
    LabelMapping,
    apply_mapping,
    load_labels,
    load_mapping,
    score_case,
)
from proctoring_runtime.replay import (  # noqa: E402
    create_replay_processor,
    predicted_from_golden,
    read_golden,
    relax_timeouts,
    run_replay,
    write_golden,
)

EXIT_INPUT = 2
EXIT_MODEL = 3


class InputError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    video: Path
    labels: Path
    label_format: str
    exam_mode: str


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"không tìm thấy tệp {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise InputError(f"{path}: nội dung phải là bảng YAML")
    return payload


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate


def _load_set(path: Path, only: set[str], mode_override: str | None) -> tuple[str, list[Case], LabelMapping | None]:
    payload = _read_yaml(path)
    root = _resolve(path.parent, str(payload.get("root", ".")))
    mapping = load_mapping(_resolve(path.parent, payload["mapping"])) if payload.get("mapping") else None
    cases: list[Case] = []
    for item in payload.get("cases", []):
        case_id = str(item["id"])
        if only and case_id not in only:
            continue
        pattern = str(item["video"])
        matches = sorted(root.glob(pattern)) if any(ch in pattern for ch in "*?[") else [_resolve(root, pattern)]
        matches = [match for match in matches if match.is_file()]
        if len(matches) != 1:
            raise InputError(f"ca {case_id}: cần đúng một video cho {pattern!r}, tìm thấy {len(matches)}")
        cases.append(Case(case_id, matches[0], _resolve(root, str(item["labels"])),
                          str(item.get("label_format", "csv")),
                          mode_override or str(item.get("exam_mode", payload.get("exam_mode", "computer")))))
    if only and len(cases) != len(only):
        missing = sorted(only - {case.case_id for case in cases})
        raise InputError(f"không có ca: {', '.join(missing)}")
    return str(payload.get("name", path.stem)), cases, mapping


def apply_options(config: dict[str, Any], options: list[str]) -> dict[str, Any]:
    """Apply ``dotted.path=value`` overrides (value parsed as YAML) for tuning experiments."""

    result: dict[str, Any] = json.loads(json.dumps(config))
    for option in options:
        path, separator, raw = option.partition("=")
        if not separator or not path:
            raise InputError(f"tùy chọn không hợp lệ {option!r}; dùng dạng khóa.con=giá_trị")
        node = result
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise InputError(f"{path}: {key} không phải bảng cấu hình")
        node[keys[-1]] = yaml.safe_load(raw)
    return result


def _hardware(device: str | None) -> dict[str, str]:
    info = {"os": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "device": device or "cpu"}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def _models(manifest: Path) -> dict[str, str]:
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return {str(item.get("name")): str(item.get("version")) if item.get("version") != "latest-sha256-pinned"
            else f"sha256:{str(item.get('sha256', ''))[:12]}"
            for item in payload.get("models", []) if isinstance(item, dict)}


async def _run_case(case: Case, config: dict[str, Any], policy: dict[str, Any], args: argparse.Namespace,
                    mapping: LabelMapping | None, out_dir: Path) -> CaseResult:
    detection = config.get("detection", {})
    factory = AnalyzerFactory(ROOT / str(detection.get("model_manifest", "models/manifest.yaml")),
                              detector_image_size=int(detection.get("image_size", 416)),
                              detector_device=detection.get("device"))
    await factory.initialize()
    health = factory.health()
    if health.get("object_detector") != "ready":
        raise RuntimeError(f"MODEL:{factory.health_details()['object_detector']}")
    try:
        pipeline = factory.build_pipeline(config)
        relax_timeouts(pipeline)
        session_id = uuid4()
        end_ms = int(args.limit_seconds * 1000) if args.limit_seconds else None
        source = VideoFileFrameSource(case.video, analysis_rate_hz=args.rate, end_ms=end_ms)
        processor = create_replay_processor(session_id, policy, config, case.exam_mode)

        def progress(offset_ms: int) -> None:
            print(f"  {case.case_id}: {offset_ms / 60000:.1f} phút", flush=True)

        result = await run_replay(source, pipeline, processor, session_id=session_id, on_progress=progress)
    finally:
        factory.close()
    duration = min(source.duration_ms, end_ms) if end_ms else source.duration_ms
    duration = max(duration, result.last_offset_ms + 1)
    raw = load_labels(case.labels, case.label_format)
    if end_ms is not None:
        raw = [label for label in raw if label.start_ms < end_ms]
    intervals = apply_mapping(raw, mapping, case.labels, duration_ms=duration)
    if args.save_events:
        write_golden(out_dir / "events" / f"{case.case_id}.jsonl", result)
    tolerance = mapping.tolerance_ms if mapping is not None else int(config.get("evaluation", {}).get("tolerance_ms", 2000))
    scores = score_case(intervals, result.predicted_events(duration), duration_ms=duration, tolerance_ms=tolerance)
    return CaseResult(case.case_id, case.exam_mode, duration, result.frames_analyzed, result.ai_fps, scores)


def _rescore(args: argparse.Namespace, cases: list[Case], mapping: LabelMapping | None) -> list[CaseResult]:
    """Score saved event streams of an earlier run with the current labels and matching rules."""

    previous = json.loads((args.rescore / "report.json").read_text(encoding="utf-8"))
    info = {item["id"]: item for item in previous["cases"]}
    results = []
    for case in cases:
        if case.case_id not in info:
            raise InputError(f"{args.rescore}: không có ca {case.case_id}")
        meta = info[case.case_id]
        duration = int(meta["duration_ms"])
        predicted = predicted_from_golden(list(read_golden(args.rescore / "events" / f"{case.case_id}.jsonl")),
                                          duration)
        intervals = apply_mapping(load_labels(case.labels, case.label_format), mapping, case.labels,
                                  duration_ms=duration)
        tolerance = mapping.tolerance_ms if mapping is not None else 2000
        results.append(CaseResult(case.case_id, str(meta["exam_mode"]), duration, int(meta["frames_analyzed"]),
                                  float(meta["ai_fps"]), score_case(intervals, predicted, duration_ms=duration,
                                                                    tolerance_ms=tolerance)))
    return results


async def _main(args: argparse.Namespace) -> int:
    config = _read_yaml(ROOT / "configs" / "default.yaml")
    policy_file = _read_yaml(ROOT / "configs" / "exam_policy.yaml")
    policy = dict(policy_file.get("exam_policy", policy_file))
    device = os.environ.get("PROCTORING_AI_DEVICE")
    if device:
        config = {**config, "detection": {**config.get("detection", {}), "device": device}}
    config = apply_options(config, args.option or [])
    set_name, cases, mapping = _load_set(args.set, set(args.case or []), args.mode)
    for case in cases:
        # Fail fast: a malformed label file must not surface only after an hour of replay.
        apply_mapping(load_labels(case.labels, case.label_format), mapping, case.labels)
    out_dir = args.out / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in [] if args.rescore else cases:
        print(f"Đang phát lại {case.case_id} ({case.video.name}, {case.exam_mode})", flush=True)
        results.append(await _run_case(case, config, policy, args, mapping, out_dir))
    if args.rescore:
        results = _rescore(args, cases, mapping)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    report = EvaluationReport(
        set_name=set_name,
        created_at=datetime.now().astimezone(),
        config_sha256=canonical_sha256(config),
        policy_sha256=canonical_sha256(policy),
        models=_models(ROOT / str(config.get("detection", {}).get("model_manifest", "models/manifest.yaml"))),
        hardware=_hardware(config.get("detection", {}).get("device")),
        analysis_rate_hz=args.rate,
        cases=results,
        baseline=baseline,
    )
    (out_dir / "report.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = report.to_markdown()
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Đã ghi báo cáo vào {out_dir}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        # Windows consoles default to a legacy code page; reports are Vietnamese.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True, help="tệp bộ đánh giá YAML")
    parser.add_argument("--case", action="append", help="chỉ chạy ca này (lặp được)")
    parser.add_argument("--mode", choices=("paper", "computer"), help="ghi đè kiểu bài thi")
    parser.add_argument("--rate", type=float, default=8.0, help="lượt phân tích mỗi giây video")
    parser.add_argument("--limit-seconds", type=float, help="chỉ phát lại N giây đầu mỗi video")
    parser.add_argument("--baseline", type=Path, help="report.json để so sánh")
    parser.add_argument("--out", type=Path, default=ROOT / "storage" / "evaluation")
    parser.add_argument("--save-events", action="store_true", help="ghi golden JSONL theo ca")
    parser.add_argument("--rescore", type=Path, help="chấm lại sự kiện đã lưu của một lần chạy (thư mục có report.json)")
    parser.add_argument("--option", action="append",
                        help="ghi đè cấu hình, ví dụ behavior.face_not_visible.minimum_duration_seconds=3")
    args = parser.parse_args()
    try:
        return asyncio.run(_main(args))
    except (InputError, LabelError, KeyError) as exc:
        print(f"Lỗi đầu vào: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except RuntimeError as exc:
        if str(exc).startswith("MODEL:"):
            print(f"Mô hình chưa sẵn sàng: {exc}", file=sys.stderr)
            return EXIT_MODEL
        raise


if __name__ == "__main__":
    raise SystemExit(main())
