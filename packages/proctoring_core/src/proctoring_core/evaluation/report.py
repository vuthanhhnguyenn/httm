"""Evaluation report assembly, baseline comparison and rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .matching import ANY, TypeMetrics, merge_totals

SCHEMA_VERSION = "1.0"
_COMPARED = ("coverage", "false_alerts_per_min", "median_delay_ms", "single_event_share")


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    exam_mode: str
    duration_ms: int
    frames_analyzed: int
    ai_fps: float
    per_type: Mapping[str, TypeMetrics]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    set_name: str
    created_at: datetime
    config_sha256: str
    policy_sha256: str
    models: Mapping[str, str]
    hardware: Mapping[str, str]
    analysis_rate_hz: float
    cases: Sequence[CaseResult]
    baseline: Mapping[str, Any] | None = field(default=None)

    @property
    def totals(self) -> dict[str, TypeMetrics]:
        return merge_totals(case.per_type for case in self.cases)

    def baseline_delta(self) -> dict[str, dict[str, float]] | None:
        if not self.baseline:
            return None
        previous = self.baseline.get("totals", {})
        delta: dict[str, dict[str, float]] = {}
        for key, metrics in self.totals.items():
            before = previous.get(key)
            if not isinstance(before, Mapping):
                continue
            current = metrics.as_dict()
            row = {}
            for name in _COMPARED:
                now, then = current.get(name), before.get(name)
                if isinstance(now, int | float) and isinstance(then, int | float):
                    row[name] = round(float(now) - float(then), 4)
            delta[key] = row
        return delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "set": self.set_name,
            "created_at": self.created_at.isoformat(),
            "config_sha256": self.config_sha256,
            "policy_sha256": self.policy_sha256,
            "models": dict(self.models),
            "hardware": dict(self.hardware),
            "analysis_rate_hz": self.analysis_rate_hz,
            "cases": [
                {
                    "id": case.case_id,
                    "exam_mode": case.exam_mode,
                    "duration_ms": case.duration_ms,
                    "frames_analyzed": case.frames_analyzed,
                    "ai_fps": round(case.ai_fps, 2),
                    "per_type": {key: value.as_dict() for key, value in case.per_type.items()},
                }
                for case in self.cases
            ],
            "totals": {key: value.as_dict() for key, value in self.totals.items()},
            "baseline_delta": self.baseline_delta(),
        }

    def to_markdown(self) -> str:
        totals = self.totals
        lines = [
            f"# Báo cáo đánh giá: {self.set_name}",
            "",
            f"- Thời điểm: {self.created_at.isoformat()}",
            f"- Nhịp phân tích: {self.analysis_rate_hz:g} lượt/giây; số ca: {len(self.cases)}",
            f"- Cấu hình: `{self.config_sha256[:12]}`; chính sách: `{self.policy_sha256[:12]}`",
            f"- Mô hình: {', '.join(f'{k}={v}' for k, v in self.models.items()) or 'không rõ'}",
            f"- Phần cứng: {', '.join(f'{k}={v}' for k, v in self.hardware.items()) or 'không rõ'}",
            "",
            "Độ phủ theo loại là cận dưới khi một khoảng nhãn liệt kê nhiều loại; dòng ANY là tỉ lệ",
            "khoảng nhãn được phát hiện bởi ít nhất một loại phù hợp.",
            "",
            "| Loại | Nhãn | Đúng | Bỏ sót | Báo nhầm | Báo nhầm/phút | Độ phủ | Trễ trung vị (ms) | 1 sự kiện/khoảng |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in sorted(totals, key=lambda item: (item == ANY, item)):
            row = totals[key]
            delay = "—" if row.median_delay_ms is None else f"{row.median_delay_ms:.0f}"
            lines.append(
                f"| {key} | {row.labelled} | {row.tp} | {row.fn} | {row.fp} | "
                f"{row.false_alerts_per_min:.3f} | {row.coverage:.1%} | {delay} | {row.single_event_share:.1%} |"
            )
        delta = self.baseline_delta()
        if delta:
            lines += ["", "## So với mốc", "", "| Loại | Δ độ phủ | Δ báo nhầm/phút | Δ trễ (ms) |", "|---|---:|---:|---:|"]
            for key in sorted(delta, key=lambda item: (item == ANY, item)):
                change = delta[key]
                lines.append(
                    f"| {key} | {change.get('coverage', 0.0):+.1%} | "
                    f"{change.get('false_alerts_per_min', 0.0):+.3f} | {change.get('median_delay_ms', 0.0):+.0f} |"
                )
        lines += ["", "## Theo ca", "", "| Ca | Kiểu bài thi | Thời lượng (phút) | AI FPS | Độ phủ ANY | Báo nhầm ANY/phút |",
                  "|---|---|---:|---:|---:|---:|"]
        for case in self.cases:
            any_row = case.per_type.get(ANY)
            coverage = f"{any_row.coverage:.1%}" if any_row else "—"
            fapm = f"{any_row.false_alerts_per_min:.3f}" if any_row else "—"
            lines.append(f"| {case.case_id} | {case.exam_mode} | {case.duration_ms / 60000:.1f} | "
                         f"{case.ai_fps:.1f} | {coverage} | {fapm} |")
        return "\n".join(lines) + "\n"


__all__ = ["CaseResult", "EvaluationReport", "SCHEMA_VERSION"]
