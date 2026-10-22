"""Right-rail status cards: realtime metrics, analyzer health and review priority."""

from __future__ import annotations

from typing import Any

from proctoring_runtime.messages import health_reason, level_name
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout

from .theme import card, muted_label, set_text_if_changed, title_label

METRICS = (("preview", "FPS hiển thị"), ("processed", "AI FPS"), ("latency", "Độ trễ AI · ms"),
           ("dropped", "Khung bỏ qua"))
ANALYZERS = (("object_detector", "YOLO"), ("face", "Mặt"), ("pose", "Tư thế"), ("hands", "Bàn tay"))
REALTIME_TEXT = {
    "REALTIME": "Đang xử lý thời gian thực",
    "DEGRADED": "AI chưa đạt mục tiêu 20 FPS hoặc có bộ phân tích chưa sẵn sàng",
    "WARMING_UP": "Đang đo tốc độ sau khởi động",
}
SHORT_STATUS = {"not_run": "Chờ", "loading": "Nạp", "skipped": "Chờ", "busy": "Bận", "timeout": "Chậm",
                "unavailable": "Thiếu", "error": "Lỗi", "disabled": "Tắt"}
HEALTH_TEXT = {
    "loading": "Đang nạp model",
    "ready": "Đang chạy",
    "no_detection": "Chưa thấy đối tượng",
    "not_run": "Chưa chạy",
    "disabled": "Đã tắt",
    "unavailable": "Thiếu model",
    "timeout": "Quá thời gian",
    "busy": "Đang bận · bỏ qua khung mới",
    "error": "Có lỗi",
    "skipped": "Chờ lượt xử lý",
}


def health_text(status: str, reason: str) -> str:
    text = HEALTH_TEXT.get(status, "Chưa xác định")
    if status in {"unavailable", "error"} and reason and ("MODEL_MISSING" in reason or "FileNotFoundError" in reason):
        text = "Thiếu model"
    return text


class MetricsPanel:
    def __init__(self) -> None:
        self.frame = card()
        layout = QVBoxLayout(self.frame)
        layout.addWidget(title_label("Hiệu năng thời gian thực"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.values: dict[str, QLabel] = {}
        for index, (key, label) in enumerate(METRICS):
            tile = QFrame()
            tile.setObjectName("metricTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(10, 7, 10, 7)
            tile_layout.setSpacing(1)
            value = QLabel("—")
            value.setObjectName("metricValue")
            caption = QLabel(label)
            caption.setObjectName("metricLabel")
            caption.setWordWrap(True)
            tile_layout.addWidget(value)
            tile_layout.addWidget(caption)
            grid.addWidget(tile, index // 2, index % 2)
            self.values[key] = value
        layout.addLayout(grid)
        self.realtime_status = muted_label("AI chưa chạy")
        layout.addWidget(self.realtime_status)

    def refresh(self, metrics: dict[str, Any], preview_fps: float) -> None:
        set_text_if_changed(self.values["preview"], f"{preview_fps:.1f}")
        set_text_if_changed(self.values["processed"], f"{float(metrics.get('processed_fps', 0)):.1f}")
        set_text_if_changed(self.values["latency"], f"{float(metrics.get('latency_ms_p95', 0)):.0f}")
        set_text_if_changed(self.values["dropped"], str(metrics.get("dropped_frames", 0)))
        realtime = str(metrics.get("realtime_status", "WARMING_UP"))
        set_text_if_changed(self.realtime_status, REALTIME_TEXT.get(realtime, "Chưa xác định tốc độ"))


class HealthPanel:
    def __init__(self) -> None:
        self.frame = card()
        layout = QVBoxLayout(self.frame)
        layout.addWidget(title_label("Bộ phân tích · FPS"))
        self.labels: dict[str, QLabel] = {}
        self.fps_labels: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        for row, (key, label) in enumerate(ANALYZERS):
            name = QLabel(label)
            name.setObjectName("healthName")
            name.setToolTip({"object_detector": "YOLO · vật thể", "face": "Khuôn mặt · hướng đầu"}.get(key, label))
            status = QLabel("Chờ")
            status.setObjectName("healthState")
            status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            fps = QLabel("—")
            fps.setObjectName("healthFps")
            fps.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            fps.setMinimumWidth(48)
            grid.addWidget(name, row, 0)
            grid.addWidget(status, row, 1)
            grid.addWidget(fps, row, 2)
            self.labels[key] = status
            self.fps_labels[key] = fps
        layout.addLayout(grid)

    def refresh(self, analyzers: dict[str, dict[str, str]], performance: dict[str, dict[str, Any]]) -> None:
        for key, label in self.labels.items():
            detail = analyzers.get(key, {})
            status = str(detail.get("status", "not_run"))
            reason = str(detail.get("reason", ""))
            device = str(detail.get("device", ""))
            if not device and key in {"face", "pose", "hands"}:
                device = "cpu"
            device_label = "GPU" if "cuda" in device.casefold() else "CPU" if device.casefold() == "cpu" else ""
            stats = performance.get(key, {})
            fps_label = f"{float(stats['fps']):.1f}" if "fps" in stats else "—"
            display_status = str(stats.get("status", status) if status == "skipped" else status)
            short = (device_label or "Chạy") if display_status in {"ready", "no_detection"} \
                else SHORT_STATUS.get(display_status, "Chờ")
            set_text_if_changed(label, short)
            set_text_if_changed(self.fps_labels[key], fps_label)
            tooltip = " · ".join(part for part in (health_text(display_status, reason),
                                                   f"Thiết bị xử lý: {device}" if device else "",
                                                   health_reason(reason)) if part)
            if stats:
                tooltip += (f" · Lượt suy luận gần nhất: {stats.get('latency_ms', 0):.0f} ms. "
                            "FPS chỉ đếm lượt xử lý thật.")
            for part in (label, self.fps_labels[key]):
                if part.toolTip() != tooltip:
                    part.setToolTip(tooltip)
            if label.property("health") == display_status:
                continue
            label.setProperty("health", display_status)
            label.style().unpolish(label)
            label.style().polish(label)


class RiskPanel:
    def __init__(self) -> None:
        self.frame = card()
        layout = QVBoxLayout(self.frame)
        layout.addWidget(title_label("Tín hiệu cần rà soát"))
        self.value = QLabel("—")
        self.value.setObjectName("riskValue")
        self.value.setWordWrap(True)
        layout.addWidget(self.value)
        layout.addWidget(muted_label("AI hỗ trợ phát hiện tín hiệu; giám thị đưa ra đánh giá cuối cùng."))
        self.history = muted_label("Lịch sử phiên: 0 sự kiện · 0 điểm tích lũy · đỉnh 0/100")
        layout.addWidget(self.history)

    def refresh(self, score: float, level: str, history: dict[str, Any]) -> None:
        self.value.setText(f"{score:.1f} / 100   ·   {level_name(level)}")
        self.history.setText(
            f"Lịch sử phiên: {int(history.get('event_count', 0))} sự kiện · "
            f"{float(history.get('cumulative_score', 0)):.0f} điểm tích lũy · "
            f"đỉnh {float(history.get('peak_score', 0)):.0f}/100"
        )


__all__ = ["HealthPanel", "MetricsPanel", "RiskPanel", "health_text"]
