"""PySide6 operator window with an in-process camera and detection overlay."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .messages import event_description, event_name, health_reason, level_name, local_clock
from .overlay import OverlayState
from .runtime import DesktopRuntime, LatestPreview


class MainWindow(QMainWindow):
    def __init__(self, repository_root: Path) -> None:
        super().__init__()
        self.repository_root = repository_root
        self.preview = LatestPreview()
        self.runtime: DesktopRuntime | None = None
        self._last_frame_id: int | None = None
        self._analysis: dict[str, Any] | None = None
        self._overlay = OverlayState()
        self._calibration_status = "uncalibrated"
        self._metrics: dict[str, Any] = {}
        self._risk_score = 0.0
        self._risk_level = "NORMAL"
        self._risk_history: dict[str, Any] = {}
        self._preview_times: deque[float] = deque()
        self._preview_fps = 0.0
        self._last_analyzers: dict[str, dict[str, str]] = {}
        self._analyzer_performance: dict[str, dict[str, Any]] = {}
        self._last_error = ""
        self._event_rows: dict[str, QListWidgetItem] = {}
        self._evidence_dialogs: list[QDialog] = []

        self.setWindowTitle("Smart Exam Proctoring · Giám sát cục bộ")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(_stylesheet())
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._refresh_preview)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        page = QVBoxLayout(root)
        page.setContentsMargins(28, 22, 28, 24)
        page.setSpacing(18)

        header = QHBoxLayout()
        heading = QVBoxLayout()
        eyebrow = QLabel("GIÁM THỊ · CHẠY TRÊN MÁY NÀY")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Giám sát phiên thi")
        title.setObjectName("pageTitle")
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        header.addLayout(heading)
        header.addStretch(1)
        self.app_status = QLabel("Sẵn sàng")
        self.app_status.setObjectName("statusBadge")
        header.addWidget(self.app_status)
        page.addLayout(header)

        controls = self._card()
        controls_row = QHBoxLayout(controls)
        controls_label = QLabel("Camera")
        controls_label.setObjectName("cardTitle")
        controls_row.addWidget(controls_label)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 16)
        self.camera_index.setValue(0)
        self.camera_index.setPrefix("Thiết bị ")
        controls_row.addWidget(self.camera_index)
        controls_row.addStretch(1)
        self.calibrate_button = QPushButton("Hiệu chỉnh hướng đầu")
        self.calibrate_button.setToolTip("Nhìn thẳng màn hình và giữ đầu yên khoảng 2 giây.")
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.clicked.connect(self._calibrate_head)
        controls_row.addWidget(self.calibrate_button)
        self.start_button = QPushButton("Bắt đầu giám sát")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start)
        controls_row.addWidget(self.start_button)
        self.stop_button = QPushButton("Dừng phiên")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        controls_row.addWidget(self.stop_button)
        page.addWidget(controls)

        content = QHBoxLayout()
        content.setSpacing(18)
        self.camera_card = self._card()
        camera_layout = QVBoxLayout(self.camera_card)
        camera_header = QHBoxLayout()
        camera_title = QLabel("Webcam trực tiếp")
        camera_title.setObjectName("cardTitle")
        camera_header.addWidget(camera_title)
        camera_header.addStretch(1)
        self.camera_status = QLabel("Camera chưa mở")
        self.camera_status.setObjectName("mutedBadge")
        camera_header.addWidget(self.camera_status)
        camera_layout.addLayout(camera_header)
        self.video = QLabel("Nhấn ‘Bắt đầu giám sát’ để mở webcam")
        self.video.setObjectName("videoSurface")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 430)
        camera_layout.addWidget(self.video, 1)
        self.preview_note = QLabel("Hình ảnh và AI được xử lý trên máy này.")
        self.preview_note.setObjectName("mutedText")
        self.preview_note.setWordWrap(True)
        camera_layout.addWidget(self.preview_note)
        content.addWidget(self.camera_card, 7)

        rail = QVBoxLayout()
        rail.setSpacing(14)
        metrics_card = self._card()
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_title = QLabel("Hiệu năng thời gian thực")
        metrics_title.setObjectName("cardTitle")
        metrics_title.setWordWrap(True)
        metrics_layout.addWidget(metrics_title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.metric_values: dict[str, QLabel] = {}
        for index, (key, label) in enumerate(
            (("preview", "FPS hiển thị"), ("processed", "AI FPS"), ("latency", "Độ trễ AI · ms"), ("dropped", "Khung bỏ qua"))
        ):
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
            self.metric_values[key] = value
        metrics_layout.addLayout(grid)
        self.realtime_status = QLabel("AI chưa chạy")
        self.realtime_status.setObjectName("mutedText")
        self.realtime_status.setWordWrap(True)
        metrics_layout.addWidget(self.realtime_status)
        rail.addWidget(metrics_card)

        health_card = self._card()
        health_layout = QVBoxLayout(health_card)
        health_title = QLabel("Bộ phân tích · FPS")
        health_title.setObjectName("cardTitle")
        health_title.setWordWrap(True)
        health_layout.addWidget(health_title)
        self.health_labels: dict[str, QLabel] = {}
        self.health_fps_labels: dict[str, QLabel] = {}
        health_grid = QGridLayout()
        health_grid.setHorizontalSpacing(8)
        health_grid.setVerticalSpacing(8)
        health_grid.setColumnStretch(0, 1)
        for key, label in (
            ("object_detector", "YOLO"),
            ("face", "Mặt"),
            ("pose", "Tư thế"),
            ("hands", "Bàn tay"),
        ):
            row_index = len(self.health_labels)
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
            health_grid.addWidget(name, row_index, 0)
            health_grid.addWidget(status, row_index, 1)
            health_grid.addWidget(fps, row_index, 2)
            self.health_labels[key] = status
            self.health_fps_labels[key] = fps
        health_layout.addLayout(health_grid)
        rail.addWidget(health_card)

        risk_card = self._card()
        risk_layout = QVBoxLayout(risk_card)
        risk_title = QLabel("Tín hiệu cần rà soát")
        risk_title.setObjectName("cardTitle")
        risk_title.setWordWrap(True)
        risk_layout.addWidget(risk_title)
        self.risk_value = QLabel("—")
        self.risk_value.setObjectName("riskValue")
        self.risk_value.setWordWrap(True)
        risk_layout.addWidget(self.risk_value)
        self.risk_note = QLabel("AI hỗ trợ phát hiện tín hiệu; giám thị đưa ra đánh giá cuối cùng.")
        self.risk_note.setObjectName("mutedText")
        self.risk_note.setWordWrap(True)
        risk_layout.addWidget(self.risk_note)
        self.risk_history_label = QLabel("Lịch sử phiên: 0 sự kiện · 0 điểm tích lũy · đỉnh 0/100")
        self.risk_history_label.setObjectName("mutedText")
        self.risk_history_label.setWordWrap(True)
        risk_layout.addWidget(self.risk_history_label)
        rail.addWidget(risk_card)

        events_card = self._card()
        events_layout = QVBoxLayout(events_card)
        events_title = QLabel("Sự kiện phiên này")
        events_title.setObjectName("cardTitle")
        events_title.setWordWrap(True)
        events_layout.addWidget(events_title)
        self.events = QListWidget()
        self.events.setMinimumHeight(150)
        self.events.itemDoubleClicked.connect(self._open_evidence)
        events_layout.addWidget(self.events)
        rail.addWidget(events_card, 1)
        rail_container = QWidget()
        rail_container.setLayout(rail)
        rail_scroll = QScrollArea()
        rail_scroll.setObjectName("railScroll")
        rail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rail_scroll.setWidgetResizable(True)
        rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail_scroll.setWidget(rail_container)
        content.addWidget(rail_scroll, 3)
        page.addLayout(content, 1)

        self.setCentralWidget(root)

    def _card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        return card

    def _start(self) -> None:
        if self.runtime is not None and self.runtime.isRunning():
            return
        self.events.clear()
        self._event_rows.clear()
        self._analysis = None
        self._overlay = OverlayState()
        self._calibration_status = "uncalibrated"
        self._metrics = {}
        self._last_analyzers = {}
        self._analyzer_performance = {}
        self._last_error = ""
        self._risk_score, self._risk_level = 0.0, "NORMAL"
        self._risk_history = {}
        self._refresh_risk()
        self.preview.clear()
        self._last_frame_id = None
        self._preview_times.clear()
        self._preview_fps = 0.0
        self.runtime = DesktopRuntime(
            camera_index=self.camera_index.value(),
            repository_root=self.repository_root,
            preview=self.preview,
        )
        self.runtime.state_changed.connect(self._on_state)
        self.runtime.analysis_ready.connect(self._on_analysis)
        self.runtime.event_ready.connect(self._on_event)
        self.runtime.runtime_error.connect(self._on_error)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_index.setEnabled(False)
        self.app_status.setText("Đang khởi tạo")
        self.runtime.start()

    def _calibrate_head(self) -> None:
        if self.runtime is not None and self.runtime.isRunning():
            self._calibration_status = "collecting"
            self.runtime.request_head_calibration()

    def _stop(self) -> None:
        if self.runtime is None or not self.runtime.isRunning():
            return
        self.stop_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.app_status.setText("Đang dừng")
        self.runtime.request_stop()

    def _on_state(self, update: dict[str, Any]) -> None:
        kind = update.get("type")
        if kind == "loading_models":
            self.app_status.setText("Đang tải model AI")
            self.realtime_status.setText("Đang khởi tạo các bộ phân tích…")
        elif kind == "camera_ready":
            self.camera_status.setText("Webcam trực tiếp · đang tải model AI")
        elif kind == "analyzer_health":
            self._last_analyzers = update.get("analyzers", {})
            self._refresh_health()
        elif kind == "running":
            self.app_status.setText("Đang giám sát")
            self.camera_status.setText("Camera đã mở")
            self.calibrate_button.setEnabled(self._last_analyzers.get("face", {}).get("status") == "ready")
        elif kind == "metrics":
            self._metrics = update.get("metrics", {})
            self._refresh_metrics()
        elif kind == "risk":
            self._risk_score = float(update.get("score", self._risk_score))
            self._risk_level = str(update.get("level", self._risk_level))
            self._risk_history = update
            self._refresh_risk()
        elif kind == "stopped":
            self.calibrate_button.setEnabled(False)
            self.app_status.setText("Cần kiểm tra" if self._last_error else "Đã dừng")
            self.camera_status.setText("Camera chưa mở")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.camera_index.setEnabled(True)

    def _on_analysis(self, result: dict[str, Any]) -> None:
        self._analysis = result
        self._overlay.update(result)
        faces = result.get("faces", [])
        if faces:
            self._calibration_status = faces[0].get("calibration_status", "uncalibrated")
        health = result.get("analyzer_health", {})
        self._analyzer_performance = result.get("analyzer_performance", {})
        self._last_analyzers = {
            key: {
                "status": value,
                "reason": self._analyzer_performance.get(key, {}).get("reason", self._last_analyzers.get(key, {}).get("reason", "")),
                "device": self._last_analyzers.get(key, {}).get("device", ""),
            }
            for key, value in health.items()
        }
        self._refresh_health()

    def _on_event(self, item: dict[str, Any]) -> None:
        if item.get("action") == "evidence":
            row = self._event_rows.get(str(item.get("event_id", "")))
            if row is not None:
                if item.get("status") == "available":
                    row.setData(Qt.ItemDataRole.UserRole, str(item.get("relative_path", "")))
                    row.setToolTip(row.toolTip() + "\nẢnh đã lưu. Nhấp đúp để xem bằng chứng.")
                else:
                    row.setToolTip(row.toolTip() + "\nKhông lưu được ảnh bằng chứng; xem nhật ký kỹ thuật.")
            return
        self._risk_score = float(item.get("risk_score", self._risk_score))
        self._risk_level = str(item.get("risk_level", self._risk_level))
        self._risk_history = item
        self._refresh_risk()
        if item.get("action") not in {"opened", "correlation"}:
            return
        timestamp = str(item.get("occurred_at", ""))
        clock = local_clock(timestamp)
        event_type = str(item.get("type", "EVENT"))
        severity = str(item.get("severity", "LOW"))
        row = QListWidgetItem(f"{clock}   {event_name(event_type)}   ·   {level_name(severity)}")
        evidence_note = {
            "pending": "\nĐang lưu ảnh bằng chứng…",
            "disabled": "\nLưu ảnh bằng chứng đang tắt.",
            "queue_full": "\nHàng đợi ảnh đã đầy; khung này chưa được lưu.",
            "frame_mismatch": "\nKhông thể gắn đúng khung phân tích với sự kiện.",
            "frame_unavailable": "\nKhung ảnh phân tích không khả dụng.",
        }.get(str(item.get("evidence_status", "")), "")
        sources = item.get("source_event_ids") or []
        source_note = "\nSự kiện nguồn: " + ", ".join(str(source) for source in sources) if sources else ""
        row.setToolTip(event_description(item) + evidence_note + source_note)
        row.setForeground(QColor("#ffbf69" if severity in {"HIGH", "CRITICAL"} else "#8bd3dd"))
        self.events.insertItem(0, row)
        self._event_rows[str(item.get("id", ""))] = row
        while self.events.count() > 40:
            removed = self.events.takeItem(self.events.count() - 1)
            self._event_rows = {key: value for key, value in self._event_rows.items() if value is not removed}

    def _open_evidence(self, row: QListWidgetItem) -> None:
        relative = row.data(Qt.ItemDataRole.UserRole)
        if not isinstance(relative, str) or not relative:
            return
        storage = (self.repository_root / "storage").resolve()
        path = (storage / relative).resolve()
        if (not path.is_relative_to(storage / "sessions") or path.suffix.lower() != ".jpg"):
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Ảnh không còn khả dụng", "Không mở được ảnh; có thể ảnh đã hết thời hạn lưu.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Ảnh bằng chứng · giám thị rà soát")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setPixmap(pixmap.scaled(1000, 700, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(label)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._evidence_dialogs.append(dialog)
        dialog.finished.connect(lambda _code, current=dialog: self._evidence_dialogs.remove(current))
        dialog.show()

    def _on_error(self, message: str) -> None:
        self._last_error = message
        self.calibrate_button.setEnabled(False)
        self.app_status.setText("Cần kiểm tra")
        self.camera_status.setText("Camera chưa mở")
        self.preview_note.setText(message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_index.setEnabled(True)

    def _refresh_health(self) -> None:
        for key, label in self.health_labels.items():
            detail = self._last_analyzers.get(key, {})
            status = str(detail.get("status", "not_run"))
            reason = str(detail.get("reason", ""))
            device = str(detail.get("device", ""))
            if not device and key in {"face", "pose", "hands"}:
                device = "cpu"
            device_label = "GPU" if "cuda" in device.casefold() else "CPU" if device.casefold() == "cpu" else ""
            performance = self._analyzer_performance.get(key, {})
            fps_label = f"{float(performance['fps']):.1f}" if "fps" in performance else "—"
            display_status = performance.get("status", status) if status == "skipped" else status
            short_status = {"ready": device_label or "Chạy", "no_detection": device_label or "Chạy",
                            "not_run": "Chờ", "loading": "Nạp", "skipped": "Chờ",
                            "busy": "Bận", "timeout": "Chậm", "unavailable": "Thiếu",
                            "error": "Lỗi", "disabled": "Tắt"}.get(str(display_status), "Chờ")
            _set_text_if_changed(label, short_status)
            _set_text_if_changed(self.health_fps_labels[key], fps_label)
            tooltip = " · ".join(part for part in (_health_text(str(display_status), reason),
                                                   f"Thiết bị xử lý: {device}" if device else "",
                                                   health_reason(reason)) if part)
            if performance:
                tooltip += f" · Lượt suy luận gần nhất: {performance.get('latency_ms', 0):.0f} ms. FPS chỉ đếm lượt xử lý thật."
            for part in (label, self.health_fps_labels[key]):
                if part.toolTip() != tooltip:
                    part.setToolTip(tooltip)
            if label.property("health") == display_status:
                continue
            label.setProperty("health", display_status)
            label.style().unpolish(label)
            label.style().polish(label)

    def _refresh_metrics(self) -> None:
        _set_text_if_changed(self.metric_values["preview"], f"{self._preview_fps:.1f}")
        _set_text_if_changed(self.metric_values["processed"], f"{float(self._metrics.get('processed_fps', 0)):.1f}")
        _set_text_if_changed(self.metric_values["latency"], f"{float(self._metrics.get('latency_ms_p95', 0)):.0f}")
        _set_text_if_changed(self.metric_values["dropped"], str(self._metrics.get("dropped_frames", 0)))
        realtime = str(self._metrics.get("realtime_status", "WARMING_UP"))
        _set_text_if_changed(
            self.realtime_status,
            {"REALTIME": "Đang xử lý thời gian thực", "DEGRADED": "AI chưa đạt mục tiêu 20 FPS hoặc có bộ phân tích chưa sẵn sàng", "WARMING_UP": "Đang đo tốc độ sau khởi động"}.get(realtime, "Chưa xác định tốc độ")
        )

    def _refresh_risk(self) -> None:
        self.risk_value.setText(f"{self._risk_score:.1f} / 100   ·   {level_name(self._risk_level)}")
        history = self._risk_history
        self.risk_history_label.setText(
            f"Lịch sử phiên: {int(history.get('event_count', 0))} sự kiện · "
            f"{float(history.get('cumulative_score', 0)):.0f} điểm tích lũy · "
            f"đỉnh {float(history.get('peak_score', 0)):.0f}/100"
        )

    def _refresh_preview(self) -> None:
        frame = self.preview.latest()
        if frame is None:
            return
        frame_id = int(frame.get("frame_id", -1))
        captured_at_ms = int(frame.get("captured_at_monotonic_ms", 0))
        frame_age_ms = max(0, time.monotonic_ns() // 1_000_000 - captured_at_ms)
        if frame_id == self._last_frame_id:
            if frame_age_ms > 1000:
                _set_text_if_changed(self.camera_status, "Khung webcam đã cũ · kiểm tra camera")
            return
        self._last_frame_id = frame_id

        image_data = frame.get("image")
        if image_data is None:
            return
        height, width = image_data.shape[:2]
        image = QImage(
            image_data.data,
            width,
            height,
            int(image_data.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        if image.isNull():
            return
        image = image.scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        pixmap = QPixmap.fromImage(image)
        analysis = self._overlay.snapshot(captured_at_ms)
        if analysis is not None:
            age_ms = captured_at_ms - int(analysis.get("captured_at_monotonic_ms", 0))
            if 0 <= age_ms <= self._overlay.max_age_ms:
                _paint_overlay(pixmap, analysis, age_ms)
        self.video.setPixmap(pixmap)

        now = time.monotonic()
        self._preview_times.append(now)
        while self._preview_times and now - self._preview_times[0] > 2.0:
            self._preview_times.popleft()
        if len(self._preview_times) > 1:
            elapsed = self._preview_times[-1] - self._preview_times[0]
            self._preview_fps = (len(self._preview_times) - 1) / elapsed if elapsed > 0 else 0.0
        self._refresh_metrics()
        if frame_age_ms > 1000:
            _set_text_if_changed(self.camera_status, "Khung webcam đã cũ · kiểm tra camera")
        else:
            _set_text_if_changed(self.camera_status, f"Đang hiển thị · {self._preview_fps:.1f} FPS")
        _set_text_if_changed(
            self.preview_note,
            self._last_error or {"collecting": "Hiệu chỉnh: nhìn thẳng màn hình, giữ đầu yên khoảng 2 giây.",
             "failed": "Chưa hiệu chỉnh được: giữ đầu yên rồi nhấn Hiệu chỉnh hướng đầu lần nữa.",
             "uncalibrated": "Chưa bật cảnh báo quay đầu: hãy nhìn thẳng và nhấn Hiệu chỉnh hướng đầu. Điện thoại vẫn được kiểm tra.",
             "calibrated": "Đã hiệu chỉnh hướng đầu. Khung nét đứt là kết quả tạm giữ; AI FPS đo riêng với preview."
             }.get(self._calibration_status, "Khung AI hiển thị kết quả gần nhất và độ trễ."),
        )

    def closeEvent(self, event: Any) -> None:
        if self.runtime is not None and self.runtime.isRunning():
            self.runtime.request_stop()
            self.runtime.wait(8000)
        super().closeEvent(event)


def _paint_overlay(pixmap: QPixmap, analysis: dict[str, Any], age_ms: int) -> None:
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    width, height = pixmap.width(), pixmap.height()
    detections = analysis.get("objects", [])
    for item in detections:
        label = str(item.get("class", "vật thể"))
        if label.casefold() not in {"person", "cell phone", "book"}:
            continue
        color = QColor("#56e39f") if label.lower() == "person" else QColor("#ff6b6b") if "phone" in label.lower() else QColor("#ffc857")
        pending = item.get("track_confirmed") is False
        suffix = " · đang xác nhận" if pending else " · tạm giữ" if item.get("held") else ""
        if pending:
            color = QColor("#ffc857")
        _draw_box(painter, item.get("bbox", {}), width, height, color, f"{_object_name(label)} · {float(item.get('confidence', 0)):.0%}{suffix}", dashed=pending or bool(item.get("held")))
    for face in analysis.get("faces", []):
        pose = f"Mặt · xoay {float(face.get('yaw', 0)):.0f}° · cúi/ngẩng {float(face.get('pitch', 0)):.0f}°"
        if not face.get("pose_valid", True):
            pose = "Mặt · chưa đo được hướng đầu"
        if face.get("held"):
            pose += " · tạm giữ"
        _draw_box(painter, face.get("bbox", {}), width, height, QColor("#5de1e6"), pose, thickness=3, dashed=bool(face.get("held")))
    for pose in analysis.get("poses", []):
        _draw_pose(painter, pose.get("landmarks", []), width, height, dashed=bool(pose.get("held")))
    for hand in analysis.get("hands", []):
        held = bool(hand.get("held"))
        _draw_box(painter, hand.get("bbox", {}), width, height, QColor("#d29bff"), "Bàn tay · tạm giữ" if held else "Bàn tay", dashed=held)
    painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
    painter.fillRect(12, 12, 226, 28, QColor(6, 17, 31, 205))
    painter.setPen(QColor("#dce8f4"))
    painter.drawText(22, 31, f"AI · khung {analysis.get('frame_id', '—')} · trễ {max(0, age_ms)} ms")
    painter.end()


def _draw_box(painter: QPainter, bbox: dict[str, Any], width: int, height: int, color: QColor, label: str, *, thickness: int = 2, dashed: bool = False) -> None:
    x = max(0.0, min(1.0, float(bbox.get("x", 0))))
    y = max(0.0, min(1.0, float(bbox.get("y", 0))))
    box_width = max(0.0, min(1.0 - x, float(bbox.get("width", 0))))
    box_height = max(0.0, min(1.0 - y, float(bbox.get("height", 0))))
    if box_width <= 0 or box_height <= 0:
        return
    left, top = round(x * width), round(y * height)
    rect_width, rect_height = round(box_width * width), round(box_height * height)
    pen = QPen(color, thickness)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawRect(left, top, rect_width, rect_height)
    font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
    painter.setFont(font)
    text_width = painter.fontMetrics().horizontalAdvance(label) + 14
    text_y = max(20, top)
    painter.fillRect(left, text_y - 20, text_width, 20, QColor(color.red(), color.green(), color.blue(), 225))
    painter.setPen(QColor("#07111f"))
    painter.drawText(left + 7, text_y - 6, label)


def _draw_pose(painter: QPainter, landmarks: list[dict[str, Any]], width: int, height: int, *, dashed: bool = False) -> None:
    points = {int(item.get("index", -1)): item for item in landmarks if float(item.get("visibility", 0)) >= 0.35}
    connections = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28))
    pen = QPen(QColor("#56e39f"), 2)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    for start, end in connections:
        if start not in points or end not in points:
            continue
        a, b = points[start], points[end]
        painter.drawLine(round(float(a["x"]) * width), round(float(a["y"]) * height), round(float(b["x"]) * width), round(float(b["y"]) * height))


def _object_name(label: str) -> str:
    return {"person": "Người", "cell phone": "Điện thoại", "book": "Sách"}.get(label.lower(), label)


def _health_text(status: str, reason: str) -> str:
    labels = {
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
    text = labels.get(status, "Chưa xác định")
    if status in {"unavailable", "error"} and reason and ("MODEL_MISSING" in reason or "FileNotFoundError" in reason):
        text = "Thiếu model"
    return text


def _set_text_if_changed(label: QLabel, value: str) -> None:
    if label.text() != value:
        label.setText(value)


def _stylesheet() -> str:
    return """
        QWidget { background: #07111f; color: #edf3f8; font-family: 'Segoe UI'; font-size: 13px; }
        QLabel#eyebrow { color: #72d6cc; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
        QLabel#pageTitle { color: #f5f8fb; font-size: 26px; font-weight: 700; }
        QFrame#card { background: #0c1a2b; border: 1px solid #1d344d; border-radius: 14px; padding: 10px; }
        QFrame#metricTile { background: #102238; border: 1px solid #1c344c; border-radius: 8px; }
        QScrollArea#railScroll { background: transparent; border: none; }
        QLabel#cardTitle { font-size: 15px; font-weight: 700; color: #f2f6fa; }
        QLabel#statusBadge, QLabel#mutedBadge { padding: 7px 11px; border-radius: 10px; background: #13283b; color: #a9c0d5; }
        QLabel#statusBadge { color: #88e1bb; }
        QLabel#videoSurface { background: #020812; border: 1px solid #203a55; border-radius: 12px; color: #8fa7be; font-size: 15px; }
        QLabel#mutedText { color: #91a9bf; font-size: 11px; }
        QLabel#metricValue { color: #f7fafc; font-size: 22px; font-weight: 700; }
        QLabel#metricLabel { color: #91a9bf; font-size: 11px; }
        QLabel#healthName, QLabel#healthState, QLabel#healthFps { font-size: 11px; }
        QLabel#healthFps { color: #a9c0d5; }
        QLabel#healthState[health='ready'] { color: #79e2af; }
        QLabel#healthState[health='unavailable'], QLabel#healthState[health='timeout'], QLabel#healthState[health='error'] { color: #ff9d91; }
        QLabel#healthState { color: #a9c0d5; }
        QLabel#riskValue { font-size: 24px; font-weight: 700; color: #f6c766; }
        QPushButton { border: 1px solid #29435d; border-radius: 9px; background: #11243a; padding: 10px 16px; font-weight: 650; }
        QPushButton:hover { background: #18324d; }
        QPushButton:disabled { color: #687b8d; background: #0b1725; }
        QPushButton#primaryButton { color: #06151a; background: #45d39a; border: none; }
        QPushButton#primaryButton:hover { background: #63e2ad; }
        QPushButton#stopButton { color: #ffd1cb; background: #351d29; border-color: #613343; }
        QSpinBox { padding: 8px; border-radius: 8px; border: 1px solid #29435d; background: #081525; }
        QListWidget { background: transparent; border: none; color: #d7e3ec; }
        QListWidget::item { padding: 7px 3px; border-bottom: 1px solid #1b2e42; }
        QScrollBar:vertical { width: 8px; background: #0a1624; }
        QScrollBar::handle:vertical { background: #29435d; border-radius: 4px; }
    """


__all__ = ["MainWindow"]
