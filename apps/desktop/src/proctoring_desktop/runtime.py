"""Qt worker thread hosting the headless :class:`SessionEngine`."""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from pathlib import Path

from proctoring_core.types import ExamMode
from proctoring_runtime.session_engine import LatestPreview, SessionCallbacks, SessionEngine
from PySide6.QtCore import QThread, Signal


class DesktopRuntime(QThread):
    """Run one session off the UI thread; UI updates contain metadata, never full frames."""

    state_changed = Signal(dict)
    analysis_ready = Signal(dict)
    event_ready = Signal(dict)
    runtime_error = Signal(str)

    def __init__(self, *, camera_index: int, repository_root: Path, preview: LatestPreview,
                 exam_mode: ExamMode = ExamMode.COMPUTER) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.repository_root = repository_root
        self.preview = preview
        self.engine = SessionEngine(
            repository_root=repository_root,
            camera_index=camera_index,
            preview=preview,
            exam_mode=exam_mode,
            callbacks=SessionCallbacks(
                on_state=self.state_changed.emit,
                on_analysis=self.analysis_ready.emit,
                on_event=self.event_ready.emit,
            ),
        )
        self.session_id = self.engine.session_id

    def request_stop(self) -> None:
        self.engine.request_stop()

    def request_head_calibration(self) -> None:
        self.engine.request_head_calibration()

    def begin_gaze_calibration(self) -> None:
        self.engine.begin_gaze_calibration()

    def set_gaze_target(self, name: str | None) -> None:
        self.engine.set_gaze_target(name)

    def finish_gaze_calibration(self) -> None:
        self.engine.finish_gaze_calibration()

    def cancel_gaze_calibration(self) -> None:
        self.engine.cancel_gaze_calibration()

    async def _run_async(self) -> None:
        await self.engine.run()

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            error_log = self.repository_root / "storage" / "runtime-errors.log"
            try:
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with error_log.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n[{datetime.now().astimezone().isoformat()}] Lỗi giám sát\n")
                    stream.write(traceback.format_exc())
                detail = f" Chi tiết kỹ thuật đã lưu tại {error_log}."
            except OSError:
                detail = " Không ghi được tệp lỗi; hãy kiểm tra quyền ghi thư mục storage."
            reason = ("CUDA chưa sẵn sàng; hãy chạy bằng scripts/run_gpu.ps1."
                      if "CUDA" in str(exc) else "Không thể tiếp tục giám sát; hãy kiểm tra camera và model AI.")
            self.runtime_error.emit(reason + detail)
        finally:
            self.preview.clear()
            self.state_changed.emit({"type": "stopped"})


__all__ = ["DesktopRuntime", "LatestPreview"]
