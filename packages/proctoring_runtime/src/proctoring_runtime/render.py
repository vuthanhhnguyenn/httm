"""Render pass: read the source video once, draw seat boxes and open events, encode H.264 through an ffmpeg pipe."""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctoring_runtime.messages import event_name
from proctoring_runtime.text import blit, label_image
from proctoring_runtime.tracks import Tracks, tracks_at

RED = (0, 0, 255)
YELLOW = (0, 200, 255)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
_FFMPEG_WAIT_S = 120


@dataclass(frozen=True)
class _Event:
    seat: str
    name: str
    red: bool
    start: int
    end: int
    counted: bool = True  # False for ``merged`` repeats: drawn, but not a distinct alert


def _collect_events(events: list[dict[str, Any]], min_show_ms: int) -> list[_Event]:
    """One entry per alert id: opened time, level, and end = max(closed, opened + min_show_ms)."""

    closed: dict[str, int] = {}
    for item in events:
        if item.get("action") == "closed" and "id" in item and "video_offset_ms" in item:
            closed[str(item["id"])] = int(item["video_offset_ms"])
    seen: set[str] = set()
    out: list[_Event] = []
    for item in events:
        action = item.get("action")
        if action not in ("opened", "merged") or "video_offset_ms" not in item:
            continue
        key = str(item.get("id", id(item)))
        if key in seen:
            continue
        seen.add(key)
        start = int(item["video_offset_ms"])
        # A merged repeat has its own id and no closed record: show it for min_show_ms from its own time.
        end = max(closed.get(key, start), start + min_show_ms) if action == "opened" else start + min_show_ms
        out.append(_Event(str(item.get("seat_id", "")), event_name(str(item.get("type", ""))),
                          item.get("level") == "red", start, end, counted=action == "opened"))
    out.sort(key=lambda e: e.start)
    return out


def _draw_frame(frame: Any, t_ms: int, tracks: Tracks, events: list[_Event], last_box: dict[str, list[float]],
                size: int) -> None:
    height, width = frame.shape[:2]
    seats = tracks_at(tracks, t_ms) or {}
    for seat, row in seats.items():
        last_box[seat] = row
    active = [e for e in events if e.start <= t_ms <= e.end]
    opened = [e for e in events if e.counted and e.start <= t_ms]
    by_seat: dict[str, list[_Event]] = {}
    for event in active:
        by_seat.setdefault(event.seat, []).append(event)

    line = max(1, height // 360)
    for seat, row in seats.items():
        if seat in by_seat:
            continue
        x, y, w, h = row[:4]
        cv2.rectangle(frame, (int(x * width), int(y * height)), (int((x + w) * width), int((y + h) * height)),
                      GREY, line)
    orphans: list[tuple[str, tuple[int, int, int]]] = []
    for seat, group in by_seat.items():
        colour = RED if any(e.red for e in group) else YELLOW
        row = seats.get(seat) or last_box.get(seat)
        names = list(dict.fromkeys(e.name for e in group))
        if row is None:
            orphans.extend((name, colour) for name in names)
            continue
        x, y, w, h = row[:4]
        x0, y0, x1, y1 = int(x * width), int(y * height), int((x + w) * width), int((y + h) * height)
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, line * 3)
        top = y0
        for name in names:
            label = label_image(name, size, colour)
            top -= label.shape[0]
            blit(frame, label, x0, max(0, top))
    red_n = sum(1 for e in opened if e.red)
    top_y = 4
    counter = label_image(f"Đỏ: {red_n} · Vàng: {len(opened) - red_n}", size, WHITE)
    blit(frame, counter, 4, top_y)
    top_y += counter.shape[0]
    for name, colour in orphans:
        label = label_image(name, size, colour)
        blit(frame, label, 4, top_y)
        top_y += label.shape[0]


def _frames(video: Path, tracks: Tracks, events: list[_Event]) -> Iterator[Any]:
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        last_box: dict[str, list[float]] = {}
        index = 0
        size = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            if not size:
                size = max(14, frame.shape[0] // 28)
            _draw_frame(frame, int(index * 1000 / fps), tracks, events, last_box, size)
            index += 1
            yield frame
    finally:
        cap.release()


def _probe(video: Path) -> tuple[int, int, str]:
    cap = cv2.VideoCapture(str(video))
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        cap.release()
    return width, height, f"{fps:.5g}" if fps != round(fps) else str(int(round(fps)))


def _encode_ffmpeg(ffmpeg: str, frames: Iterator[Any], out: Path, width: int, height: int, fps: str) -> int:
    """Pipe frames into ffmpeg; returns the frame count, raises RuntimeError on any failure.

    Odd frame sizes are padded by one pixel (yuv420p needs even dimensions).
    """

    pad = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] if width % 2 or height % 2 else []

    with tempfile.TemporaryFile() as errlog:
        try:
            proc = subprocess.Popen(
                [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", fps,
                 "-i", "-", *pad, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 str(out)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errlog)
        except OSError as exc:
            raise RuntimeError(f"không chạy được ffmpeg: {exc}") from exc
        count = 0
        failure: str | None = None
        try:
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                count += 1
        except (BrokenPipeError, OSError):
            failure = "ffmpeg đóng ống giữa chừng"
        finally:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                code = proc.wait(timeout=_FFMPEG_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                code = -9
                failure = failure or "ffmpeg quá thời gian chờ"
        if code != 0 or failure:
            errlog.seek(0)
            tail = errlog.read()[-300:].decode("utf-8", "replace").strip()
            raise RuntimeError(f"{failure or 'ffmpeg lỗi'} (mã {code}): {tail}")
        return count


def _encode_mp4v(frames: Iterator[Any], out: Path, width: int, height: int, fps: str) -> int:
    """Write with OpenCV ``mp4v``; raises RuntimeError when the writer cannot be opened."""

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))  # type: ignore[attr-defined]
    if not writer.isOpened():
        raise RuntimeError("không mở được bộ ghi mp4v")
    count = 0
    try:
        for frame in frames:
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    return count


def render_video(video: Path, out: Path, tracks: Tracks, events: list[dict[str, Any]], *,
                 min_show_ms: int = 2000, ffmpeg: str | None = "ffmpeg") -> dict[str, Any]:
    """Draw seats and open events on every source frame; H.264 via ffmpeg, falling back to OpenCV ``mp4v``."""

    started = time.monotonic()
    width, height, fps = _probe(video)
    shaped = _collect_events(events, min_show_ms)
    result: dict[str, Any] = {"frames": 0, "encoder": "libx264", "error": None, "seconds": 0.0}
    make: Callable[[], Iterator[Any]] = lambda: _frames(video, tracks, shaped)  # noqa: E731
    if width <= 0 or height <= 0:
        result.update(error="không mở được video", encoder="mp4v")
    elif ffmpeg:
        try:
            result["frames"] = _encode_ffmpeg(ffmpeg, make(), out, width, height, fps)
        except RuntimeError as exc:
            result["error"] = str(exc)
    else:
        result["error"] = "không có ffmpeg"
    if result["error"] is None:
        pass
    elif width > 0 and height > 0:
        result["encoder"] = "mp4v"
        out.unlink(missing_ok=True)
        try:
            result["frames"] = _encode_mp4v(make(), out, width, height, fps)
        except RuntimeError as exc:
            result.update(frames=0, error=f"{result['error']}; {exc}")
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


__all__ = ["render_video"]
