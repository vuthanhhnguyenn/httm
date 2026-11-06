from __future__ import annotations

import stat
import time
from pathlib import Path

import cv2
import numpy as np
from proctoring_runtime.render import render_video
from proctoring_runtime.text import put_text

BOX = [0.4, 0.3, 0.2, 0.4, "candidate"]


def make_video(path: Path, frames: int, fps: float, size: tuple[int, int] = (160, 120)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)  # type: ignore[attr-defined]
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), 40 + i % 5, np.uint8))
    writer.release()
    return path


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def tracks_for(frames: int, fps: float) -> list[tuple[int, dict]]:
    return [(int(i * 1000 / fps), {"1": BOX}) for i in range(frames)]


def red_pixels(frame: np.ndarray) -> int:
    b, g, r = (frame[:, :, i].astype(int) for i in range(3))
    return int(((r > 180) & (g < 90) & (b < 90))[30:90, 60:110].sum())


def test_all_frames_written(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 30, 10)
    result = render_video(video, tmp_path / "out.mp4", tracks_for(30, 10), [])
    assert result["error"] is None and result["encoder"] == "libx264" and result["frames"] == 30
    assert len(read_frames(tmp_path / "out.mp4")) == 30


def test_ntsc_fps_exact(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 30, 29.97)
    result = render_video(video, tmp_path / "out.mp4", [], [])
    assert result["frames"] == 30
    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 29.97) < 0.01
    assert len(read_frames(tmp_path / "out.mp4")) == 30


def test_red_event_drawn_only_while_open(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 30, 10)
    events = [{"action": "opened", "id": "a", "seat_id": "1", "type": "PHONE_DETECTED", "level": "red",
               "video_offset_ms": 1000},
              {"action": "closed", "id": "a", "seat_id": "1", "type": "PHONE_DETECTED", "video_offset_ms": 2000}]
    render_video(video, tmp_path / "out.mp4", tracks_for(30, 10), events, min_show_ms=0)
    frames = read_frames(tmp_path / "out.mp4")
    assert red_pixels(frames[15]) > 20
    assert red_pixels(frames[5]) == 0


def test_min_show_and_missing_track_uses_last_box(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 30, 10)
    tracks = [row for row in tracks_for(30, 10) if row[0] < 1200]
    events = [{"action": "opened", "id": "a", "seat_id": "1", "type": "PHONE_DETECTED", "level": "red",
               "video_offset_ms": 1000}]
    render_video(video, tmp_path / "out.mp4", tracks, events, min_show_ms=1500)
    frames = read_frames(tmp_path / "out.mp4")
    assert red_pixels(frames[20]) > 20  # track gap: last known box
    assert red_pixels(frames[28]) == 0  # past opened + min_show


def test_unseen_seat_and_no_level_do_not_crash(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 10, 10)
    events = [{"action": "opened", "id": "a", "seat_id": "9", "type": "PHONE_DETECTED", "video_offset_ms": 0},
              {"action": "merged", "id": "b", "merged_into": "a", "seat_id": "9", "type": "PHONE_DETECTED",
               "level": "red", "video_offset_ms": 100},
              {"action": "dismissed", "seat_id": "9"}]
    result = render_video(video, tmp_path / "out.mp4", [], events)
    assert result["frames"] == 10 and result["error"] is None


def test_no_events_still_outputs(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 5, 10)
    result = render_video(video, tmp_path / "out.mp4", [], [])
    assert (tmp_path / "out.mp4").stat().st_size > 0 and result["frames"] == 5


def test_missing_ffmpeg_falls_back_to_mp4v(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 10, 10)
    result = render_video(video, tmp_path / "out.mp4", [], [], ffmpeg="khong-co")
    assert result["encoder"] == "mp4v" and result["error"] and result["frames"] == 10
    assert len(read_frames(tmp_path / "out.mp4")) == 10


def test_ffmpeg_exiting_early_does_not_hang(tmp_path: Path) -> None:
    fake = tmp_path / "fake-ffmpeg"
    fake.write_text("#!/bin/sh\nfor last; do :; done\necho partial > \"$last\"\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    video = make_video(tmp_path / "in.mp4", 200, 10, size=(640, 360))
    started = time.monotonic()
    result = render_video(video, tmp_path / "out.mp4", [], [], ffmpeg=str(fake))
    assert time.monotonic() - started < 60
    assert result["encoder"] == "mp4v" and result["error"] and result["frames"] == 200
    assert len(read_frames(tmp_path / "out.mp4")) == 200


def test_put_text_still_works() -> None:
    image = np.zeros((40, 200, 3), np.uint8)
    assert put_text(image, "Điện thoại", (2, 2)).any()


def region_has(frame: np.ndarray, colour: tuple[int, int, int], box: tuple[int, int, int, int], tol: int = 60) -> int:
    x0, y0, x1, y1 = box
    diff = np.abs(frame[y0:y1, x0:x1].astype(int) - np.array(colour)).max(axis=2)
    return int((diff < tol).sum())


def test_no_level_is_yellow_and_counter_drawn(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 10, 10)
    events = [{"action": "opened", "id": "a", "seat_id": "1", "type": "PHONE_DETECTED", "video_offset_ms": 0}]
    render_video(video, tmp_path / "out.mp4", tracks_for(10, 10), events)
    frame = read_frames(tmp_path / "out.mp4")[5]
    assert region_has(frame, (0, 200, 255), (60, 30, 110, 90)) > 20
    assert frame[0:20, 0:80].max() > 150  # counter text


def test_merged_repeat_drawn_but_not_counted(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 40, 10)
    events = [{"action": "opened", "id": "a", "seat_id": "1", "type": "PHONE_DETECTED", "level": "yellow",
               "video_offset_ms": 0},
              {"action": "closed", "id": "a", "seat_id": "1", "type": "PHONE_DETECTED", "video_offset_ms": 500},
              {"action": "merged", "id": "b", "merged_into": "a", "seat_id": "1", "type": "PHONE_DETECTED",
               "level": "red", "video_offset_ms": 3000}]
    render_video(video, tmp_path / "out.mp4", tracks_for(40, 10), events, min_show_ms=1000)
    frames = read_frames(tmp_path / "out.mp4")
    assert red_pixels(frames[35]) > 20  # repeat makes the seat red
    assert red_pixels(frames[15]) == 0  # nothing open between
    from proctoring_runtime.render import _collect_events
    assert [e.counted for e in _collect_events(events, 1000)] == [True, False]


def test_odd_dimensions_encode_with_libx264(tmp_path: Path) -> None:
    video = make_video(tmp_path / "in.mp4", 6, 10, size=(161, 121))
    result = render_video(video, tmp_path / "out.mp4", [], [])
    assert result["encoder"] == "libx264" and result["error"] is None and result["frames"] == 6


def test_unclassified_not_drawn_on_video() -> None:
    from proctoring_runtime.render import _collect_events
    events = [{"action": "opened", "id": "u", "seat_id": "1", "type": "UNCLASSIFIED_SUSPECT", "level": "yellow",
               "video_offset_ms": 1000},
              {"action": "closed", "id": "u", "seat_id": "1", "type": "UNCLASSIFIED_SUSPECT",
               "video_offset_ms": 11000},
              {"action": "opened", "id": "p", "seat_id": "1", "type": "PHONE_DETECTED", "level": "red",
               "video_offset_ms": 1000},
              {"action": "closed", "id": "p", "seat_id": "1", "type": "PHONE_DETECTED", "video_offset_ms": 11000}]
    shown = _collect_events(events, 2000)
    # Unclassified windows are report-only: on Kaggle they were 9 of 26 (clip 1) and 10 of 14 (clip 2) boxes.
    assert [(e.start, e.end) for e in shown] == [(1000, 11000)]


def test_font_paths_fall_back_to_matplotlib_dejavu(tmp_path: Path, monkeypatch) -> None:
    import importlib.util
    from types import SimpleNamespace

    from proctoring_runtime import text

    font = tmp_path / "matplotlib" / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    font.parent.mkdir(parents=True)
    font.write_bytes(b"")
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(tmp_path / "matplotlib" / "__init__.py")))
    assert text.font_paths()[-1] == str(font)
