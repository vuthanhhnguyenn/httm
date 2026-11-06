import csv
import re
from pathlib import Path

from proctoring_runtime.report import write_report


def _ev(i, level, ms, **kw):
    d = {"id": i, "action": "opened", "type": "HEAD_TURN", "seat_id": "S1",
         "video_offset_ms": ms, "level": level, "level_reason": "lý do"}
    d.update(kw)
    return d


def _events():
    return [
        _ev("a", "yellow", 5000),
        _ev("b", "red", 9000, label_vi="Nhãn <b>x</b>"),
        _ev("c", "yellow", 1000),
        {"id": "m", "action": "merged", "merged_into": "a", "video_offset_ms": 6000},
        {"id": "z", "action": "closed", "video_offset_ms": 7000},
    ]


def test_report_rows_order_and_csv(tmp_path: Path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "b.jpg").write_bytes(b"x")
    write_report(tmp_path, _events(), video_name="v.mp4",
                 timing={"total_s": 10, "video_s": 20, "ratio": 0.5})
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    rows = re.findall(r'<tr class="ev"[^>]*data-level="(\w+)"', html)
    assert rows == ["red", "yellow", "yellow"]
    assert "http://" not in html and "https://" not in html
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    assert 'src="evidence/b.jpg"' in html
    assert "không có ảnh" in html
    assert "lặp lại 1 lần" in html
    assert "annotated.mp4" in html
    with open(tmp_path / "events.csv", encoding="utf-8-sig", newline="") as f:
        data = list(csv.reader(f))
    assert data[0] == ["level", "type", "label_vi", "time_s", "seat_id", "reason", "evidence"]
    assert len(data) == 4
    assert data[1][0] == "red" and data[1][3] == "9.0" and data[1][6] == "evidence/b.jpg"
    assert (tmp_path / "events.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_empty(tmp_path: Path):
    write_report(tmp_path, [], video_name="v.mp4", timing={})
    assert "Không có sự kiện" in (tmp_path / "report.html").read_text(encoding="utf-8")
    with open(tmp_path / "events.csv", encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.reader(f))) == 1
