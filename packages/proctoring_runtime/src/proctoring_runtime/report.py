"""Offline event report: one self-contained report.html plus events.csv."""
from __future__ import annotations

import csv
import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .messages import event_name

CSV_COLUMNS = ["level", "type", "label_vi", "time_s", "seat_id", "reason", "evidence"]

_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:12px;background:#f5f5f5;color:#222}
video{width:100%;max-height:60vh;background:#000}
.note{background:#fff4d6;border-left:4px solid #d9a400;padding:6px 10px;margin:6px 0}
.bar{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;align-items:center}
.bar button,.bar select{padding:6px 10px;font-size:15px}
.red{color:#b00020;font-weight:600}.yellow{color:#9a6700;font-weight:600}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:#fff;min-width:560px}
th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}
tr.ev[data-level=red]{border-left:6px solid #d32f2f}
tr.ev[data-level=yellow]{border-left:6px solid #f9a825}
img.evi{max-width:160px;height:auto}
.noimg{color:#888;font-style:italic}
a.seek{cursor:pointer;color:#0645ad;text-decoration:underline}
"""

_JS = """
var lv='all';
function apply(){var t=document.getElementById('type').value;
document.querySelectorAll('tr.ev').forEach(function(r){
var ok=(lv==='all'||r.dataset.level===lv)&&(t===''||r.dataset.type===t);
r.style.display=ok?'':'none';});}
document.querySelectorAll('button[data-lv]').forEach(function(b){
b.onclick=function(){lv=b.dataset.lv;apply();};});
document.getElementById('type').onchange=apply;
document.querySelectorAll('a.seek').forEach(function(a){
a.onclick=function(e){e.preventDefault();var v=document.getElementById('vid');
v.currentTime=parseFloat(a.dataset.s);v.play();};});
"""


def _fmt_time(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def _num(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else ""


def write_report(
    out_dir: Path, events: list[dict[str, Any]], *, video_name: str, timing: Mapping[str, Any]
) -> None:
    """Write report.html and events.csv (UTF-8 BOM) into out_dir."""
    out_dir = Path(out_dir)
    repeats: dict[str, int] = {}
    for rec in events:
        if rec.get("action") == "merged" and rec.get("merged_into"):
            key = str(rec["merged_into"])
            repeats[key] = repeats.get(key, 0) + 1
    rows: list[dict[str, Any]] = []
    for rec in events:
        if rec.get("action") != "opened":
            continue
        level = "red" if rec.get("level") == "red" else "yellow"
        code = str(rec.get("type", ""))
        rid = str(rec.get("id", ""))
        rows.append({
            "level": level,
            "type": code,
            "label": str(rec.get("label_vi") or event_name(code)),
            "time_s": float(rec.get("video_offset_ms", 0)) / 1000.0,
            "seat": str(rec.get("seat_id", "")),
            "reason": str(rec.get("level_reason", "")),
            "id": rid,
            "evidence": f"evidence/{rid}.jpg",
            "has_img": (out_dir / "evidence" / f"{rid}.jpg").is_file(),
            "repeats": repeats.get(rid, 0),
        })
    rows.sort(key=lambda r: (r["level"] != "red", r["time_s"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "events.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            writer.writerow([r["level"], r["type"], r["label"], f"{r['time_s']:.1f}",
                             r["seat"], r["reason"], r["evidence"]])

    e = html.escape
    n_red = sum(1 for r in rows if r["level"] == "red")
    n_yellow = len(rows) - n_red
    parts = [f"Đỏ: {n_red} · Vàng: {n_yellow}"]
    for key, label, suffix in (("total_s", "Thời gian xử lý", " s"),
                               ("video_s", "Độ dài video", " s"), ("ratio", "Tỉ lệ", "")):
        if key in timing and _num(timing[key]):
            parts.append(f"{label}: {_num(timing[key])}{suffix}")
    summary = " | ".join(e(p) for p in parts)
    notes = "".join(f'<p class="note">{e(str(timing[key]))}</p>' for key in ("domain_warning", "note")
                    if timing.get(key))

    types = sorted({(r["type"], r["label"]) for r in rows})
    options = '<option value="">Tất cả loại</option>' + "".join(
        f'<option value="{e(c, quote=True)}">{e(lab)}</option>' for c, lab in types)

    if rows:
        body_rows = []
        for r in rows:
            img = (f'<img class="evi" src="{e(r["evidence"], quote=True)}" alt="">'
                   if r["has_img"] else '<span class="noimg">không có ảnh</span>')
            rep = f' (lặp lại {r["repeats"]} lần)' if r["repeats"] else ""
            body_rows.append(
                f'<tr class="ev" data-level="{r["level"]}" data-type="{e(r["type"], quote=True)}">'
                f'<td class="{r["level"]}">{"Đỏ" if r["level"] == "red" else "Vàng"}</td>'
                f'<td>{e(r["label"])}{e(rep)}</td>'
                f'<td><a class="seek" href="#" data-s="{r["time_s"]:.1f}">'
                f'{_fmt_time(r["time_s"])}</a></td>'
                f'<td>{e(r["seat"])}</td><td>{e(r["reason"])}</td><td>{img}</td></tr>')
        table = ('<div class="wrap"><table><thead><tr><th>Mức</th><th>Loại</th><th>Thời điểm</th>'
                 '<th>Ghế</th><th>Lý do</th><th>Ảnh</th></tr></thead><tbody>'
                 + "".join(body_rows) + "</tbody></table></div>")
    else:
        table = "<p>Không có sự kiện</p>"

    doc = (
        '<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Báo cáo sự kiện - {e(video_name)}</title><style>{_CSS}</style></head><body>"
        f"<h1>Báo cáo sự kiện: {e(video_name)}</h1><p>{summary}</p>{notes}"
        '<video id="vid" controls preload="metadata" src="annotated.mp4"></video>'
        '<div class="bar"><button data-lv="all">Tất cả</button>'
        '<button data-lv="red">Đỏ</button><button data-lv="yellow">Vàng</button>'
        f'<select id="type">{options}</select></div>{table}'
        f"<script>{_JS}</script></body></html>")
    (out_dir / "report.html").write_text(doc, encoding="utf-8")
