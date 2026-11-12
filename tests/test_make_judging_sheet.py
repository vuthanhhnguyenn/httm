"""``scripts/make_judging_sheet.py``: which alerts still need Nguyên's verdict, and the sheet itself."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from proctoring_ai.multi.cctv.review_score import Verdict, load_verdicts

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("make_judging_sheet", ROOT / "scripts" / "make_judging_sheet.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opened(event_id, kind, ms, head, seat="s1"):
    return {"action": "opened", "id": event_id, "type": kind, "video_offset_ms": ms, "head": list(head),
            "seat_id": seat, "level": "yellow"}


def _closed(event_id, ms):
    return {"action": "closed", "id": event_id, "video_offset_ms": ms}


def test_sheet_lists_only_unjudged_alerts() -> None:
    records = [_opened("a", "LOOK_AT_LAP", 30000, (0.7, 0.7)), _closed("a", 32000),
               _opened("b", "PHONE_DETECTED", 10000, (0.2, 0.2)), _closed("b", 11000),
               _opened("c", "TURN_AROUND", 50000, (0.4, 0.4)), _closed("c", 52000)]
    verdicts = [Verdict("LOOK_AT_LAP", 31.0, ((0.7, 0.71),), True)]
    found = _module().select_candidates(records, verdicts, limit=30)
    assert [(c["id"], c["type"], c["t_s"]) for c in found] == [("b", "PHONE_DETECTED", 10.0),
                                                                ("c", "TURN_AROUND", 50.0)]
    assert all(c["heads"] for c in found)


def test_sheet_merges_alerts_on_same_head_within_2s() -> None:
    records = [_opened("a", "LOOK_AT_LAP", 10000, (0.5, 0.5)), _closed("a", 10500),
               _opened("b", "TURN_AROUND", 11000, (0.505, 0.5)), _closed("b", 11500),
               _opened("c", "TURN_AROUND", 11000, (0.9, 0.2))]
    found = _module().select_candidates(records, [], limit=30)
    assert len(found) == 2 and found[0]["id"] == "a" and found[0]["t_s"] == 10.0


def test_limit_spreads_over_time() -> None:
    records = [_opened(str(i), "LOOK_AT_LAP", i * 10000, (0.1 * (i % 9 + 1), 0.5)) for i in range(20)]
    found = _module().select_candidates(records, [], limit=5)
    assert len(found) == 5 and [c["t_s"] for c in found] == sorted(c["t_s"] for c in found)
    assert found[0]["id"] == "0" and found[-1]["id"] == "19"


def test_main_writes_sheets_of_at_most_12_cells_and_unjudged_yaml(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))  # type: ignore[attr-defined]
    for _ in range(200):
        writer.write(np.full((180, 320, 3), 90, np.uint8))
    writer.release()
    run = tmp_path / "run"
    run.mkdir()
    records = [_opened(str(i), "LOOK_AT_LAP", 1000 + i * 1000, (0.05 + 0.065 * i, 0.4)) for i in range(14)]
    (run / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "sheets"
    code = _module().main([str(video), str(run / "events.jsonl"), "--out", str(out), "--limit", "14"])
    assert code == 0
    assert sorted(p.name for p in out.glob("sheet-*.jpg")) == ["sheet-1.jpg", "sheet-2.jpg"]
    payload = yaml.safe_load((out / "judging.yaml").read_text(encoding="utf-8"))
    assert len(payload["verdicts"]) == 14 and all(v["verdict"] is None for v in payload["verdicts"])
    assert load_verdicts(out / "judging.yaml") == []
