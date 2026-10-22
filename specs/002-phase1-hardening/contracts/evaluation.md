# Hợp đồng: bộ phát lại và đánh giá

Công cụ dòng lệnh cục bộ, không có dịch vụ mạng.

## Lệnh

```powershell
uv run --no-sync python scripts/evaluate.py --set <path/to/set.yaml> [--case <case_id> ...]
    [--mode paper|computer] [--rate 8] [--limit-seconds N] [--baseline <report.json>]
    [--out storage/evaluation] [--save-events]
```

- Mã thoát: `0` chạy xong; `2` lỗi đầu vào (tệp thiếu, nhãn sai, ánh xạ thiếu); `3` mô hình thiếu hoặc
  sai SHA-256; `1` lỗi khác.
- In bảng tóm tắt tiếng Việt ra stdout; ghi `report.json` (máy đọc) và `report.md` (người đọc).
- `--limit-seconds` chỉ phát lại N giây đầu mỗi video (vòng lặp nhanh); nhãn sau mốc này bị bỏ.
- Đặt `PROCTORING_AI_DEVICE=0` để YOLO chạy CUDA (dùng `.venv-gpu`).
- `--save-events` ghi thêm `events/<case_id>.jsonl` theo định dạng golden bên dưới.

## Tệp bộ đánh giá (`set.yaml`)

```yaml
name: oep-webcam
source: "OEP dataset, Atoum et al., IEEE TMM 2017 — chỉ dùng đánh giá nội bộ"
root: E:/datasets/oep/OEP database
mapping: oep_mapping.yaml          # tương đối theo set.yaml
exam_mode: computer
cases:
  - id: subject1
    video: subject1/<username>1.avi  # glob được: subject1/*1.avi
    labels: subject1/gt.txt
    label_format: oep               # oep | csv
  - id: paper-normal-01
    video: E:/datasets/local/paper-normal-01.mp4
    labels: E:/datasets/local/paper-normal-01.csv
    label_format: csv
    exam_mode: paper                # ghi đè theo ca
```

## Định dạng nhãn

- `oep`: mỗi dòng `MMSS<TAB>MMSS<TAB>code` (ví dụ `0455	0512	2`); dòng trống bị bỏ qua.
- `csv`: header `start,end,label`; thời gian `mm:ss`, `mm:ss.fff` hoặc số giây; `label` là `EventType`
  hoặc mã nguồn có trong ánh xạ.

## Ánh xạ (`oep_mapping.yaml`)

```yaml
version: 1
codes:
  "1": { events: [DOCUMENT_DETECTED, LOOK_DOWN, GAZE_OFF_SCREEN], note: "<xác minh>" }
  "2": { events: [POSSIBLE_TALKING, MULTIPLE_PERSON_DETECTED, HEAD_TURN_LEFT, HEAD_TURN_RIGHT], note: "<xác minh>" }
  "6": { ignore: true }
tolerance_ms: 2000
```

- Một khoảng nhãn có nhiều `events` nghĩa là bất kỳ loại nào trong danh sách khớp đều tính là phát hiện
  khoảng đó (hành vi tổng hợp như "hỏi người trong phòng" biểu hiện qua nhiều tín hiệu). Độ phủ theo
  loại được tính cho từng loại riêng và cho nhóm "any".
- Sự kiện của loại không xuất hiện trong ánh xạ của ca được tính báo nhầm trên toàn bộ thời lượng ca.

## `report.json`

```json
{
  "schema_version": "1.0",
  "set": "oep-webcam",
  "created_at": "2026-10-16T10:00:00+07:00",
  "config_sha256": "…", "policy_sha256": "…",
  "models": {"yolo11n": "8.3.0", "mediapipe-face-landmarker": "sha256:64184e…"},
  "hardware": {"cpu": "…", "gpu": "…", "device": "cuda:0|cpu"},
  "analysis_rate_hz": 8,
  "cases": [{"id": "subject1", "frames_analyzed": 9600, "ai_fps": 7.4,
             "per_type": {"PHONE_DETECTED": {"tp": 3, "fn": 1, "fp": 2, "duplicates": 0,
                          "coverage": 0.75, "false_alerts_per_min": 0.08, "median_delay_ms": 900}}}],
  "totals": {"PHONE_DETECTED": {"…": "…"}, "ANY": {"…": "…"}},
  "baseline_delta": {"PHONE_DETECTED": {"false_alerts_per_min": -0.05, "coverage": 0.0}}
}
```

## Golden sự kiện (hồi quy tái cấu trúc)

Mỗi dòng JSONL: `{"t": <video_ms>, "type": "...", "action": "opened|updated|closed|correlation",
"severity": "...", "risk": <làm tròn 2 chữ số>}`. Không chứa UUID hay thời gian tường. Hai lần chạy
cùng đầu vào phải giống hệt từng byte.
