# Sổ ghi chép: độ chính xác lõi luật (2026-11-11)

Số đo cảnh báo trên clip 1 (`EPf9O7UXRoM`, 9 nhãn đúng, 14 nhãn nhầm) bằng `scripts/score_alerts.py`.
Ghép cảnh báo với nhãn: đầu cách <= 0,06, cửa sổ [đầu - 2 s, cuối + 2 s], không xét loại.
"Đúng/đã chấm" là số cảnh báo phủ ít nhất một nhãn đúng trên số cảnh báo phủ được nhãn nào đó.

Lệnh đo (ĐO): `process_video.py <video> --all-seats --reuse-fast --device intel:cpu --out /home/claude/rc/<clip>`
rồi `score_alerts.py /home/claude/rc/<clip>/events.jsonl data/labels/<video>-*.yaml`.
`process_video.py` hiện không có cờ `--overwrite`; chạy lại vào thư mục có sẵn thì tự ghi đè, nên bỏ cờ này.
Cột "Thời gian luật" lấy `deep_stats.rules_s` trong `timing.json` (tầng kỹ cả lần chạy: 15,3 s).

| Bước | Bắt được | Đúng/đã chấm | Nhầm bị phủ | Đỏ | Tổng cảnh báo | Thời gian luật (s) | Ghi chú |
|---|---|---|---|---|---|---|---|
| Gốc (trước khi đổi luật) | 8/9 | 5/6 | 3/14 | 2 | 20 | 2,7 | Khớp số cũ của `anyalert.py`. Tầng kỹ 15,3 s, tổng 33,1 s cho video 71,7 s. |

Clip 2 (`LUy81HI7wS0`, tập giữ lại) đã chạy một lần để có `events.jsonl` cho Task 2: đỏ 3, vàng 3, tổng 11,7 s cho
video 103,2 s. Chưa có nhãn nên chưa có số chấm.
