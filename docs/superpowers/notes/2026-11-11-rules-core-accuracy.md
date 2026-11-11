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
| Bước 1: LAP (mặc định, không kiểm tĩnh) | 8/9 | 5/6 | 3/14 | 2 | 20 | 3,0 | Khuỷu tay thấp cho cổ tay khuất; giấy nằm trên bàn chặn LAP. Trùng số gốc (replay: 9/9, golden không đổi). |
| Bước 1: LAP, `lap_still_widths=0,5` (thử) | 8/9 | 5/6 | 3/14 | 2 | 19 | - | Bớt 1 cảnh báo nhưng replay chỉ còn 8/9 (mất 59.0 và 63.0), nên không chọn. |
| Bước 3: điện thoại (`phone_paper_fraction=0,6`) | 8/9 | 5/6 | 3/14 | 2 | 20 | - | Số đo không đổi: clip 1 có 2 PHONE_DETECTED trước và sau (cả hai không phủ nhãn đúng nào, không nằm trên giấy); clip 2 có 0 trước và sau, 6 cảnh báo như cũ. Luật chỉ chặn điện thoại nằm trên giấy ở >= 60% số khung; có test, golden replay không đổi (9/9). |
| Bước 1b: LAP (nguyên nhân gốc) | 7/9 | 4/5 | 2/14 | 2 | 14 | 3,6 | Đường bàn không bao giờ học được (tay viết nằm dưới đáy khung ghế, dưới vạch mặc định 0,8) nên mọi người cúi đầu viết đều thành "tay dưới bàn"; nay cẳng tay phải thõng (cổ tay thấp hơn khuỷu >= 0,25 bề rộng vai, `lap_forearm_drop`). LAP 9 -> 3, mất 4 ca nhầm trên tờ chấm (3.0/3.6/4.2/29.2); mất nhãn đúng 51.2 (lap chỉ đủ 3 s, bị lọc < 1 s). Kèm sửa: cảnh báo chớp nhoáng bị lọc trước bộ gộp, để không nuốt lap 60,0-65,4 s. Clip 2: 6 cảnh báo, đỏ 3, giống trước. |

Clip 2 (`LUy81HI7wS0`, tập giữ lại) đã chạy một lần để có `events.jsonl` cho Task 2: đỏ 3, vàng 3, tổng 11,7 s cho
video 103,2 s. Chưa có nhãn nên chưa có số chấm.

Clip 2 sau bước 1 (tập giữ lại, chưa có nhãn): 6 cảnh báo mở (TURN_AROUND 2, LOOK_AT_NEIGHBOR 3, HEADS_TOGETHER 1),
đỏ 3, vàng 3, không có LOOK_AT_LAP; giống hệt trước bước 1. Kiểm tĩnh cổ tay: replay cho thấy dao động khung xương của
ca lap thật (0,25 đến 0,88 bề rộng ghế) lớn hơn ca nhầm 3.0 (0,16), nên tắt mặc định (`lap_still_widths=None`).
