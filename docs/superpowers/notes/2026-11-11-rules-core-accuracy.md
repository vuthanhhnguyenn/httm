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
| Bước 2: sự kiện cặp | 7/9 | 4/5 | 2/14 | 2 | 14 | 3,0 | Chỉ phần PASSING: `hands_only` cần cổ tay một bên nằm trong ô ghế bên kia >= 2 khung của đoạn (`passing_inside_frames`, mọi hàng, thay `far_row_inside_frames`). Số đo không đổi (lượt chạy mọi ghế không có PASSING nào trước và sau; replay 9/9, golden không đổi). Phần HEADS_TOGETHER "ngẩng đầu >= 50% số khung" **không commit** (luật dừng): bắt được 7/9 -> 5/9, đỏ 2 -> 1, tổng 14 -> 10, vì nó bỏ HEADS 26,4 s (đỏ, đúng, phủ 3 nhãn): người quay lại đưa đồ nên không thấy mũi (13/18 khung không có head_lift). Không có ngưỡng nào giữ được 26,4 s mà vẫn bỏ được ca nhầm (bỏ khung không thấy mũi khỏi mẫu số: 26,4 s chỉ còn 1/5 khung ngẩng). Clip 2: 6 cảnh báo, đỏ 3, giống trước (có phần HEADS cũng vậy). |
| Quay đầu 1,5/1/0,75 s: 1,5 s (chọn, giữ mặc định) | 7/9 | 4/5 | 2/14 | 2 | 14 | 2,8 | `ReviewRuleConfig.turn_hold_ms` (chỉ tầng kỹ; tầng nhanh giữ `CctvRuleConfig`). Lọc "ít nhất 1 s" của lượt mọi ghế nay theo loại (`require_sustained(min_ms_by_type=...)`): các loại quay đầu chỉ cần bằng giữ quay đầu khi nó < 1 s. Clip 2: 6 cảnh báo, đỏ 3. |
| Quay đầu 1,5/1/0,75 s: 1 s (thử) | 7/9 | 4/5 | 2/14 | 2 | 16 | - | Thêm 1 LOOK_AT_NEIGHBOR và 1 TURN_AROUND, cả hai không phủ nhãn nào. Replay: 9/9 vẫn phủ, golden đổi. Clip 2: 11 cảnh báo (LOOK_AT_NEIGHBOR 7, TURN_AROUND 3, HEADS 1), đỏ 2. |
| Quay đầu 1,5/1/0,75 s: 0,75 s (thử) | 7/9 | 4/5 | 2/14 | 2 | 16 | - | Như 1 s trên clip 1. Clip 2: 12 cảnh báo (LOOK_AT_NEIGHBOR 8), đỏ 2. Đúng/đã chấm bằng nhau ở cả ba mức, nên giữ 1,5 s (ít cảnh báo nhất, golden không đổi). |
| Đỏ từ 2 s | 7/9 | 4/5 | 2/14 | 7 | 14 | - | `require_sustained(red_ms)` 3200 -> 2000 (Nguyên quyết). Đỏ clip 1: 2 -> 7. Trước: cả 2 đỏ (LOOK_AT_NEIGHBOR 47,0 s; HEADS 26,4 s) đều phủ nhãn đúng. Sau: 5 đỏ mới (LOOK_AT_NEIGHBOR 49,0 s; HEADS 37,6/42,2/44,0/67,6 s) không phủ nhãn nào (đúng lẫn nhầm), tức chưa được chấm; 37,6/49,0/67,6 nằm trên tờ chấm rộng, chưa có kết luận. Chỉ 2/7 đỏ nằm trên nhãn đúng. Clip 2: đỏ 3 -> 3 (cả ba LOOK_AT_NEIGHBOR đã dài >= 3,2 s), 6 cảnh báo. |
| Bước 4: gán người (đo SAI, bỏ) | 7/9 | 4/5 | 2/14 | 7 | 14 | - | Số này KHÔNG có giá trị: `--reuse-fast` dùng lại dòng thời gian đã lưu khung đã gán sẵn, nên `assign_poses` mới chưa chạy. |
| Bước 4: gán người (dựng lại, ngưỡng chung 0,5 theo khung người) | 3/9 | 2/3 | 2/14 | 4 | 11 | - | Luật dừng. Dòng thời gian dựng lại hẳn. Nguyên nhân: khung người cao hơn nhiều so với ô ghế (0,27 so với 0,16 ở ghế 52d38d) nên phần chung < 0,5 và ghế mất người ở 14-26 khung quanh LOOK_AT_LAP 58-63 s. |
| Bước 4: gán người (dựng lại, chỉ tính khung người tới đáy ô) | 5/9 | 3/4 | 2/14 | 2 | 11 | - | Vẫn dưới ngưỡng 6/9 (luật dừng), nên `assign_by_body` mặc định **False** (đo âm). Mất HEADS_TOGETHER ở (0,21; 0,40) và (0,25; 0,27) vì người ở đầu y = 0,385 (rìa giữa ô 98e74d và 82f648) nay thuộc 82f648; luật cũ (duyệt ghế theo thứ tự, đầu trong ô) đưa họ cho 98e74d, theo thứ tự chứ không theo hình học. Hai bản dựng khác nhau còn lệch nhẹ (529 khung cùng người lệch < 0,02) nên có nhiễu giữa các lần chạy, nhưng không đổi ghế của ai. |

Clip 2 (`LUy81HI7wS0`, tập giữ lại) đã chạy một lần để có `events.jsonl` cho Task 2: đỏ 3, vàng 3, tổng 11,7 s cho
video 103,2 s. Chưa có nhãn nên chưa có số chấm.

Clip 2 sau bước 1 (tập giữ lại, chưa có nhãn): 6 cảnh báo mở (TURN_AROUND 2, LOOK_AT_NEIGHBOR 3, HEADS_TOGETHER 1),
đỏ 3, vàng 3, không có LOOK_AT_LAP; giống hệt trước bước 1. Kiểm tĩnh cổ tay: replay cho thấy dao động khung xương của
ca lap thật (0,25 đến 0,88 bề rộng ghế) lớn hơn ca nhầm 3.0 (0,16), nên tắt mặc định (`lap_still_widths=None`).
