# Kết quả đánh giá

Số liệu theo [spec 002](../specs/002-phase1-hardening/spec.md) (SC-001..SC-009). Báo cáo đầy đủ nằm ở
`storage/evaluation/<lần chạy>/report.{md,json}` (ngoài Git); bảng dưới chỉ chép tóm tắt.

## Hồ sơ chạy

| Mục | Giá trị |
|---|---|
| Máy | Windows 11, Intel Core (Family 6 Model 158), Quadro M1200 (CUDA, `.venv-gpu`) |
| Mô hình | yolo11n 8.3.0 (`image_size` 416), MediaPipe face/pose-lite/hand (SHA-256 trong manifest) |
| Nhịp phân tích | 8 lượt/giây video |
| Dữ liệu | OEP dataset (Atoum et al., IEEE TMM 2017), video webcam 640×480, 25 FPS, ~15 phút/người; chỉ dùng nội bộ |

## Trạng thái

- [x] Bộ phát lại và chấm điểm (T004–T009), golden tổng hợp trên CI (T010).
- [ ] Nghĩa mã nhãn OEP (T011) — chờ tải xong `gt.txt` của các subject có video.
- [ ] Mốc trên mã trước tái cấu trúc (T012) — chạy từ bản chụp mã trước tái cấu trúc.
- [ ] So sánh sau thay đổi và tinh chỉnh (T026, T034, T035, T040, T043, T046, T052).
- [ ] Clip tự quay cho ca OEP không có (thi giấy, bộ bài/ví, che camera) — cần người dùng quay theo
      `specs/002-phase1-hardening/quickstart.md` §3.

## Kiểm tra sơ bộ trên OEP (chưa có nhãn)

| Lần chạy | Kết quả |
|---|---|
| subject4, 60 s, GPU | Pipeline phát lại chạy đầu-cuối; thông lượng 23,5 lượt/giây (một video 15 phút ≈ 5 phút). |
| subject4, 120 s, `image_size` 416 vs 640 | Điện thoại: 1 vs 3 sự kiện; thông lượng 13,9 vs 13,3 lượt/giây. Chưa kết luận khi chưa có nhãn. |

Quan sát định tính từ khung ảnh tại sự kiện:

- Người tham gia OEP đeo **camera trên kính** (wearcam); YOLO có thể nhận thiết bị này là điện thoại. Đây là
  đặc thù của bộ dữ liệu: báo nhầm điện thoại trên OEP không phản ánh trực tiếp môi trường thi thật.
- Thí sinh OEP **cúi nhìn bàn** nhiều; khi cúi sâu MediaPipe mất mặt và "Không thấy mặt" mở sau 1 giây.
  Cần đo để quyết định ngưỡng dài hơn (khi vẫn thấy người) trong T035.

## Chế độ CCTV cả phòng (phase 2d, 2026-10-29)

Clip YouTube `EPf9O7UXRoM` (71.7 s, 1280×720, ~25 thí sinh, camera cao; chỉ dùng thử nội bộ), nhãn tay
`data/labels/EPf9O7UXRoM.yaml` (Nguyên). Chấm bằng `scripts/eval_cctv.py`, chạy
`--scene cctv --mode full --fps 5`. Trên cloud và trên M1200 cho cùng số sự kiện.

| Phiên bản | Nhãn bắt được | Sự kiện ngoài nhãn | FPS (M1200) |
|---|---|---|---|
| Trước phase 2d (`37fc290`) | 2/4 | 13/16 | 10.3 |
| Sau phase 2d, `--fps 5` | 5/6 | 0/7 | ~10 (đo ở `--fps 10`: 9.8) |
| Sau phase 2d, `--fps 10` | 5/6 | 1/8 | 9.8 |

- Nhãn còn sót: 0:02 (hai người sau bên trái) rơi vào 3 giây đầu clip, khi chưa có mốc tư thế bình thường
  nào; khi dùng thật camera bật trước giờ thi.
- Nhãn A, B, C (0:35, 0:44, 0:57) được thêm sau khi Nguyên xem ảnh cắt các sự kiện ngoài nhãn cũ.
- Chỉ có một clip có nhãn: ngưỡng có thể khớp quá với clip này. Cần thêm ít nhất một clip phòng thi khác.
- Sự kiện sai còn lại ở `--fps 10` (1:00, chị áo đen): chị đang cúi viết nhưng YOLO-pose vẫn đặt mũi và mắt
  với độ tin cậy cao lên tóc, ra hướng đầu "ngẩng, quay ngang". Lỗi của mô hình khung xương, không sửa được
  bằng ngưỡng mà không làm sót ca thật.

### Clip thứ hai: `LUy81HI7wS0` (103 s, 480×548, ~10 thí sinh, chưa có nhãn)

Phòng ghế cao, bàn cao, camera gần và cao: thấy cả chân người ngồi, nên khung người ngồi cao gấp ~3 lần
rộng (clip 1 chỉ ~1). Ngưỡng cũ coi gần như mọi người là đang đứng.

| Phiên bản | Ghế | Bị coi là đi lại | Sự kiện |
|---|---|---|---|
| `4a83249` | 10, phần lớn thời gian trống | 9 | 0 |
| Sau sửa | 10 | 2 | 3 `LOOK_AT_NEIGHBOR` (chờ Nguyên xác nhận) |

Sửa: ngưỡng "đứng" so với dáng trung vị của mọi người trong khung hình; khung nhỏ hơn một nửa chiều cao
quen của ghế (một mảnh người: đầu, tay) không được chiếm ghế; đầu cúi thấp chỉ thấy đỉnh đầu, không thấy tai,
không còn tính là `TURN_AROUND`. Clip 1 giữ nguyên kết quả ở cả `--fps 5` và `--fps 10`.

## Phase 2e: hai tầng (2026-11-04, cloud CPU, clip `EPf9O7UXRoM`)

| Tầng | Kết quả |
|---|---|
| Nhanh (`--suspects`, `--fps 5`) | sự kiện như phase 2d (5/6, 0 sai); 30 đoạn nghi vấn khoanh trúng **6/6** nhãn (cả 0:02); đoạn nghi vấn chiếm 26 % thời gian-ghế, tầng kỹ phải xem 5,6× độ dài clip |
| Kỹ (`review_suspects.py --fps 5 --warmup-ms 6000`, yolo11x-pose + yolo11x) | bắt **6/6** nhãn; 39 sự kiện, 29 ngoài nhãn; 11 đoạn bị loại; CPU 113 s/đoạn (p50) |

- Nhãn clip 1 chỉ có hành vi `look`, nên các sự kiện điện thoại / chuyền đồ / cúi nhìn đùi / tài liệu đều
  bị tính là "ngoài nhãn" dù có thể đúng. Xem ảnh `phase2e/clip1/su-kien-tang-ky.jpg` (project files).
- Báo nhầm thấy rõ: máy tính bỏ túi bị gọi là điện thoại (31,6 s); `PASSING_OBJECT` lặp cho cùng một cặp
  ở hàng sau (17–22 s); `LOOK_AT_LAP` khi đang cúi viết bình thường. Cần Nguyên xác nhận rồi chỉnh ngưỡng,
  và cần clip dàn dựng để đo (a), (b), (d).

## Phase 2f: bớt báo nhầm, tầng kỹ chạy nền (2026-11-05, cloud CPU, clip `EPf9O7UXRoM`)

Đầu ra model của tầng kỹ (30 đoạn, 5 khung/giây, `review_suspects.py --cache-dir`) được lưu lại; mọi con
số dưới đây là cùng một cache chạy qua các bộ luật khác nhau (`scripts/replay_review.py`), chấm theo 23
ảnh Nguyên đã đánh giá (`data/labels/EPf9O7UXRoM-review.yaml`: 9 đúng, 14 nhầm). Test hồi quy:
`tests/test_cctv_review_regression.py`, fixture trong `packages/proctoring_ai/tests/fixtures/cctv_clip1_review/`.

| Luật | Đúng (9) | Ảnh nhầm có cảnh báo mới (14) | Tính cả cảnh báo đang mở / đã gộp | Chưa chấm |
|---|---|---|---|---|
| `main` (phase 2e) | 9 | 12 (18 sự kiện) | 9 đúng, 12 nhầm | 0 |
| nhánh này, cả đoạn (offline) | 8 + 1 đã gộp = **9** | **6** | 9 đúng, 8 nhầm | 2 |
| nhánh này, live: khúc 10 s, 3 khung/giây, nhớ khúc trước | 7 + 2 đang mở = **9** | **4** | 9 đúng, 5 nhầm | 6 |
| live với 12 khung/đoạn (kế hoạch ban đầu) | 0 | 3 | — | 8 |

- Bỏ được: tài liệu nằm im dưới bàn khi đang viết (10,0 s, 32,0 s), một cánh tay bị gán cho hai người
  (31,3 s), hai người để tay cạnh nhau lâu mà không có vật (19–20 s), cảnh báo lặp cùng ghế cùng loại.
- Còn nhầm: `LOOK_AT_LAP` 3,0 / 28,2 / 54,4 s (cùng tư thế với các ảnh đúng 51,2–63,0 s của cùng ghế,
  pose không tách được), `PASSING_OBJECT` 17,2 s, `PHONE_DETECTED` 56,8 s, `TURN_AROUND` 58,8 s.
- 59,0 s và 63,0 s là cùng cậu bé áo đỏ vẫn để tay dưới bàn: cảnh báo 51,2 s vẫn đang hiện nên được tính.
- 12 khung/đoạn làm sót gần hết: luật cần giữ 3 s, đường mép bàn và mốc đầu cần khung sát nhau và ngữ cảnh
  trước chỗ cắt. Mặc định live đổi thành 3 khung/giây, 18 khung khởi động (6 s), tối đa 48 khung/khúc, và
  khúc sau dùng lại khung khúc trước (không tốn thêm lần chạy model). Gấp khoảng 3–4 lần số khung spec dự
  tính, nên trên M1200 có thể phải hạ model (l/m) mới đạt "trong 1 phút" — cần đo bằng `phase2f/do-m1200.ps1`.
- CPU cloud (2 model x): 246 s/đoạn p50 khi xem mọi khung 5/giây; chỉ để so sánh, không phải số trên máy thi.

### Phase 2f, clip 2 (`LUy81HI7wS0`, chưa có nhãn)

Tầng nhanh: 29 đoạn nghi vấn (83 % clip). Cache tầng kỹ 5 khung/giây chạy qua các bộ luật:

| Luật | Sự kiện tầng kỹ (trừ UNCLASSIFIED) |
|---|---|
| `main` | 5: LOOK_AT_NEIGHBOR 12,0 / 17,2 / 20,0 s, HEADS_TOGETHER 87,2 s, TURN_AROUND 89,2 s |
| nhánh này, offline | 4 (17,2 và 20,0 s cùng ghế được gộp) |
| nhánh này, live (khúc 10 s) | 6: thêm HEAD_TURN_RIGHT 54,6 s và TURN_AROUND 87,0 s (cùng người với HEADS_TOGETHER) |

Bản live thêm 2 sự kiện quay đầu vì mốc hướng đầu của từng khúc 10 s ngắn hơn cả đoạn. Ảnh để chấm:
`phase2f/clip2/su-kien.jpg` (project files).

## Phase 2g: xử lý video offline

- Độ chính xác clip 1 (hồi quy fixture): `true_covered` 9/9, `true_red` 4, `false_red` 0, `false_opened` ≤ 7.
- Tốc độ trên Kaggle (2× T4): chưa đo; sẽ ghi sau khi chạy `notebooks/kaggle_process_video.ipynb`.
- Thử trên CPU với đoạn 15 s: `video_s` 15, `total_s` 683,5 (chất lượng `balanced`, chỉ CPU). Đây không phải số đại diện cho GPU.
