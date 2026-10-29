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
| Sau phase 2d | 5/6 | 0/7 | chưa đo |

- Nhãn còn sót: 0:02 (hai người sau bên trái) rơi vào 3 giây đầu clip, khi chưa có mốc tư thế bình thường
  nào; khi dùng thật camera bật trước giờ thi.
- Nhãn A, B, C (0:35, 0:44, 0:57) được thêm sau khi Nguyên xem ảnh cắt các sự kiện ngoài nhãn cũ.
- Chỉ có một clip có nhãn: ngưỡng có thể khớp quá với clip này. Cần thêm ít nhất một clip phòng thi khác.
