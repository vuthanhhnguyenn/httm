# Hướng dẫn khởi chạy ứng dụng desktop

## Yêu cầu

- Windows 10/11 64-bit.
- Python 3.12 và `uv`.
- Webcam được Windows nhận diện; đóng ứng dụng khác đang giữ webcam.
- Dung lượng trống để tải model YOLO và MediaPipe. Lần tải model cần mạng; sau đó xử lý camera chạy cục bộ.

## Cài dependency

Chạy trong PowerShell tại thư mục gốc repo:

```powershell
uv sync --no-default-groups --group desktop --extra opencv-python
```

## Tải và xác minh model

```powershell
uv run --no-default-groups --group desktop --extra opencv-python python scripts/download_model.py
uv run --no-default-groups --group desktop --extra opencv-python python scripts/verify_models.py
```

YOLO có điều kiện license riêng. Kiểm tra `docs/licensing.md` trước khi phân phối hoặc dùng thương mại.

## Mở ứng dụng

```powershell
uv run --no-default-groups --group desktop --extra opencv-python python apps/desktop/main.py
```

Chọn camera trong danh sách, nhấn **Bắt đầu giám sát**, chờ trạng thái analyzer rồi xác nhận khung webcam đang cập nhật. Nhấn **Dừng phiên** để giải phóng thiết bị.

## Đọc các chỉ số

- **Preview FPS**: frame webcam thực sự được vẽ mỗi giây.
- **AI FPS**: frame suy luận hoàn tất mỗi giây; thường thấp hơn preview.
- **Latency** và **Dropped frames**: độ trễ suy luận và số frame bỏ do AI không theo kịp camera.
- Trạng thái YOLO/face/pose/hands phải là sẵn sàng hoặc trạng thái lỗi cụ thể; trạng thái camera Connected không có nghĩa analyzer đã hoạt động.
- BBox/keypoint được ẩn nếu kết quả quá cũ, tránh vẽ khung trên hình không còn khớp.

## Dữ liệu cục bộ

Event metadata được ghi tại `storage/sessions/<session-id>/events.jsonl`. Ảnh
preview chỉ giữ trong RAM; ảnh bằng chứng của event có thể được lưu theo cấu
hình/retention. Không cần khởi chạy tiến trình phụ nào.

## Khi gặp lỗi

- Không mở được camera: thử device index khác, đóng Zoom/Teams/Camera và kiểm tra quyền camera của Windows.
- Analyzer unavailable: xác minh model đã tải, SHA-256 khớp manifest, rồi đọc lý do hiển thị trên giao diện.
- Preview FPS thấp: giảm resolution hoặc target FPS trong `configs/default.yaml`.
- AI FPS thấp: giảm `detection.image_size`, kiểm tra GPU/runtime; không tăng số FPS hiển thị giả tạo.
- Điện thoại không có khung: kiểm tra ánh sáng/khoảng cách, class confidence và giới hạn của model COCO.

## Kiểm tra mã nguồn

Các lệnh kiểm tra tự động của repository được cấu hình trong CI. Không dùng benchmark FPS làm tiêu chí đạt nếu chưa chạy trên máy camera/GPU mục tiêu và lưu cấu hình phần cứng cùng kết quả.

## Kịch bản xác nhận mở rộng core (áp dụng sau khi triển khai)

Các tình huống dưới đây là tiêu chí kiểm thử tương lai, **chưa phải chức năng
đã có**. Dùng video/ảnh có quyền sử dụng; không ghi hình người khác khi chưa
được đồng ý. Nên bắt đầu bằng replay từ file để kết quả lặp lại được, sau đó
mới thử webcam thật. `scripts/replay.py` hiện chỉ đọc fixture/manifest, chưa
thực thi pipeline AI; P0 phải hoàn thiện runner đó trước khi coi replay là test.

1. Chạy test hiện có tại gốc repo:

   ```powershell
   .\.venv-gpu\Scripts\python.exe -m pytest -q
   ```

   Khi P0–P4 được triển khai, suite phải có test cho event episode, bằng chứng
   cùng frame, review báo nhầm, calibration invalid, gaze/MAR disabled và
   correlation không cộng trùng.
2. Với holdout bộ bài, hộp bài, ví, điều khiển và sách: kiểm tra không mở event
   điện thoại; nếu có, xem ảnh review đã vẽ đúng bbox/confidence/frame/model
   và đánh dấu `báo_nhầm`. Một vật hiện liên tục chỉ tối đa một episode.
3. Với điện thoại thật có/không ốp, ở nhiều khoảng cách/ánh sáng: đo recall,
   cảnh báo nhầm/phút trên clip âm tính, p50/p95 từ lúc vật vào hình đến event.
   So sánh với baseline trước sửa; không chỉ báo số FPS.
4. Với hiệu chỉnh đầu: thử nhìn thẳng đủ ổn định, vào phiên khi đang quay lệch,
   mặt bị che, webcam rung và giám thị hiệu chỉnh thủ công lại. Baseline không
   chắc phải hiển thị `Chưa hiệu chỉnh`, không âm thầm mở rule lệch đầu/mắt.
5. Với gaze/MAR thử nghiệm: đọc đề, nhìn bàn phím, ngáp, cười, đeo kính và quay
   đầu. Không có phán quyết “nói chuyện/gian lận” từ riêng các chỉ số này; khi
   analyzer lỗi hoặc kết quả cũ, không duy trì cảnh báo mãi.
6. Với safety/correlation: thử che camera thật, thay ánh sáng, đứng lên/ngồi
   thấp, người thứ hai xuất hiện rồi rời đi, và analyzer bị timeout. Mỗi
   compound phải liên kết đúng event nguồn/cùng người và chịu cap.

Ghi cùng mỗi lần chạy: model SHA-256, CPU/GPU, camera/resolution, policy,
preview/AI FPS, p95 inference, drop rate, dữ liệu holdout và phiên bản config.
Không lấy ngưỡng từ tài liệu đầu vào làm chuẩn khi chưa có kết quả này.
