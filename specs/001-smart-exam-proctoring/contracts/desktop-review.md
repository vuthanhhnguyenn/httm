# Hợp đồng giao diện cục bộ cho core mở rộng (đề xuất)

Tài liệu này định nghĩa dữ liệu giữa pipeline AI, core event, evidence và cửa sổ
PySide6. Không tạo HTTP API hoặc WebSocket. Trường mới là tùy chọn khi đọc log
cũ; khi ghi event phiên bản mới phải có provenance nếu analyzer sẵn sàng.

## Observation từ cùng một frame

- `frame_id`, `captured_at_monotonic_ms`, `captured_at_utc`, `frame_size` gắn
  chung cho mọi vật thể/mặt/pose/tay trong bundle.
- Vật thể: `class`, `confidence` trong `[0,1]`, `bbox` chuẩn hóa, `object_track_id`,
  `track_confirmed`, `model_version`, `model_sha256` và analyzer health. Track
  vật thể không được thay bằng person track chỉ vì có một người trong ảnh.
- Feature mặt nếu được bật: `yaw`, `pitch`, `mar`, `iris_ratio_left/right`,
  `quality`, `pose_valid`, `calibration_status`, `calibration_id`, timestamp.
  Thiếu/invalid phải là trạng thái rõ, không dùng 0 như kết quả đo hợp lệ.
- Không chuyển landmark thô, ảnh hoặc video vào JSONL mặc định. Feature mắt/môi
  tắt mặc định và không kích hoạt event nghiêm trọng độc lập.

## Event và evidence

- `PHONE_DETECTED` là tín hiệu cần rà soát, không phải xác nhận gian lận. UI
  hiển thị **“Nghi vấn điện thoại”** và không gọi confidence là xác suất gian lận.
- Một `episode_id` gắn với một session, object track và rule. `opened` xảy ra
  tối đa một lần cho episode; `updated` không tạo hàng cảnh báo mới; `closed`
  khi đối tượng thực sự hết hiệu lực. Gap ngắn theo config chỉ giữ cùng episode.
- Event có `event_id`, `episode_id`, `source_frame_id`, `source_object_track_id`,
  bbox, model hash, confidence, threshold/duration, analyzer health và
  `evidence_status`. Nếu không có frame phù hợp, hiển thị “Không có ảnh đúng
  frame”, không ghép ảnh mới với detection cũ.
- Ảnh review dùng chính frame mở event và vẽ overlay từ bbox đã phân tích;
  ảnh gốc vẫn có thể giữ để so sánh. Overlay ghi nhãn model, confidence,
  thời điểm và trạng thái “AI gợi ý”. Không thay đổi ảnh bằng chứng gốc sau
  khi lưu. Chỉ ghi khi evidence policy cho phép; retention kế thừa cấu hình.
- Event correlation phải có `source_event_ids`, cùng session/track nếu rule
  yêu cầu, cửa sổ/freshness, cap; không tự tạo nhãn “gian lận cộng tác”.

## Phản hồi của giám thị

- Giao diện cho phép `chưa_rà_soát`, `tín_hiệu_hợp_lệ`, `báo_nhầm`,
  `không_kết_luận`; lưu event ID, thời gian và ghi chú tùy chọn cục bộ.
- Đánh dấu `báo_nhầm` không tự sửa model trong lúc thi và không xóa evidence
  trước retention; dữ liệu chỉ vào tập huấn luyện nếu có quyền sử dụng rõ ràng.
- Nếu model/analyzer lỗi, mất mặt hoặc calibration không hợp lệ, UI phải hiện
  tình trạng chưa đánh giá thay vì suy diễn “bình thường” hay “vi phạm”.

## Tương thích và kiểm chứng

- Reader JSONL chấp nhận event cũ thiếu `episode_id`, provenance hoặc review;
  các trường thiếu hiển thị “không có dữ liệu”, không suy tạo bbox/confidence.
- Test contract: một frame -> một provenance, một object episode -> tối đa một
  `opened`, `updated` không nhân đôi điểm risk, evidence có đúng `frame_id`,
  timeout không tạo correlation, review báo nhầm không đổi dữ liệu gốc.
- Tất cả nhãn UI và log vận hành hướng giám thị bằng tiếng Việt; mã kỹ thuật
  như FPS, YOLO và event code có thể giữ nguyên.
