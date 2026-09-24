# Mô hình dữ liệu cục bộ

## Runtime session

Một session tồn tại trong tiến trình desktop và giữ UUID, tên hiển thị, nguồn webcam, snapshot config/policy, trạng thái, thời điểm bắt đầu/kết thúc và risk hiện tại. Trạng thái runtime: `CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED`; lỗi worker đưa session sang trạng thái lỗi và luôn phải giải phóng camera.

## ObservationBundle

Kết quả phân tích gắn với cùng một frame: frame ID, UTC timestamp, monotonic timestamp, kích thước ảnh, người/vật thể/mặt/pose/tay/image quality và health của analyzer. Bundle chỉ ở bộ nhớ; ảnh gốc không được serialize vào event log.

## Event metadata

Event được tạo sau khi temporal rule thỏa threshold/duration. JSONL mỗi dòng lưu tối thiểu ID, loại, mức độ, confidence, giải thích, thời điểm, risk score và risk level sau cập nhật. File nằm tại `storage/sessions/<session-id>/events.jsonl`.

Event hiện là gợi ý review, không phải phán quyết. Không ghi landmark thô hoặc
ảnh trực tiếp vào JSONL. Evidence JPEG event tùy cấu hình đã có; MVP chưa có DB,
audit đa người dùng hoặc truy cập từ xa.

## Risk

RiskEngine tổng hợp event và decay theo thời gian, giữ điểm trong 0–100 và mức `NORMAL/LOW/MEDIUM/HIGH/CRITICAL`. Giao diện cần gọi đây là mức ưu tiên rà soát, không phải xác suất hay kết luận gian lận.

## Cấu hình và model

- `configs/default.yaml`: camera, detector, behavior thresholds, risk và runtime.
- `configs/exam_policy.yaml`: vật dụng được phép trong phiên.
- `models/manifest.yaml`: model task, version, license, URL, SHA-256 và class map.
- Session giữ snapshot config/policy trong RAM; không sửa giữa phiên.

## Mở rộng dữ liệu được thiết kế, chưa triển khai

### CandidateObservation và EventEpisode

`CandidateObservation` là metadata từ đúng `ObservationBundle`: `frame_id`,
`captured_at_monotonic_ms`, `class`, `confidence`, bbox chuẩn hóa, `object_track_id`,
`track_confirmed`, model name/version/hash và analyzer health. Không gán nhãn
“gian lận”. `EventEpisode` có ID ổn định theo session + object track + rule,
`opened_at`, `last_confirmed_at`, `closed_at`, nguồn frame và trạng thái
`CANDIDATE -> OPEN -> CLOSING -> CLOSED`. Person track chỉ là liên kết có kiểm
chứng, không phải định danh của chính vật thể. Một episode không được mở hai
event do một khoảng YOLO mất detection ngắn.

### EvidenceReview và nhãn phản hồi

Evidence giữ JPEG frame đã phân tích, `frame_id`, transform/bbox gốc,
`candidate_observation_id`, `event_id`, cấu hình/model hash, ngày hết hạn. Bản
review do giám thị tạo gồm `event_id`, `review_status` (`chưa_rà_soát`,
`tín_hiệu_hợp_lệ`, `báo_nhầm`, `không_kết_luận`), thời điểm và ghi chú tùy chọn.
Nhãn review không tự biến thành verdict đối với thí sinh. Chỉ xuất dữ liệu
huấn luyện từ bằng chứng đã được phép sử dụng; tách tập train/validation/holdout
theo phiên/người để tránh rò dữ liệu.

### FaceFeatureObservation và CalibrationState

Feature cùng frame, cùng anonymous track: `mar`, `iris_ratio_left/right`,
`yaw/pitch`, quality, occlusion flags, `calibration_id` và trạng thái analyzer.
Chỉ lưu chỉ số cần thiết, không ghi toàn bộ 478 landmark vào log mặc định.
`CalibrationState`: `UNCALIBRATED -> COLLECTING -> PROPOSED -> CALIBRATED`
hoặc `FAILED`; lưu source (`auto`/`manual`), độ ổn định, thời điểm và tuổi tối
đa trong RAM. Nếu không có baseline đáng tin, gaze/head mới không được khẳng
định “lệch”. Nút manual vẫn hiện để hiệu chỉnh lại.

### QualityObservation, CorrelationFinding và RiskSnapshot

Quality giữ brightness/entropy/blur và trạng thái camera/analyzer; camera bị
che là rule nhiều mẫu, không phải một feature cứng. CorrelationFinding hiện có
được mở rộng theo rule code, hai event nguồn, cùng track, cửa sổ thời gian,
freshness và cap. RiskSnapshot giữ riêng `current_score` (có decay và floor chỉ
khi event còn hoạt động) và `history` (`event_count`, `cumulative_score`,
`peak_score`); không có session floor vĩnh viễn.
