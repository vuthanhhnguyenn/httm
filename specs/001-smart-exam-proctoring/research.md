# Nghiên cứu và quyết định

## Chạy native thay vì stream ảnh qua trình duyệt

Camera, inference và giao diện nằm trong cùng tiến trình trên máy người dùng. Cách này bỏ chi phí encode và truyền frame qua mạng, đồng thời cho phép vẽ preview trực tiếp. Đổi lại, cài đặt Python/native dependencies phức tạp hơn và hiệu năng vẫn phụ thuộc phần cứng.

## Preview và inference tách nhịp

Camera capture tiếp tục lấy frame khi YOLO chậm. Queue chỉ giữ frame mới nhất để inference không xử lý dữ liệu quá cũ; preview FPS và AI FPS được đo riêng. Không nội suy hay báo target FPS như kết quả thật. Máy Quadro M1200 trước đây đạt khoảng 5.5 AI FPS với cấu hình YOLO đang dùng, do đó mức 15 FPS cần benchmark/tối ưu riêng.

## Nhận diện cục bộ

- YOLO local phục vụ người và vật thể COCO như điện thoại/sách. Giấy tờ nhỏ, xa, che khuất hoặc không thuộc class map có thể bị bỏ sót.
- MediaPipe Tasks local cung cấp landmark mặt, hướng đầu ước lượng, pose và tay. Hướng đầu không thay thế gaze calibration.
- Model được pin bằng version/hash; analyzer health tách biệt để báo rõ model chưa có, license không được chấp nhận, lỗi load hoặc chưa có detection.

## Quyền riêng tư và quyết định của con người

Frame preview/inference ở RAM. JSONL lưu metadata event cục bộ; ảnh bằng chứng
event có thể lưu theo cấu hình và retention. Không lưu video phiên mặc định.
Risk chỉ sắp xếp mức cần xem; giám thị xác minh từng cảnh và giữ thẩm quyền cuối cùng.

## Lưu trữ MVP

MVP ghi event metadata append-only vào file JSONL theo session và đã có ảnh bằng
chứng event tùy cấu hình, ghi nền với retention. Chưa có database, login, đồng bộ,
audit trail đa người dùng hoặc giao diện review lịch sử đầy đủ.

## Quyết định cho mở rộng core

### P0/P1: ưu tiên đo và giảm báo nhầm điện thoại

**Quyết định:** thêm provenance bbox/confidence/model/object track vào bằng chứng,
gắn nhãn báo nhầm của giám thị, chạy bộ video holdout gồm bộ bài và điện thoại
trước khi đổi ngưỡng hoặc model. Cần kiểm tra episode sự kiện độc lập với person
track. **Lý do:** ca bộ bài cho thấy nhãn sai ổn định có thể đi qua temporal rule;
đợi lâu hơn chỉ giảm tốc độ báo điện thoại thật. **Phương án xét:** tăng confidence
hoặc duration ngay; chưa chọn vì chưa biết precision–recall trên dữ liệu mục tiêu.
Ultralytics có công cụ đánh giá precision/recall và FP/FN trên tập riêng
([tài liệu validation](https://docs.ultralytics.com/modes/val/)).

### Đặc trưng khuôn mặt

**Quyết định:** khai thác landmark hiện có để tạo đặc trưng mắt/môi tối thiểu,
nhưng để gaze/MAR ở trạng thái thử nghiệm, tắt mặc định. `face_landmarks` thuộc
kết quả khi nhận được mặt; `face_blendshapes` là đầu ra tùy chọn và mặc định
tắt, nên tài liệu đầu vào nói đã sẵn 52 blendshapes là không đúng với cấu hình
app hiện tại ([MediaPipe options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions),
[result](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult)).
**Lý do:** mắt, môi và vùng mặt dễ bị ảnh hưởng bởi ánh sáng, góc đầu, kính,
độ phân giải. Nghiên cứu gaze trên webcam cho thấy cần xét sai khác cá nhân và
calibration; ngưỡng iris ratio 0,30/0,70 trong tài liệu đầu vào không phải chuẩn
phổ quát ([nghiên cứu gaze](https://arxiv.org/abs/1711.09017),
[nghiên cứu calibration webcam](https://pmc.ncbi.nlm.nih.gov/articles/PMC10640920/)).
Chuyển động môi không chứng minh đang nói; ngáp, ăn, biểu cảm cũng tạo chuyển
động tương tự ([AVA ActiveSpeaker](https://arxiv.org/abs/1901.01342)).
**Phương án xét:** cố định MAR/gaze threshold của tài liệu; bác bỏ vì chưa kiểm
chứng trên camera và người dùng mục tiêu.

### Hiệu chỉnh và an toàn

**Quyết định:** chỉ thử auto-baseline khi mặt đủ rõ, ổn định và trạng thái hiển
thị rõ; giữ nút thủ công để sửa baseline. Không âm thầm lấy tư thế ban đầu làm
“nhìn thẳng”. Camera blocked kết hợp brightness/entropy/blur theo baseline,
không lấy các mốc 0,05/2,0 làm chân lý; rời chỗ theo track/bbox qua thời gian,
tách khỏi person missing. **Phương án xét:** bỏ nút hiệu chỉnh, chỉ dùng y>0,9
hoặc blur; bác bỏ vì dễ tạo mốc sai và báo nhầm.

### Correlation và risk

**Quyết định:** mở rộng `events/correlation.py` hiện có, không tạo engine thứ hai
trong `behavior/`. Chỉ ghép event đã xác nhận, có cùng người/cửa sổ/freshness,
cap và link nguồn. Giữ RiskEngine decay cho điểm hiện tại, lịch sử phiên không
decay nhưng không tạo floor hiển thị. **Lý do:** floor tích lũy sẽ giữ rủi ro
hiện tại cao ngay cả khi tín hiệu đã hết và có thể tích lũy báo nhầm; trái
nguyên tắc tách hiện tại/lịch sử. Các nhãn “đang nói chuyện”, “đọc dưới bàn”,
“gian lận cộng tác”, mức CRITICAL từ phone+hand đều vượt quá điều quan sát được.

### Chưa có cơ sở chấp nhận

Các tỷ lệ 35–50%, 60–70%, ngưỡng MAR/iris phổ quát và mô tả sản phẩm thương mại
trong tài liệu đầu vào chưa kèm nguồn kiểm chứng phù hợp. Kế hoạch không dùng
chúng làm tiêu chí đạt hoặc cam kết sản phẩm. Chưa có dataset có nhãn, nên chưa
chốt threshold mới hay tuyên bố “chuẩn công nghiệp”.
