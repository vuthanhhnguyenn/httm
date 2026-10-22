# Research: Hoàn thiện Phase 1

**Feature**: [spec.md](./spec.md) | **Date**: 2026-10-16

Mỗi mục ghi: quyết định, lý do, phương án đã cân nhắc.

## R1. Bộ dữ liệu đánh giá

- **Decision**: Dùng OEP dataset (Atoum et al., "Automated Online Exam Proctoring", IEEE TMM 2017),
  bản chính thức `https://www.cse.msu.edu/computervision/OEP_database.tar.gz` (11,8 GB), lưu tại
  `E:\datasets\oep`, ngoài Git. Chỉ dùng video webcam (không dùng wearcam). Ánh xạ nhãn gốc sang
  `EventType` trong `oep_mapping.yaml` đặt cạnh dữ liệu và một bản mẫu trong repo
  (`tests/video_samples/oep_mapping.example.yaml`). Bổ sung clip tự quay cho thi giấy, ca âm tính khó
  (bộ bài, ví, điều khiển), che camera, hai người.
- **Rationale**: OEP là bộ video webcam công khai duy nhất tìm được có người thật làm bài và nhãn
  khoảng thời gian gian lận (đọc tài liệu, nói chuyện, dùng điện thoại, nhờ người khác…). Trang dữ liệu
  chỉ yêu cầu trích dẫn; không có giấy phép phân phối lại nên không đưa vào repo.
- **Alternatives considered**:
  - Mendeley "Cheating Scenario Dataset" (CC BY 4.0): chỉ là ảnh tĩnh của 3 người, không đánh giá được
    luật thời gian.
  - Mendeley "Students suspicious behaviors" (39xs8th543): bảng đặc trưng đã trích xuất, không có video.
  - Stock video (Pexels, Mixkit): không có nhãn, góc quay không giống webcam bài thi; chỉ dùng được làm
    ca âm tính nếu cần.
- **Đã xác minh (2026-10-16)**: 24 video webcam 640×480, 25 FPS, 14–18 phút; `gt.txt` dạng `MMSS MMSS mã`.
  Mã dùng: 1 (192 khoảng), 2 (285), 3 (52), 5 (15), 6 (21); không có mã 4. Xem khung ảnh giữa khoảng nhãn:
  1 = xem sách/ghi chú (cúi nhìn bàn, có khi có người đưa sách), 2 = nói chuyện với người trong phòng,
  3 = tra Internet trên máy (webcam không quan sát được → `ignore`), 5 = gọi điện thoại, 6 = cúi sâu
  xuống dưới bàn. Người tham gia đeo camera trên kính; YOLO có thể nhận nhầm thành điện thoại.

## R2. Phát lại tất định

- **Decision**: Thêm nguồn khung hình từ tệp video; dấu thời gian = `frame_index * 1000 / fps` của
  video; UTC = mốc cố định + dấu thời gian. Phân tích theo nhịp cố định (mặc định 8 lượt/giây, cấu hình
  được) bằng cách chọn frame theo bước, không dùng hàng đợi bỏ frame theo tốc độ máy. Chạy cùng
  `ObservationPipeline` và cùng luật như webcam.
- **Rationale**: Lịch chạy pose/tay của pipeline đã dựa vào dấu thời gian frame, nên khi dấu thời gian
  lấy từ video thì cả chuỗi lặp lại được. Nhịp 8 lượt/giây gần tốc độ AI đo trên máy tham chiếu
  (5,5–8 FPS), nên số đo phản ánh điều kiện thật.
- **Alternatives considered**: Phát lại JSONL quan sát đã ghi (nhanh, nhưng không kiểm tra được thay
  đổi phía AI như hướng mắt); phát theo thời gian thực (không tất định).

## R3. Chấm điểm mức sự kiện

- **Decision**: Mỗi khoảng nhãn (loại, bắt đầu, kết thúc) khớp với sự kiện dự đoán cùng loại (hoặc loại
  được ánh xạ tương đương) nếu sự kiện bắt đầu trong `[bắt đầu − dung sai, kết thúc + dung sai]`. Một
  khoảng nhãn được tính đúng một lần; sự kiện thừa trong cùng khoảng tính là "trùng lặp" (SC-004), sự kiện
  không khớp khoảng nào tính là báo nhầm. Báo nhầm mỗi phút = báo nhầm / thời lượng không có nhãn của
  loại đó. Độ trễ = `max(0, bắt đầu sự kiện − bắt đầu nhãn)`. Hỗ trợ vùng `ignore`.
- **Rationale**: Chấm theo frame không phản ánh trải nghiệm giám thị; chấm theo sự kiện khớp với các chỉ
  số hiến pháp yêu cầu (báo nhầm/phút, bỏ sót, độ trễ).
- **Alternatives considered**: IoU theo khoảng thời gian (khắt khe với hành vi có ranh giới mờ); chấm
  theo frame (thổi phồng độ chính xác).

## R4. Hướng mắt

- **Decision**: Dùng 478 điểm mốc của mô hình mặt hiện có (gồm mống mắt 468–477). Mỗi mắt tính tỉ lệ
  ngang `(tâm mống mắt − khóe trong) / độ rộng mắt` và tỉ lệ dọc theo mí trên/dưới; lấy trung bình hai
  mắt có trọng số theo độ rộng mắt nhìn thấy. Kết hợp với yaw/pitch đầu bằng ánh xạ tuyến tính học trong
  bước hiệu chỉnh (thi máy: tâm + 4 góc màn hình; thi giấy: tâm + nhìn xuống bài). Chất lượng: độ rộng
  mắt tối thiểu (pixel), độ mở mắt (loại chớp mắt), |yaw| tối đa. Luật: "Nhìn ra ngoài màn hình" (thi
  máy) và "Liếc ngang kéo dài" (cả hai).
- **Rationale**: Không cần mô hình hay dependency mới; điểm mốc mống mắt đủ để phân biệt trong/ngoài màn
  hình ở mức thô; hiệu chỉnh theo người bù khác biệt hình học mắt và vị trí camera.
- **Alternatives considered**: Mô hình gaze chuyên dụng (L2CS-Net, ETH-XGaze): thêm model PyTorch nặng,
  giấy phép phải rà soát, tốn FPS trên Quadro M1200; để dành nếu số đo cho thấy cách hình học không đủ.

## R5. Nói chuyện

- **Decision**: Tính độ mở miệng `MAR = khoảng cách môi trong (13–14) / độ rộng miệng (61–291)` mỗi lượt
  phân tích mặt. Trong cửa sổ trượt (mặc định 2 giây) đếm số dao động quanh trung bình có biên độ ≥ ngưỡng
  và độ lệch chuẩn ≥ ngưỡng. Nói = đủ số dao động trong cửa sổ, kéo dài đủ thời lượng. Ngáp = MAR cao giữ
  ổn định (độ lệch chuẩn thấp) → loại trừ. Tín hiệu hỗ trợ, có trần điểm.
- **Rationale**: Không cần bật thêm đầu ra biểu cảm; chỉ dùng điểm mốc sẵn có. Ở 8 lượt/giây chỉ thấy
  được dao động ≤ 4 Hz nên luật đếm dao động thô chứ không nhận dạng âm tiết; ghi rõ giới hạn và đo trên
  OEP (có nhãn nói chuyện).
- **Alternatives considered**: Hệ số biểu cảm `jawOpen` của mô hình mặt (thêm chi phí suy luận, tương đương
  MAR cho mục đích này); âm thanh micro (ngoài phạm vi quyền riêng tư phase 1).

## R6. Tay

- **Decision**: Ghép tay giữa hai lượt bằng handedness rồi vị trí gần nhất (tối đa 2 tay, tham lam là đủ).
  Tốc độ = độ dời cổ tay / độ rộng mặt / Δt. "Gần mặt/tai" = đầu ngón trỏ hoặc cổ tay nằm trong bán kính
  (theo độ rộng mặt) quanh vùng tai (điểm 234, 454) hoặc miệng; loại trừ chống cằm khi tay nằm dưới cằm
  (điểm 152) và chạm vùng cằm.
- **Rationale**: Sửa lỗi ghép theo thứ tự index; chuẩn hóa để không phụ thuộc khoảng cách tới camera và tốc
  độ AI.
- **Alternatives considered**: Thuật toán Hungarian (thừa cho ≤ 2 tay); dùng bbox mặt phóng to (nguồn báo nhầm
  hiện tại).

## R7. Kiểu bài thi

- **Decision**: Cấu hình có `behavior` (mặc định chung) và `exam_modes.paper` / `exam_modes.computer` là
  phần ghi đè được hợp nhất sâu vào `behavior` khi bắt đầu phiên. Kiểu bài thi lưu trong ảnh chụp chính
  sách phiên. Giao diện có lựa chọn trước khi bắt đầu, nhớ lựa chọn lần trước bằng QSettings.
- **Rationale**: Tập trung khác biệt vào cấu hình (nguyên tắc III), luật không cần biết kiểu bài thi trừ
  khi đọc tham số đã được hợp nhất.
- **Alternatives considered**: Hai tệp cấu hình riêng (lặp nhiều); điều kiện `if mode == "paper"` trong luật
  (vi phạm FR-006).

## R8. Thứ bậc tín hiệu

- **Decision**: Thêm `SignalArbiter` trong lớp nghiệp vụ, chạy sau các luật và trước `EventAggregator`.
  Tín hiệu gốc: `PERSON_MISSING`, `CAMERA_BLOCKED`. Khi gốc đang hoạt động, các tín hiệu hệ quả
  (`FACE_NOT_VISIBLE`, `LEAVING_SEAT`, hướng đầu, hướng mắt, nói chuyện, tay) không mở sự kiện riêng; tên
  của chúng được ghi vào `metrics.supporting_signals` của sự kiện gốc. `LEAVING_SEAT` là gốc của
  `FACE_NOT_VISIBLE`. Bảng thứ bậc nằm trong cấu hình.
- **Rationale**: Một sự việc → một sự kiện → một lần cộng điểm (FR-011, SC-004).
- **Alternatives considered**: Xử lý trùng trong `RiskEngine` (vẫn hiện nhiều cảnh báo trên giao diện).

## R9. Điện thoại và vật thể

- **Decision**: Episode theo lớp vật thể (không theo track): giữ mở qua khoảng mất phát hiện ≤ khoảng hở
  cấu hình và qua đổi track. Thêm tùy chọn lọc hình học (tỉ lệ cạnh, kích thước so với người, gần tay) bật
  theo cấu hình, chỉ bật mặc định khi báo cáo đánh giá cho thấy giảm báo nhầm mà không giảm độ phủ. Thử
  `image_size` 640 cho điện thoại và ghi tác động FPS. Lớp tài liệu chỉ còn `book`.
- **Rationale**: Hiện evaluator đóng điều kiện khi đổi track id, gây cảnh báo thứ hai (FR-010); COCO không
  có lớp `paper`/`document`.
- **Alternatives considered**: Huấn luyện lại YOLO với lớp bộ bài (cần dữ liệu có nhãn, ngoài phạm vi phase 1).

## R10. Cấu hình có kiểu

- **Decision**: Mở rộng `proctoring_core.config.schema` (Pydantic, `extra="forbid"`) thành nguồn duy nhất:
  mọi tham số luật, ngưỡng hiện viết cứng, `exam_modes`, `gaze`, `talking`, `hands`, `hierarchy`,
  `evaluation`. Runtime nạp qua `load_config_files` rồi `resolve_for_mode(exam_mode)`; evaluator nhận đúng
  model con của mình. Giữ nguyên giá trị mặc định hiện tại trong bước tái cấu trúc.
- **Rationale**: Bỏ chuỗi `.get(..., default)` rải rác và các mặc định mâu thuẫn; từ chối khóa sai (FR-008).

## R11. Mốc hồi quy cho tái cấu trúc

- **Decision**: Trước khi tái cấu trúc, ghi "golden": chạy phát lại trên một tập con cố định (3 người OEP +
  các ca tổng hợp trong `tests/fixtures`) và lưu danh sách sự kiện (loại, hành động, mốc thời gian video,
  mức độ) ở dạng chuẩn hóa. Mỗi bước tái cấu trúc so khớp đúng 100%. Ca tổng hợp (không cần mô hình) được
  đưa vào bộ kiểm thử CI; ca OEP chạy cục bộ.
- **Rationale**: FR-031; CI không có mô hình và dữ liệu OEP nên cần mốc nhẹ chạy được trên CI.

## R12. Rời chỗ và vùng ngồi

- **Decision**: Vùng ngồi = trung vị bbox người trong lúc hiệu chỉnh (hoặc 3 giây đầu khi đang chạy). Rời
  chỗ khi tâm người ra ngoài vùng cộng biên, hoặc diện tích bbox < tỉ lệ cấu hình của vùng, hoặc vai (pose)
  ra khỏi khung, kéo dài đủ thời lượng.
- **Rationale**: Thay ngưỡng `bbox.y > 0.9` không phụ thuộc vị trí camera.

## R13. Che camera và lỗi thiết bị

- **Decision**: Bộ trích đặc trưng ảnh trả về độ sáng trung bình, độ lệch chuẩn (tương phản), mật độ cạnh
  (Laplacian), độ mờ. Che camera khi (rất tối hoặc tương phản rất thấp hoặc gần như không có cạnh) kéo dài.
  Frame giống hệt nhau liên tục hoặc đọc frame thất bại → trạng thái thiết bị "CAMERA_FAULT" trên giao diện,
  không phải sự kiện hành vi.

## R14. Hiệu chỉnh tự đề xuất

- **Decision**: Khi mặt chất lượng tốt và góc thô ổn định (biên độ < 5°) trong 2 giây, đầu trong ±25° quanh
  trục camera, ứng dụng hiển thị "Đề xuất hiệu chỉnh — nhấn để chấp nhận". Không tự áp dụng im lặng. Hiệu
  chỉnh thủ công giữ nguyên. Trạng thái: `uncalibrated`, `suggested`, `collecting`, `calibrated`, `failed`.
- **Rationale**: FR-014; giữ con người quyết định mốc trung tính.

## R15. Bố cục gói

- **Decision**: Thêm gói `packages/proctoring_runtime` (không phụ thuộc Qt) chứa điều phối phiên, nạp cấu
  hình, lưu trữ JSONL/bằng chứng (implement các port), và bộ phát lại/đánh giá. `apps/desktop` chỉ còn giao
  diện Qt và lớp bọc `QThread` quanh runtime. Lớp nghiệp vụ `proctoring_core` nhận thêm quan sát có kiểu,
  `BehaviorEngine`, `SignalArbiter`, luật mới và chấm điểm đánh giá.
- **Rationale**: Bộ đánh giá dòng lệnh và giao diện dùng chung một đường điều phối (FR-001), và lớp ứng dụng
  không chứa luật (FR-029).
- **Alternatives considered**: Để điều phối trong `apps/desktop` (kéo PySide6 vào công cụ dòng lệnh); đặt vào
  `proctoring_ai` (trộn adapter mô hình với lưu trữ).
