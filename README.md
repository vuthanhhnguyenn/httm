# Smart Exam Proctoring

Ứng dụng desktop chạy trực tiếp trên Windows để hỗ trợ giám thị theo dõi một thí sinh qua webcam. Camera, preview, AI và giao diện cùng chạy trên máy; ứng dụng không cần khởi động web server.

AI có thể đánh dấu người, khuôn mặt, điện thoại/sách, tư thế và bàn tay. Các tín hiệu như quay đầu, cúi đầu, thiếu khuôn mặt, hoạt động tay hoặc điện thoại cần được giám thị xem lại. Ứng dụng không tự kết luận gian lận.

## Chạy lần đầu

Cần Python 3.12 và `uv`. Tại thư mục gốc repository, cài dependency:

```powershell
uv sync --group desktop --no-default-groups --extra opencv-python
```

Tải các model từ nguồn khai báo trong manifest và kiểm tra SHA-256:

```powershell
uv run --group desktop --no-default-groups --extra opencv-python python scripts/download_model.py
```

Mở ứng dụng:

```powershell
uv run --group desktop --no-default-groups --extra opencv-python python apps/desktop/main.py
```

Trong cửa sổ, chọn thiết bị camera rồi nhấn **Bắt đầu giám sát**. Nhấn **Dừng phiên** để giải phóng webcam. Model được lưu trong `models/pretrained/` và không đưa vào Git.

Khi AI sẵn sàng, nhìn thẳng màn hình và nhấn **Hiệu chỉnh hướng đầu**, giữ đầu yên khoảng 2 giây.
Khung điện thoại màu vàng “đang xác nhận” là ứng viên; core cần nhiều lần phát hiện và đủ
thời gian trước khi tạo sự kiện. Khung nét đứt “tạm giữ” chỉ giúp preview bớt nhấp nháy,
không được dùng làm bằng chứng mới.

Xem [kế hoạch độ ổn định và hiệu năng AI](docs/ai-stability-performance-plan.md) để thử bản sửa,
đo từng analyzer và chuẩn bị môi trường CUDA phù hợp Quadro M1200.

## Chạy bằng GPU NVIDIA

Không cần cấu hình thêm: khi khởi động, app tự cho YOLO dùng GPU NVIDIA đầu tiên (`cuda:0`)
nếu PyTorch trong môi trường đang chạy dùng được CUDA. Nếu không dùng được (PyTorch bản CPU,
không thấy GPU, GPU không nằm trong các kiến trúc PyTorch hỗ trợ, hoặc khởi động YOLO trên GPU
bị lỗi), app chuyển YOLO sang CPU và ghi lý do ra terminal, ví dụ
`WARNING proctoring_ai.acceleration: YOLO chạy CPU vì không dùng được CUDA: …`.
Trạng thái YOLO trong app ghi **GPU** hoặc **CPU** theo thiết bị thực tế.

MediaPipe (khuôn mặt, tư thế, bàn tay) cũng thử GPU trước, nhưng MediaPipe Tasks cho Python
hiện chỉ có GPU delegate trên Linux. Trên Windows ba bộ này vẫn chạy CPU, và terminal ghi
`MediaPipe … chạy CPU: MediaPipe Tasks Python chưa hỗ trợ GPU trên Windows`.

`.venv` cài từ lockfile có PyTorch bản CPU trên Windows, nên muốn YOLO chạy GPU thì dùng môi
trường `.venv-gpu` (PyTorch CUDA 11.8, hỗ trợ Quadro M1200). Đóng phiên/app cũ, rồi từ thư mục gốc chạy:

```powershell
.\scripts\run_gpu.ps1
```

Script này đặt `PROCTORING_AI_DEVICE=0`, tức là bắt buộc GPU: nếu CUDA không dùng được, app báo lỗi
thay vì âm thầm chạy CPU. Trên Task Manager, card NVIDIA hiện là **GPU 1** vì **GPU 0** là Intel;
hai cách đánh số này không mâu thuẫn. Nếu PowerShell không cho chạy file `.ps1`, có thể chạy
trực tiếp (tự chọn GPU, lùi về CPU nếu cần):

```powershell
.\.venv-gpu\Scripts\python.exe apps/desktop/main.py
```

`PROCTORING_AI_DEVICE` nhận `auto` (mặc định), `cpu`, hoặc chỉ số GPU như `0`.
Không chạy `uv sync` hoặc `uv run` cho `.venv-gpu` sau bước cài CUDA: lockfile
chung có thể cài lại PyTorch CPU. Cách chạy CPU cũ vẫn dùng `.\.venv\Scripts\python.exe`.

## Trên màn hình

- Preview webcam được cập nhật độc lập với AI. `FPS hiển thị` đo tốc độ ảnh thực sự được vẽ trên cửa sổ.
- `AI FPS` đo tốc độ các frame được xử lý xong. Khung nhận diện có nhãn và thời điểm phân tích; khung quá cũ sẽ không vẽ chồng lên hình mới.
- YOLO đánh dấu người và vật thể; MediaPipe đánh dấu mặt, khung xương tư thế và bàn tay. Hướng đầu là ước lượng từ landmark, không phải nhận dạng danh tính.
- Event chỉ xuất hiện sau khi điều kiện kéo dài đủ thời gian theo cấu hình. Sự kiện và mức risk là gợi ý để rà soát.
- Sự kiện được lưu trong `storage/sessions/<session-id>/events.jsonl`; tổng kết phiên ở `risk-summary.json`. Điểm hiện tại giảm dần nhưng giữ mức tối thiểu khi sự kiện đã xác nhận còn được cập nhật. Nếu AI không còn thấy sự kiện quá 2 giây, mức tối thiểu hết hiệu lực. Số sự kiện, điểm tích lũy và đỉnh điểm giữ riêng cho lịch sử rà soát.
- Khi có sự kiện mới, app lưu đúng khung ảnh đã phân tích thành JPEG trong `storage/sessions/<session-id>/evidence/`. Nhấp đúp sự kiện để xem ảnh khi đã lưu xong. App không ghi video phiên. Hàng đợi tối đa 4 ảnh; khi đầy, sự kiện vẫn ghi lại và ghi rõ ảnh chưa được lưu.
- Tên sự kiện, mức cảnh báo và giải thích trên giao diện dùng tiếng Việt. JSONL giữ mã kỹ thuật và bổ sung `label_vi`, `severity_vi`, `message_vi`; lỗi ứng dụng lưu chi tiết tại `storage/runtime-errors.log`. Cảnh báo nội bộ MediaPipe/CUDA ở terminal có thể vẫn dùng tiếng Anh.
- YOLO và khuôn mặt chạy mỗi lượt AI; tư thế tối đa 4 lượt/giây, bàn tay tối đa 5 lượt/giây. Mỗi bộ có FPS đo thực tế riêng. Khung tay/tư thế nét đứt là kết quả tạm giữ tối đa 350 ms, không dùng lại để sinh sự kiện. Chỉnh `detection.pose_interval_ms` và `hands_interval_ms`; đặt `0` để so sánh với xử lý mọi lượt.

## Hiệu năng và giới hạn

Preview và AI cùng xử lý trực tiếp trên máy, không phải chờ truyền frame qua mạng. Tốc độ suy luận vẫn phụ thuộc CPU/GPU, kích thước vật thể, độ sáng, khoảng cách camera và model. Máy Quadro M1200 trước đó xử lý YOLO khoảng 5,5 FPS; vì vậy không thể cam kết AI đạt 15 FPS chỉ bằng thay đổi giao diện. Ứng dụng hiển thị preview FPS và AI FPS riêng để phân biệt rõ hai tốc độ.

Model COCO nhận diện lớp `cell phone` và `book`; điện thoại nhỏ, bị che hoặc ở xa có thể không được phát hiện. Giấy rời/tài liệu không được phát hiện (chỉ lớp `book`). Hướng mắt là ước lượng hình học từ mống mắt sau hiệu chỉnh, dễ sai khi ánh sáng kém, đeo kính hoặc mặt nghiêng; nói chuyện chỉ nhận biết nhịp mở/đóng miệng thô ở tốc độ AI hiện tại (≈ 8 lượt/giây), không nhận dạng lời nói.

## Cấu hình

Thiết kế và cách kiểm thử cảnh báo điện thoại/quay đầu: [hướng dẫn hiệu chỉnh sự kiện](docs/event-detection-tuning.md).
Trong cấu hình mặc định, phải **Hiệu chỉnh hướng đầu** trước khi nhận cảnh báo quay/cúi đầu;
điện thoại vẫn hoạt động và có đường xác nhận nhanh cho nhiều khung có confidence cao.

- `configs/default.yaml`: webcam, resolution, target FPS, ngưỡng detection và kích thước ảnh AI.
- Trong `risk`, `recovery_half_life_seconds: 1.0` giúp điểm hiện tại giảm một nửa mỗi giây khi không còn sự kiện hoạt động (ví dụ 40 → 20 → 10 → 5); dưới 0,5 điểm thì về 0. Tính theo thời gian giữa các khung AI, không trừ từng điểm theo nhịp một giây. Giá trị nhỏ hơn giúp giảm nhanh hơn; đặt `null` để trở lại giảm tuyến tính bằng `decay_per_second` (mặc định 1,5). Khi sự kiện vẫn hoạt động, điểm chỉ giảm tuyến tính và không thấp hơn điểm sàn của sự kiện. `active_event_ttl_seconds` mặc định 2 giây: mất cập nhật quá hạn thì bỏ sàn; sự kiện đóng sẽ bỏ sàn ngay. Lịch sử phiên, điểm tích lũy, điểm đỉnh và ảnh bằng chứng vẫn giữ nguyên.
- Correlation ghép các sự kiện đã xác nhận trong cửa sổ 5 giây, yêu cầu cặp tồn tại thêm 0,25 giây, và có tổng trần 20 điểm/phiên. Các cặp gồm điện thoại + nhiều người, quay đầu + tay, cúi đầu + tay; hai cặp có tay chỉ ghép khi nhận diện được cùng người. Bản ghi tổ hợp chứa `source_event_ids` để đối chiếu sự kiện và ảnh nguồn. Đây là tín hiệu để giám thị rà soát.
- Trong `evidence`, đặt `evidence_frame_enabled: false` để tắt lưu ảnh; `retention_days` mặc định 30. Nếu policy cũng có các trường này, điều kiện lưu cần cả hai bên cho phép và dùng thời hạn ngắn hơn. Ảnh hết hạn được xóa khi app bắt đầu phiên mới; chỉ ảnh JPEG đã được lập chỉ mục mới bị xóa.
- `configs/exam_policy.yaml`: cho phép/cấm điện thoại, sách và giấy nháp.
- `models/manifest.yaml`: nguồn tải, license và SHA-256 của từng model.

## Kiểu bài thi, hiệu chỉnh và rà soát

- **Kiểu bài thi** chọn trước khi bắt đầu và cố định trong phiên. *Thi giấy*: cúi nhìn bài là bình thường,
  không cảnh báo cúi đầu hay "nhìn ra ngoài màn hình". *Thi máy*: chỉ cúi đầu từ 2 giây trở lên mới cần
  rà soát. Khác biệt nằm trong `exam_modes` của `configs/default.yaml`.
- **Hiệu chỉnh hướng đầu**: khi đầu thí sinh đứng yên khoảng 2 giây, app *đề xuất* hiệu chỉnh; giám thị
  bấm "Chấp nhận hiệu chỉnh đề xuất" nếu thí sinh đang nhìn thẳng. App không tự áp dụng mốc, và từ chối
  mốc lệch quá 35°.
- **Hiệu chỉnh hướng mắt** (sau hướng đầu): thí sinh nhìn lần lượt các chấm trên màn hình (thi máy: tâm và
  4 góc) hoặc tâm màn hình rồi bài thi (thi giấy); tổng thời gian dưới 15 giây. Chưa hiệu chỉnh thì không có
  cảnh báo hướng mắt.
- **Tín hiệu thử nghiệm**: hướng mắt (`GAZE_OFF_SCREEN`, `GAZE_SIDEWAYS`) và nói chuyện (`POSSIBLE_TALKING`)
  có nhãn "Thử nghiệm", mức thấp, trần điểm riêng, và cả nhóm tín hiệu hỗ trợ (mắt, miệng, tay) không thể
  một mình đưa điểm lên mức "Cao" (`risk.supporting_group_cap`).
- **Một sự việc, một sự kiện**: thí sinh rời đi chỉ tạo "Vắng người"; "Không thấy mặt", quay đầu… được ghi
  là tín hiệu kèm theo, không cộng điểm riêng. Điện thoại cầm liên tục, dù bị che ngắn hoặc đổi track, chỉ
  mở một sự kiện.
- **Rà soát**: nhấp đúp một sự kiện để xem đúng khung ảnh đã phân tích kèm khung bao, độ tin cậy, mô hình,
  kiểu bài thi và tín hiệu kèm theo; bấm *Hợp lệ*, *Báo nhầm* hoặc *Không kết luận*. Nhãn được ghi thêm vào
  `events.jsonl` (không sửa dòng cũ, không đổi điểm đã ghi).
- Camera bị treo (khung hình lặp lại) được báo trên giao diện như lỗi thiết bị; app tạm không đánh giá hành
  vi thay vì coi ảnh cũ là bằng chứng.

## Đánh giá bằng video có nhãn

```powershell
uv run --no-sync python scripts/evaluate.py --set E:\datasets\oep\oep_set.yaml --save-events
```

Công cụ phát lại video qua đúng chuỗi phân tích và luật của app (thời gian lấy theo video nên kết quả lặp
lại được), rồi báo cáo theo từng loại sự kiện: đúng, bỏ sót, báo nhầm/phút, độ trễ, tỉ lệ một sự kiện cho
mỗi hành vi. Xem `specs/002-phase1-hardening/quickstart.md`, `docs/evaluation-results.md`,
`docs/evaluation-methodology.md` (số liệu chứng minh được gì) và `docs/privacy-policy.md` (dữ liệu phiên thi).

## Cấu trúc

```text
apps/desktop/                 Giao diện PySide6 (cửa sổ, widget, hộp thoại hiệu chỉnh/rà soát)
packages/proctoring_ai/       Camera/video, YOLO, MediaPipe, đặc trưng mắt/miệng/tay, observation pipeline
packages/proctoring_core/     Domain: quan sát có kiểu, luật hành vi, thứ bậc tín hiệu, hiệu chỉnh, risk, đánh giá
packages/proctoring_runtime/  Điều phối phiên không phụ thuộc Qt, lưu JSONL/bằng chứng, phát lại video
configs/                      Cấu hình ứng dụng, kiểu bài thi và chính sách phiên
models/manifest.yaml          Nguồn và checksum của model
storage/                      Metadata phiên trên máy cục bộ
```

## Nguyên tắc vận hành

Hệ thống chỉ tạo tín hiệu nghi vấn và giải thích dựa trên dữ liệu quan sát. Giám thị là người đưa ra quyết định cuối cùng. Không dùng kết quả để nhận dạng danh tính.
