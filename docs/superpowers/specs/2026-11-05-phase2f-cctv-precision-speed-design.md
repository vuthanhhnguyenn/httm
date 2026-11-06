# Phase 2f: Ít báo nhầm hơn và tầng kỹ gần thời gian thực ở chế độ CCTV — Thiết kế

- Ngày: 2026-11-05
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-11-04-phase2e-cctv-recall-design.md` (PR #41 đã gộp vào `main`)

## 1. Mục tiêu và điều đã chốt

Nguyên (2026-11-05) muốn làm cả hai việc trong một spec, theo thứ tự:

- **Phần A: giảm báo nhầm** ở tầng kỹ, không làm mất sự kiện đúng.
- **Phần B: tăng tốc**, để tầng kỹ chạy gần như thời gian thực: giám thị thấy kết quả trong khoảng
  1 phút kể từ lúc có đoạn nghi vấn (Nguyên chọn phương án A cho câu hỏi về độ trễ).

Cách làm đã chốt cho phần B là "tầng kỹ gọn, chạy nền trên cùng máy", không gộp vào tầng nhanh và không
chuyển sang máy GPU khác. **Model của tầng kỹ vẫn là `yolo11x-pose` và `yolo11x`**: Nguyên muốn giữ
model x. Model cỡ m/l chỉ là phương án dự phòng, và chỉ dùng khi Nguyên đồng ý (mục 6.4).

Hệ thống vẫn chỉ gợi ý để giám thị rà soát, không tự kết luận gian lận.

## 2. Hiện trạng (clip 1, `EPf9O7UXRoM.mp4`, 72 s)

### 2.1 Độ chính xác

Lần chạy phase 2e mở 39 sự kiện. Nguyên chấm 23 ô trong `phase2e/clip1/su-kien-tang-ky.jpg`
(2026-11-05): **9 đúng, 14 nhầm**. 9 sự kiện còn lại là `UNCLASSIFIED_SUSPECT`, không nằm trong lần chấm này.

| Mốc (s) | Loại | Nguyên chấm | Nguyên nhân |
|---|---|---|---|
| 25.5 | PASSING_OBJECT | đúng | |
| 27.2 | LOOK_AT_NEIGHBOR | đúng | |
| 45.8 | DOCUMENT_DETECTED | đúng | |
| 47.0 | LOOK_AT_NEIGHBOR | đúng | |
| 47.4 | HEADS_TOGETHER | đúng | |
| 43.2, 51.2, 59.0, 63.0 | LOOK_AT_LAP | đúng | bạn nam áo đỏ đô sát tường, tay dưới bàn |
| 17.2, 20.1, 22.1 | PASSING_OBJECT | nhầm | cùng một cặp ở hàng sau. Lý do `HANDS_MEET` là keypoint rung ở người nhỏ. Mỗi lần báo cả hai người, và báo lặp vì các đoạn nghi vấn chồng lên nhau (17.2 và 17.5 là cùng một lần) |
| 31.3 | PASSING_OBJECT | nhầm | cùng ghế với ca đúng 25.5 |
| 19.0 | PHONE_DETECTED | nhầm | vật trên bàn |
| 31.6 | PHONE_DETECTED | nhầm | máy tính bỏ túi nằm trên bàn |
| 56.8, 60.8 | PHONE_DETECTED | nhầm | cùng một vật đứng yên, báo hai lần |
| 10.0, 32.0 | DOCUMENT_DETECTED | nhầm | tờ bài thi trên bàn (COCO gọi là `book`) |
| 3.0, 28.2, 54.4 | LOOK_AT_LAP | nhầm | 28.2 là cùng ghế với các ca đúng 51.2–63.0 |
| 58.8 | TURN_AROUND | nhầm | |

Bài học: không được tắt luật `LOOK_AT_LAP` với người cúi đầu, vì có ca thật. Phải phân biệt bằng tay và
thời gian.

Nhãn sự kiện hiện có (`data/labels/EPf9O7UXRoM.yaml`) là các vùng có hành vi. Không có danh sách "sự
kiện này là nhầm", nên không đo được độ chính xác của tầng kỹ.

### 2.2 Tốc độ

- Tầng nhanh trên M1200: khoảng 10 FPS ở `--fps 5` (memory phase 2d). Mỗi khung gọi model lần lượt: cả
  khung hình, rồi 2 ô cắt (`TiledPoseDetector`), rồi dò điện thoại (`CctvPipeline.detect`). Đọc video
  cùng luồng với AI.
- Tầng kỹ (`scripts/review_suspects.py`) chạy sau, đọc lại file video: 10 khung/giây cộng 6 s khởi
  động, nên khoảng 150 khung mỗi đoạn. Mỗi khung chạy 2 model x ở 960 px, từng khung một. Trên CPU cloud
  mất p50 113 s, tối đa 422 s mỗi đoạn. Clip 1 có 30 đoạn trong 72 s, tức trung bình mỗi đoạn chỉ có
  khoảng 2,4 s nếu muốn theo kịp.

## 3. Tiêu chí đạt

| Tiêu chí | Đo ở | Mức đạt |
|---|---|---|
| Sự kiện đúng tầng kỹ trên clip 1 | cloud, bộ kiểm tra hồi quy | 9/9 vẫn mở |
| Sự kiện nhầm tầng kỹ trên clip 1 | cloud, bộ kiểm tra hồi quy | ≤ 7 (hiện 14) |
| Clip 2 (`LUy81HI7wS0`, chưa có nhãn) | cloud | số sự kiện không tăng so với trước khi sửa; báo Nguyên danh sách để chấm |
| Độ phủ tầng nhanh trên clip 1 | cloud, `eval_cctv.py --suspects` | vẫn 6/6 |
| Độ trễ tầng kỹ | M1200, `--mode realtime` | p95 ≤ 60 s từ lúc đoạn đóng tới lúc có kết quả |
| Tầng nhanh khi tầng kỹ chạy nền | M1200 | ≥ 10 FPS |

## 4. Phần A: giảm báo nhầm

### A1. Bộ kiểm tra hồi quy cho tầng kỹ

- **File nhãn mới** `data/labels/EPf9O7UXRoM-review.yaml`: mỗi mục có `type`, `t_s` (mốc giây),
  `head` (toạ độ đầu, chép từ `events.jsonl` phase 2e) và `verdict: true|false`, theo bảng 2.1. Một
  sự kiện khớp với một mục khi cùng loại, lệch dưới 1,5 s và đầu cách dưới 0,05 khung hình.
- **Lưu đầu ra model theo đoạn**: tầng kỹ ghi `review_cache/<window>.json` gồm pose và vật thể của từng
  khung đã phân tích, đã đổi về toạ độ khung hình đầy đủ. Script `scripts/replay_review.py
  <suspects.jsonl> <review_cache/> <labels>` chạy lại `review_window` từ cache mà không cần model, rồi
  in số đúng, số nhầm và số chưa chấm. Mỗi lần chỉnh luật chỉ mất vài giây.
- **Test**: một test pytest chạy replay trên cache clip 1 (đưa vào `packages/proctoring_ai/tests/
  fixtures/`, cỡ vài trăm KB). Test kiểm tra 9/9 đúng và số nhầm ≤ 7, nên sau này sửa gì làm hỏng thì
  test báo đỏ.

### A2. Gộp trùng sự kiện

- Mỗi `(ghế, loại sự kiện)` chỉ được mở một lần trong `dedupe_ms = 10000`, kể cả khi các đoạn nghi
  vấn chồng nhau. Lần sau trong khoảng đó được ghi `action: merged`, trỏ về sự kiện đầu và không có ảnh
  bằng chứng mới.
- `PASSING_OBJECT` chỉ mở cho **người với tay sang**: người có cổ tay vượt qua đường giữa hai ghế.
  Không còn mở đôi cho cả hai người, đúng như Nguyên đã chốt cho `LOOK_AT_NEIGHBOR` ở phase 2d. Nếu cả
  hai cùng với tay thì mở cho người với xa hơn.

### A3. Vật thể phải được cầm, không nằm im

Áp dụng cho `PHONE_DETECTED` và `DOCUMENT_DETECTED` ở tầng kỹ:

- Vật phải được thấy ở **≥ 2 khung** trong đoạn (cùng nhãn, IoU > 0,3 giữa các khung).
- Vật phải **gần cổ tay** (≤ 0,6 chiều rộng khung người) ở ít nhất một khung, **hoặc** phải dịch
  chuyển hơn 0,5 chiều rộng của chính nó trong đoạn. Vật nằm im cả đoạn, cách xa tay thì là đồ trên bàn.
- Vẫn giữ các điều kiện đã có (`phone_min_conf`, cao hơn đầu thì là bàn trước).
- Máy tính bỏ túi ở 31.6 nằm im cạnh tờ bài và bị loại theo luật này. Nếu vẫn lọt (vì tay bấm máy tính),
  thêm điều kiện thứ hai: `phone` mà nằm trên tờ `book`/`paper` của chính ghế đó (tâm vật nằm trong khung
  tờ giấy, mở rộng 20%) thì không tính là điện thoại.

### A4. "Tay chạm tay" tính theo cỡ người

- Khoảng cách hai cổ tay được chia cho chiều cao khung người nhỏ hơn, không tính theo toạ độ khung hình.
- Chỉ dùng cổ tay có độ tin cậy keypoint ≥ 0,5.
- Người ở hàng xa (khung người cao dưới 0,12 khung hình) phải có thêm một trong hai điều kiện: có vật
  thể giữa hai tay, hoặc cổ tay đã nằm trong khung ghế của người kia ở ≥ 2 khung.

### A5. `LOOK_AT_LAP`: giữ luật, thêm điều kiện về tay

- Vẫn cần đầu cúi như cũ, và thêm: **cả hai cổ tay ở dưới mép bàn** (`lines[main]`), hoặc không thấy
  cổ tay (bị bàn che), liên tục ≥ `lap_hold_ms`. Người đang viết có ít nhất một tay trên bàn.
- Nếu A5 làm mất một trong 4 ca đúng (43.2, 51.2, 59.0, 63.0) thì không áp dụng A5, và ghi lý do vào
  kết quả.

### A6. Thứ tự làm và cách chống "học thuộc" clip 1

Làm A1 trước, rồi A2 → A5 lần lượt. Sau mỗi bước chạy replay và ghi số đúng/nhầm vào
`docs/evaluation-results.md`. Mỗi ngưỡng mới phải có lý do bằng lời (như ở trên), không được dò số cho
khớp. Cuối phần A chạy clip 2 và gửi Nguyên ảnh các sự kiện để chấm.

## 5. Phần B: tầng kỹ gần thời gian thực

### B1. Chạy nền cùng tiến trình với tầng nhanh

```text
nguồn video ─► luồng đọc ─► hàng khung ─► TẦNG NHANH ─► sự kiện luật
                    │                          │
                    └─► bộ đệm khung (20 s) ◄──┘ đoạn nghi vấn đóng
                                 │
                                 ▼
                       hàng đợi ưu tiên ─► TẦNG KỸ (luồng nền) ─► sự kiện tầng kỹ + ảnh bằng chứng
```

- **`FrameRing`** (mới, `multi/cctv/frame_ring.py`): giữ các khung đã giải mã của 20 s gần nhất ở
  5 khung/giây (khoảng 100 khung 1280×720, cỡ 280 MB). Có thể chọn lưu JPEG chất lượng 90 để giảm còn
  khoảng 20 MB (`--ring-jpeg`). Tra theo mốc ms và trả về khung gần nhất.
- **`ReviewWorker`** (mới, `multi/cctv/review_worker.py`): một luồng riêng nhận `SuspectWindow` từ
  `SuspectTracker`. Một đoạn được gửi đi khi nó đóng, hoặc khi đã kéo dài 10 s (lúc đó gửi phần 10 s đầu
  và đoạn vẫn tiếp tục). Worker lấy khung từ `FrameRing`, không mở lại file video.
- `scripts/analyze_video_multi.py --scene cctv --review-live` bật worker. `scripts/review_suspects.py`
  vẫn giữ để chạy lại sau buổi thi, và dùng chung phần lõi mới (B2, B3).
- Luồng đọc video tách khỏi luồng AI, với hàng chờ tối đa 8 khung. Ở `--mode realtime` khung bị bỏ khi
  AI chậm, như hiện nay.

### B2. Ít khung hơn cho mỗi đoạn

- Lấy 3 khung/giây trong đoạn và 2 khung trong 6 s khởi động, **tối đa 12 khung** mỗi đoạn. Nếu đoạn
  dài hơn, chọn đều các khung trong đoạn và luôn giữ khung có lý do nghi vấn mạnh nhất. Tầng nhanh ghi
  mốc này vào `SuspectWindow.peak_ms` (trường mới).
- Mọi luật "giữ đủ lâu" (`lap_hold_ms`, …) đã tính theo mốc ms của khung chứ không theo số khung. Mình sẽ
  kiểm tra lại từng luật trong `review_rules.py` và thêm test cho trường hợp khung thưa.
- Chạy replay A1 ở 12 khung. Nếu số đúng dưới 9/9 thì tăng lên 16 khung trước khi đụng tới bất cứ thứ gì
  khác.

### B3. Gom batch

- `ReviewDetectors.pose` và `.objects` nhận **danh sách** ảnh cắt và gọi `model.predict(source=[...])`
  một lần cho mỗi model, thay vì gọi từng khung. Kích thước batch tối đa 8 (giới hạn VRAM 4 GB của
  M1200, sẽ đo thật). Nếu gặp lỗi hết bộ nhớ thì chia đôi batch rồi thử lại.
- Giữ `yolo11x-pose` và `yolo11x` ở 960 px.

### B4. Chia việc cho hai card

- Pose của tầng kỹ chạy CUDA trên M1200, cùng card với tầng nhanh.
- Dò vật thể của tầng kỹ (`yolo11x`) và dò điện thoại của tầng nhanh chạy **OpenVINO trên card Intel**
  (`device="intel:gpu"` của Ultralytics, model xuất bằng `yolo export format=openvino`). Cần thêm gói
  `openvino` (Apache-2.0) vào `uv.lock`, và thêm vào `scripts/download_model.py` bước xuất model.
- Mục mới trong `configs/review_models.yaml`: `device: cuda | intel:gpu | cpu | auto`. Nếu không có
  card hoặc không có OpenVINO thì chuyển sang CPU, ghi lý do vào `timing.json` và vào log, giống
  `acceleration.py` đang làm với CUDA.
- Cloud không có card Intel, nên phần này chỉ kiểm tra được đường chuyển sang CPU. Còn tốc độ thật thì
  Nguyên đo trên máy (mục 7).

### B5. Hàng đợi ưu tiên và không bỏ im lặng

- Gộp: nếu một đoạn mới của cùng ghế chồng lên một đoạn đang chờ thì nối hai đoạn lại (vẫn tối đa 12
  khung).
- Ưu tiên, cao trước: đoạn có `RULE_EVENT`, rồi đoạn có nhiều lý do hơn, rồi đoạn cũ hơn.
- Đoạn chờ quá `review_deadline_ms = 60000` thì không chạy nữa. Nó được ghi sự kiện
  `REVIEW_SKIPPED` (loại mới, tiếng Việt "Chưa kiểm tra kỹ"), kèm ảnh khung mạnh nhất, để giám thị tự
  xem. Không có đoạn nào bị bỏ mà không để lại dấu vết.
- `timing.json` có thêm `review_latency_ms` (p50, p95, max), `review_queue_max` và `review_skipped`.

### B6. Tầng nhanh

- `TiledPoseDetector`: khung đầy đủ và 2 ô cắt được đưa vào **một** lời gọi `predict` dạng danh sách
  (Ultralytics tự co từng ảnh về `imgsz`). Ở các khung không chạy ô cắt thì vẫn chỉ có 1 ảnh.
- `CctvPipeline.detect`: nếu máy dò điện thoại ở trên card Intel thì chạy song song với pose
  (`asyncio.gather`). Nếu nó cũng ở CUDA thì vẫn chạy lần lượt như cũ, vì chạy song song trên cùng một
  card không nhanh hơn.
- Không dùng FP16, vì M1200 (Maxwell, sm_50) không có FP16 nhanh.

## 6. Lỗi và trường hợp biên

1. **Hết VRAM khi tầng kỹ và tầng nhanh cùng chạy**: batch chia đôi rồi thử lại. Nếu batch 1 vẫn lỗi
   thì worker chuyển pose của tầng kỹ sang CPU và ghi lý do. Tầng nhanh không bao giờ bị dừng vì tầng kỹ.
2. **Khung đã rời khỏi bộ đệm** (đoạn chờ quá 20 s): worker đọc lại từ file nếu nguồn là file. Nếu
   nguồn là trực tiếp thì ghi `REVIEW_SKIPPED`.
3. **Model hỏng hoặc thiếu**: vẫn như phase 2e, bỏ qua model đó và ghi lý do vào `status`.
4. **Không đạt 60 s trên M1200** sau B1–B6: báo Nguyên số đo và đề xuất dùng model l hoặc m cho tầng kỹ.
   Chỉ đổi khi Nguyên đồng ý.

## 7. Kiểm thử

- **Unit test** (pytest, không cần model):
  - `FrameRing`: tra mốc, khung đã hết hạn, chế độ JPEG.
  - Hàng đợi: gộp, thứ tự ưu tiên, hạn 60 s, sinh `REVIEW_SKIPPED`.
  - Chọn khung: tối đa 12, luôn có `peak_ms`.
  - Gộp trùng A2 và chọn người với tay.
  - A3, A4, A5, mỗi luật có một ca đúng và một ca nhầm lấy từ cache clip 1.
  - Batch: model giả, kiểm tra một lời gọi cho nhiều ảnh và chia đôi khi hết bộ nhớ.
- **Hồi quy**: test replay trên cache clip 1 (A1).
- **Cloud**: chạy cả clip 1 và clip 2 với `--review-live --mode full`. Không có card Intel nên OpenVINO
  chạy CPU. Ghi số đúng/nhầm, độ phủ và độ trễ (độ trễ trên CPU chỉ để tham khảo).
- **Máy Nguyên (M1200 + card Intel)**: một bước cuối của kế hoạch. Chạy `scripts/run_gpu.ps1` với
  `--mode realtime --review-live` trên clip 1, rồi gửi lại `timing.json`. Đạt khi FPS tầng nhanh ≥ 10
  và `review_latency_ms.p95` ≤ 60000.

## 8. Ngoài phạm vi

- Train model mới hoặc thêm lớp vật thể mới.
- Nguồn CCTV trực tiếp (RTSP). `FrameRing` được thiết kế để dùng được cho nguồn này sau, nhưng phase này
  chỉ chạy trên file.
- Giao diện desktop cho `REVIEW_SKIPPED` và cho trạng thái hàng đợi. Sự kiện đã có trong `events.jsonl`;
  phần hiển thị sẽ làm sau.
- TensorRT: Maxwell đã bị các bản TensorRT mới bỏ hỗ trợ, nên phải kiểm tra riêng trước khi dùng.
