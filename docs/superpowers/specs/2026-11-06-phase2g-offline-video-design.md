# Phase 2g: Từ video thi ra video đã đánh dấu, chính xác nhất trong 2× thời lượng — Thiết kế

- Ngày: 2026-11-06
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-11-05-phase2f-cctv-precision-speed-design.md` (PR #42, chưa gộp). Nhánh này xây tiếp trên
  code của PR #42 (cache tầng kỹ, `review_core`, luật mới, `review_score`).

## 1. Mục tiêu và điều đã chốt

Nguyên (2026-11-06): đưa vào **một file video phòng thi**, nhận ra **video đã đánh dấu sự kiện** chính xác
nhất có thể, ít lỗi nhất, trong thời gian chấp nhận được: **tổng thời gian ≤ 2 × thời lượng video**
(video 1 phút xong trong khoảng 2 phút). Không cần chạy live.

Đã chốt:

| Câu hỏi | Nguyên chọn |
|---|---|
| Máy chuẩn cho mốc 2× | **Kaggle Notebook, GPU T4** (16 GB; có khi 2 × T4). Model tầng kỹ giữ `yolo11x-pose` + `yolo11x`. |
| Ít sót hay ít nhầm | **Cân bằng, có mức tin cậy**: đỏ = gần như chắc, vàng = nên xem lại. |
| Đầu ra | **Video + báo cáo sự kiện** (HTML và CSV: ảnh, mốc giây, mức tin cậy). |
| Hướng làm | **Ba lượt offline** (nhanh → kỹ trọn đoạn → dựng video). |

Hệ thống vẫn chỉ gợi ý để người xem rà soát, không tự kết luận gian lận.

## 2. Vì sao chọn chạy trọn đoạn

Đo trên clip 1 (`EPf9O7UXRoM`, 72 s), cùng một cache đầu ra model, chấm theo 23 ảnh Nguyên đã đánh giá
(9 đúng, 14 nhầm), xem `docs/evaluation-results.md`:

| Cách xem đoạn nghi vấn | Đúng (9) | Ảnh nhầm |
|---|---|---|
| Trọn đoạn, 5 khung/giây (offline, phase 2f) | 9 | 6 |
| Khúc 10 s, 3 khung/giây (live, phase 2f) | 9 | 4, nhưng clip 2 ra thêm 2 cảnh báo quay đầu |
| 12 khung mỗi đoạn | 0 | 3 |

Vì không cần live, ta dùng cấu hình offline trọn đoạn ở 5 khung/giây, và dồn công sức vào tốc độ GPU thay
vì bớt khung.

## 3. Luồng chạy

Một lệnh: `scripts/process_video.py VIDEO --out THU_MUC [--quality max|balanced] [--device 0] [--device2 1]`.
Ghi ra `THU_MUC/`:

| File | Nội dung |
|---|---|
| `annotated.mp4` | video gốc (đủ khung hình gốc), H.264, có khung ghế và cảnh báo đỏ/vàng đang mở |
| `report.html` | danh sách sự kiện, ảnh bằng chứng, mốc giây (bấm để nhảy tới trong video nhúng), mức tin cậy, lý do |
| `events.csv`, `events.jsonl` | cùng nội dung dạng bảng / máy đọc |
| `evidence/` | ảnh bằng chứng từng sự kiện |
| `review_cache.json.gz` | đầu ra model tầng kỹ (chạy lại luật bằng `replay_review.py` không cần GPU) |
| `timing.json` | thời gian từng lượt, thiết bị, tỉ lệ `total_s / video_s` |

### 3.1 Lượt 1: nhanh

Tầng nhanh hiện có (`analyze_video_multi.py --scene cctv --mode full --fps 5 --suspects`), gọi như hàm thay
vì tiến trình con, không ghi video. Thêm: ghi `tracks.jsonl`, mỗi dòng là một khung đã phân tích
`{t, seats: {seat_id: [x, y, w, h, role]}}`, để lượt 3 vẽ khung ghế mà không chạy lại model. Đầu ra:
`suspects.jsonl`, sự kiện tầng nhanh, `tracks.jsonl`.

### 3.2 Lượt 2: kỹ, trọn đoạn

Mỗi đoạn nghi vấn được xem ở 5 khung/giây từ `start − 6 s` tới `end`, giống `review_suspects.py
--max-frames 0` hiện nay, luật và kết quả không đổi. Chỉ đổi cách chạy để nhanh:

1. **Giải mã một lần.** Gom mọi mốc khung mà các đoạn cần, đọc video tuần tự một lượt (không `seek` từng
   khung), mỗi khung dùng chung cho mọi đoạn cần nó.
2. **Crop dùng chung.** Mỗi đoạn có vùng crop (ghế + 2 ghế gần nhất, `crop_region`). Ở cùng một khung, các
   đoạn có vùng crop chồng nhau ≥ 70 % được gộp thành một crop chung, model chạy một lần, kết quả
   `assign_poses` cho từng đoạn theo ghế của nó.
3. **Batch lớn và FP16.** Crop được đẩy qua model theo batch 16–32 (cấu hình), `half=True` trên CUDA. Hết VRAM
   thì giảm batch như hiện có.
4. **Thiết bị.** `process_video.py` đặt thiết bị cho cả hai model từ `--device`/`--device2` (mặc định `cuda`
   cho cả hai), bỏ qua `device: intel:gpu` trong `configs/review_models.yaml` (máy Kaggle không có card Intel;
   để nguyên sẽ rơi về CPU).
5. **Hai card song song.** Có card thứ hai (Kaggle 2 × T4) thì `yolo11x` chạy trên card 2 cùng lúc với
   `yolo11x-pose` trên card 1, mỗi model một luồng.
6. **Ghi cache** như hiện tại, rồi chạy luật (`review_window` + `ReviewDeduper`) theo thứ tự thời gian.

`--quality balanced` (dự phòng khi `max` vượt mốc 2×): hạ `imgsz` của crop tầng kỹ từ 960 xuống 640. Chỉ bật
mặc định nếu số đo trên Kaggle cho thấy `max` không đạt (mục 5).

### 3.3 Lượt 3: dựng

1. **Gộp sự kiện.** Sự kiện tầng kỹ là chính. Sự kiện luật của tầng nhanh dùng để xác nhận (mục 4). Đoạn mà
   tầng kỹ chỉ trả `UNCLASSIFIED_SUSPECT` thành sự kiện vàng "nghi vấn chưa rõ".
2. **Mức tin cậy** cho từng sự kiện (mục 4).
3. **Vẽ video.** Đọc video gốc một lượt với đủ khung gốc. Mỗi khung lấy khung ghế từ `tracks.jsonl` gần nhất
   (≤ 200 ms), vẽ ghế mảnh màu xám, ghế có cảnh báo đang mở vẽ dày màu đỏ hoặc vàng kèm nhãn tiếng Việt
   (dùng `put_text` có sẵn trong `review_suspects.py`, chuyển sang module chung). Góc trên có dòng đếm
   "Đỏ: n · Vàng: m" tới thời điểm đó. Mỗi cảnh báo được giữ trên hình ít nhất 2 s cho dễ nhìn.
4. **Mã hoá.** Ghi khung thô qua ống vào `ffmpeg -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p`
   (có trên Kaggle). Không có ffmpeg thì ghi `mp4v` bằng OpenCV và ghi cảnh báo vào `timing.json`.
5. **Báo cáo.** `report.html` một file, ảnh nhúng dạng đường dẫn tương đối tới `evidence/`, sắp đỏ trước rồi
   theo thời gian, có bộ lọc đỏ/vàng/loại. Không cần mạng để mở.

## 4. Mức tin cậy

Mỗi sự kiện có `level` (`red` | `yellow`) và `level_reason` (một câu tiếng Việt). Luật nằm trong
`configs/confidence.yaml`, đọc bởi `proctoring_ai/multi/cctv/confidence.py`:

| Loại | Đỏ khi | Còn lại |
|---|---|---|
| LOOK_AT_NEIGHBOR, HEADS_TOGETHER | luôn | — |
| DOCUMENT_DETECTED | luôn (luật đã đòi tài liệu ở trong tay hoặc đang di chuyển) | — |
| PASSING_OBJECT | có vật thể giữa hai bàn tay, hoặc vật chuyển từ ghế này sang ghế kia (`_transfers`) | vàng: chỉ thấy tay chạm tay |
| PHONE_DETECTED | điện thoại giơ cao hơn mép bàn và độ tin cậy ≥ 0,6 | vàng (máy tính bỏ túi, vật trên bàn) |
| TURN_AROUND, HEAD_TURN_LEFT/RIGHT | tầng nhanh có sự kiện luật cùng ghế, lệch ≤ 3 s | vàng |
| LOOK_AT_LAP | — | luôn vàng (pose không tách được ca đúng/nhầm trên clip 1) |
| UNCLASSIFIED_SUSPECT | — | luôn vàng, "nghi vấn chưa rõ" |

Để biết PASSING có vật hay không, `review_rules` ghi thêm vào bản ghi `opened` trường `evidence_kind`
(`"object_between"`, `"transfer"`, `"hands_only"`), và PHONE ghi `raised` (bool) và `confidence`. Đây là thông tin
luật đã tính, không đổi kết quả luật.

**Mục tiêu trên clip 1**: cả 9 ca đúng có mặt trong video (đỏ hoặc vàng, tính theo `true_covered`), và
**không ảnh nhầm nào là đỏ**. `review_score` thêm `false_red` (ảnh nhầm khớp sự kiện đỏ) và `true_red`.

## 5. Tốc độ: mốc và cách đo

Mốc: `total_s ≤ 2 × video_s` trên Kaggle T4, đo bằng `timing.json` (gồm cả nạp model, không gồm cài thư viện
và tải model lần đầu).

Ước lượng cho clip 1 (72 s, mốc 144 s), cần đo lại: lượt nhanh ~15 s (360 khung), lượt kỹ ~80 s (khoảng 2000
crop × 2 model x, FP16), dựng ~15 s. Tổng ~110 s.

**Bước đầu của kế hoạch là notebook đo** `notebooks/kaggle_benchmark.ipynb`: chạy clip 1 trên Kaggle, in thời
gian từng lượt, số crop trước và sau khi gộp, tốc độ từng model theo batch 8/16/32 và FP16/FP32, 1 và 2 card.
Nguyên chạy và gửi `timing.json`. Từ số đo đó chốt mặc định (batch, `--quality`, có gộp crop hay không). Nếu
cả `balanced` vẫn vượt mốc thì báo số đo và đề xuất (`yolo11l`, hoặc TensorRT) để Nguyên quyết.

Trên cloud CPU (không có GPU) các lượt vẫn chạy được để kiểm thử, nhưng không dùng để đo mốc.

## 6. Kaggle

`notebooks/kaggle_process_video.ipynb`:

1. Bật Internet và GPU trong cài đặt notebook (ghi hướng dẫn ở ô đầu).
2. `git clone` repo (nhánh chỉ định ở một biến đầu notebook; repo private thì dùng Kaggle Secret
   `GITHUB_TOKEN`), `pip install` các gói cần (ultralytics, opencv, mediapipe, pyyaml, openvino không cần).
3. `scripts/download_model.py` tải model.
4. Chạy `process_video.py` cho mọi `*.mp4` trong `/kaggle/input/<dataset>/`, kết quả ở
   `/kaggle/working/ket-qua/<tên video>/`, nén thành `ket-qua.zip` để tải về.
5. In bảng tóm tắt: thời lượng, thời gian chạy, tỉ lệ, số sự kiện đỏ/vàng.

Ô đầu nhắc: để Dataset ở chế độ Private, video thí sinh là dữ liệu cá nhân.

## 7. Xử lý lỗi

- Video không mở được, hoặc 0 khung: thoát mã 1, thông báo tiếng Việt.
- Không có GPU: chạy CPU, ghi `device_reasons` như hiện có, in cảnh báo "sẽ chậm hơn nhiều".
- Một đoạn lỗi ở lượt kỹ (model lỗi, khung hỏng): đoạn đó thành sự kiện vàng "chưa kiểm tra kỹ" (dùng
  `REVIEW_SKIPPED`), các đoạn khác vẫn chạy.
- ffmpeg lỗi giữa chừng: giữ file `mp4v` dự phòng, ghi lỗi vào `timing.json`.
- Thời gian vượt mốc không làm dừng chạy: chỉ ghi `over_budget: true` và in cảnh báo.

## 8. Kiểm thử

- **Hồi quy độ chính xác** (cloud, không GPU, dùng fixture cache clip 1 có sẵn): chạy lượt 2–3 (từ cache) và
  chấm: `true_covered == 9`, `false_red == 0`, `false_opened ≤ 7`.
- **Lượt 2 mới cho kết quả giống cũ**: trên một video tổng hợp ngắn và detector giả, đường giải mã một lần +
  gộp crop + batch cho cùng `ReviewFrame` (sai số ≤ 1e-4) như cách đọc từng đoạn hiện nay.
- **Gộp crop**: hai đoạn có vùng crop trùng nhau ≥ 70 % chỉ gọi model một lần cho khung chung.
- **Mức tin cậy**: mỗi dòng của bảng mục 4 có một test.
- **Dựng video**: video tổng hợp 3 s → `annotated.mp4` có đủ số khung gốc, khung có cảnh báo có điểm ảnh đỏ
  ở vùng ghế; không có ffmpeg thì vẫn ra file.
- **Báo cáo**: HTML có một dòng cho mỗi sự kiện, đỏ đứng trước.
- **Đầu-cuối trên cloud CPU**: 10 s đầu clip 1, đủ các file đầu ra.
- **Trên Kaggle** (Nguyên chạy): clip 1 và clip 2, mốc 2× và các con số ở mục 4.

## 9. Ngoài phạm vi

- Chạy live, `--review-live` (giữ nguyên như PR #42, không phát triển thêm).
- Huấn luyện model mới, thêm loại hành vi mới.
- Giao diện desktop.
