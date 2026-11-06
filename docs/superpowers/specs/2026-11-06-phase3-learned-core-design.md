# Phase 3: Lõi nhận diện học từ nhãn (thay ngưỡng viết tay) — Thiết kế

- Ngày: 2026-11-06
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-11-06-phase2g-offline-video-design.md` (offline, Kaggle T4). Nghiên cứu nền:
  tài liệu "Nghiên cứu: lõi nhận diện gian lận CCTV" (https://claude.ai/artifact/9duvvWcqYsbpCD8Dgeq43b).

## 1. Mục tiêu và điều đã chốt

Nguyên (2026-11-06): video ra từ phase 2g "sai rất nhiều", **cả hai chiều**: người không làm gì vẫn bị bắt,
người có sự kiện lại bị bỏ qua. Mốc thời gian nới lên **khoảng 3 × thời lượng video** trên Kaggle T4.
Nguyên muốn lõi nhận diện được viết lại theo cách tốt nhất mà các nguồn khoa học và repo đã chứng minh.

| Câu hỏi | Nguyên chọn |
|---|---|
| Lỗi nào quan trọng | Báo nhầm và bỏ sót **ngang nhau** |
| Ai gán nhãn | **Claude tự gán**, Nguyên duyệt nhanh khoảng 50 đoạn |
| Hướng | **Bắt rộng rồi lọc bằng mô hình học** (không dùng VLM làm trọng tài, không chỉnh ngưỡng tiếp) |
| Thiết kế 4 tầng, đặc trưng, cách đo | Đồng ý (09:08) |

Hệ thống vẫn chỉ gợi ý để người xem rà soát, không tự kết luận gian lận.

## 2. Vì sao phải đổi lõi

Lần chạy Kaggle ngày 2026-11-06 trên clip 1 (`EPf9O7UXRoM`, 72 s, chấm theo 23 phán quyết của Nguyên):

- **Sót vì cổng trước.** Tầng kỹ chỉ xem đoạn mà tầng nhanh (yolo11n) nghi. Trên GPU tầng nhanh ra 32 đoạn
  thay vì 30 như bản CPU, và 4/9 ca đúng mất: chuyển vật 25,5, chụm đầu 47,4, nhìn đùi 51,2 và 59,0.
- **Sót vì ngưỡng sát mép.** Khoảng 40 ngưỡng hình học chỉnh trên một clip; khung xương GPU lệch rất ít
  (trung vị 0,004–0,01) nhưng đủ làm "nhìn xuống đùi" đóng sớm.
- **Nhầm vì luật không phân biệt được.** Người cúi viết và người nhìn đùi cho cùng hình dạng khung xương:
  9 cảnh báo vàng LOOK_AT_LAP ở clip 1; ở clip 2, nghi vấn chưa rõ chiếm 10/14 cảnh báo.

Các nguồn (bảng trong tài liệu nghiên cứu) thống nhất ba điều: quyết định trên **đoạn khoảng 3 giây quanh
từng người** (SAV: 16 khung là đủ), **cắt riêng từng người** thay vì cả khung (+16 điểm, arXiv 2604.16234),
và với ít dữ liệu thì **giữ phần trích đặc trưng cố định, chỉ train một bộ phân loại nhỏ** (Cheatomaly).
VLM nhỏ chưa đủ tin cậy (Qwen2.5-VL-7B khoảng 35% trên UCF-Crime). SlowFast, VideoMAE fine-tune toàn bộ
và ST-GCN cần hàng nghìn mẫu, nên để sau.

## 3. Kiến trúc: 4 tầng

```
video ──► 1. Dòng thời gian từng ghế ──► 2. Cửa sổ 3 s + đặc trưng ──► 3. LightGBM ──► 4. Ghép sự kiện ──► render + báo cáo (giữ nguyên)
                                                   │                          ▲
                                                   └── đoạn mẫu ──► gán nhãn ──► train (Kaggle) ──► model file
```

### 3.1 Tầng 1: dòng thời gian từng ghế

- Tầng nhanh hiện có vẫn chạy để dựng **bản đồ ghế** (`seat_map.py`) và track, nhưng **không còn là cổng**.
- Mỗi ghế (kèm các ghế kề) trở thành một `SuspectWindow` phủ **toàn bộ video** (lý do `"ALL_SEATS"`), rồi đi qua
  `review_offline` như hiện nay: giải mã một lần, cắt ảnh phóng to quanh cụm, `yolo11x-pose` + `yolo11x`,
  FP16, batch, 2 GPU, 5 khung/giây. Các ghế có vùng cắt chồng nhau dùng chung một ảnh cắt trong cùng khung (`_group`, `share_iou`).
- Đầu ra: `timeline.json.gz` (định dạng `review_cache`, mỗi ghế một dòng), chứa là `(ghế, offset_ms, keypoints, vật thể gần ghế, hộp ghế)`. Đây cũng là
  cache để train lại mà không chạy GPU lại.

### 3.2 Tầng 2: cửa sổ 3 giây và đặc trưng

- Mỗi ghế, cửa sổ dài 3 000 ms, trượt 1 000 ms. Khoảng 15 khung mỗi cửa sổ ở 5 khung/giây.
- Đặc trưng (module mới `multi/cctv/features.py`, hàm thuần, không phụ thuộc GPU). Mọi khoảng cách đều chuẩn hoá
  theo kích thước hộp ghế:
  - **Đầu:** yaw và lift so với **trung vị của chính ghế đó trên cả video** (chạy offline mới có mốc này);
    trung bình, max, độ lệch chuẩn, tỷ lệ khung vượt 0,35.
  - **Tay:** vị trí cổ tay so với mép bàn (`DeskLine`), tỷ lệ khung tay dưới bàn, tốc độ tay lớn nhất,
    độ vươn ra ngoài hộp ghế về phía người bên cạnh.
  - **Cặp ghế:** với người kề gần nhất (cùng hàng hoặc hàng trước/sau), khoảng cách cổ tay–cổ tay nhỏ nhất,
    khoảng cách đầu–đầu so với trung vị của cặp, độ nghiêng về phía nhau.
  - **Vật:** độ tin cậy lớn nhất của điện thoại/giấy gần cổ tay, có giơ cao hơn mép bàn không, có di chuyển không.
  - **Luật cũ thành cờ:** mỗi luật hiện có (`review_window` và `CctvSeatRules`) có kích hoạt trong cửa sổ hay
    không. Luật không còn tự ra sự kiện.
  - **Độ tin cậy:** điểm keypoint trung bình, chiều cao hộp ghế so với khung (hàng xa), số khung có người.
  - **Tuỳ chọn (mục 6):** điểm "trông như gian lận" từ bộ phân loại ảnh cắt.
- Đầu ra: bảng `features.csv` (một dòng mỗi ghế mỗi cửa sổ) cùng tên cột cố định `FEATURE_NAMES`.

### 3.3 Tầng 3: bộ phân loại

- **LightGBM** đa lớp (giấy phép MIT), 8 lớp: `NORMAL`, `LOOK_AT_NEIGHBOR`, `HEADS_TOGETHER`, `PASSING_OBJECT`,
  `PHONE_DETECTED`, `DOCUMENT_DETECTED`, `LOOK_AT_LAP`, `TURN_AROUND`. Trọng số lớp cân bằng theo tần suất.
  `TURN_AROUND` gộp chung với nhìn bạn nếu có dưới 10 mẫu; quyết định này ghi lại trong log khi train.
- Model lưu ở `models/learned_core/<phiên bản>/model.txt` kèm `meta.json` (tên đặc trưng, ngưỡng đỏ/vàng, các video đã
  train). Khi suy luận, đặc trưng thiếu hoặc khác tên thì báo lỗi rõ ràng, không đoán.
- Train chạy trên CPU (vài giây cho khoảng 2 000 dòng). Notebook Kaggle chỉ cần cho tầng 1.

### 3.4 Tầng 4: ghép thành sự kiện

- Với mỗi ghế và mỗi lớp, chuỗi xác suất theo cửa sổ được làm mượt bằng **trễ hai ngưỡng**: mở khi
  `p ≥ p_open` ở 2 cửa sổ liền nhau, đóng khi `p < p_close` (mặc định 0,5 và 0,35).
- Mức tin cậy: **đỏ** nếu xác suất lớn nhất trong sự kiện ≥ `p_red`, **vàng** nếu ≥ `p_yellow`. Ngưỡng
  `p_red` chọn trên tập kiểm tra sao cho độ đúng ≥ 90%; `p_yellow` sao cho ≥ 50%. Ngưỡng ghi vào `meta.json`,
  thay `configs/confidence.yaml` khi chạy lõi mới.
- Sự kiện ghép ra có cùng dạng bản ghi `opened`/`closed` như hiện nay, kèm `probability` và `stage: "learned"`.
  Vì vậy `ReviewDeduper`, render và báo cáo giữ nguyên. Sự kiện cặp (chuyển vật, chụm đầu) gán cho người vươn
  sang, như quy tắc phase 2d (`_reacher`).

## 4. Nhãn

- **Dữ liệu:** clip 1, clip 2, và 3–5 video phòng thi mới Nguyên tải lên dataset Kaggle Private.
- **Chọn đoạn để gán:** mọi cửa sổ có ít nhất một cờ luật hoặc đặc trưng vượt phân vị 95 của ghế đó, cộng 20%
  số cửa sổ yên lặng chọn ngẫu nhiên (hạt giống cố định) làm mẫu âm. Cửa sổ chồng nhau của cùng ghế gộp thành
  một đoạn khi gán.
- **Ảnh ghép:** script `scripts/label_sheets.py` xuất mỗi đoạn một ảnh gồm 6 khung, cắt quanh ghế có viền
  ghế đích và để thấy người kề bên, đánh số đoạn. Claude xem từng ảnh và gán 1 trong 8 lớp hoặc `UNSURE`
  (bị loại khi train).
- **Định dạng:** `data/labels/learned/<video>.jsonl`, mỗi dòng
  `{video, seat_box, start_ms, end_ms, label, labeler, note}`. Ghi hộp ghế để khớp lại khi id ghế đổi giữa các
  lần chạy (khớp theo IoU ≥ 0,5).
- **Kiểm tra:** 23 phán quyết của Nguyên ở clip 1 là chuẩn để đo độ khớp của nhãn Claude. Nguyên duyệt ngẫu
  nhiên 50 đoạn qua trang xem nhanh. Nếu khớp dưới 85% thì gán lại các lớp lệch nhiều nhất.
- **Vòng hai:** sau lần train đầu, gán thêm những cửa sổ có xác suất lớn nhất trong khoảng 0,35–0,65 (mô hình
  phân vân nhất).
- Video và ảnh thí sinh là dữ liệu cá nhân: ảnh ghép chỉ nằm trong project files hoặc dataset Private, không đưa
  vào repo. Repo chỉ chứa file nhãn (toạ độ, không ảnh).

## 5. Đo, mục tiêu, thời gian

- **Cách chấm:** train trên các video còn lại, chấm trên video để riêng, lần lượt từng video. Chấm theo sự kiện
  (cùng ghế, lệch ≤ 1,5 s, dùng lại `review_score` cho clip 1 và nhãn mới cho video khác). Báo cáo bỏ sót và báo
  nhầm cho từng lớp và từng video.
- **Mục tiêu nghiệm thu** (trên clip 1 để riêng, so với lần chạy Kaggle 2026-11-06):
  - Ca đúng được bắt ≥ 8/9 (hiện 5/9).
  - Số ảnh báo nhầm giảm ít nhất một nửa.
  - Không có ca nhầm nào ở mức đỏ.
  - Trên các video khác: bỏ sót và báo nhầm đều thấp hơn lõi cũ.
- **Thời gian:** tổng ≤ 3 × thời lượng trên Kaggle T4. Đo tầng 1 trước tiên bằng notebook bench. Nếu vượt, lần
  lượt hạ xuống 4 khung/giây, rồi dùng `yolo11l`, rồi dùng TensorRT; mỗi bước đều chấm lại độ chính xác.
- **Chuyển đổi an toàn:** `process_video.py --core learned|rules` (mặc định `rules`). Chỉ đổi mặc định sang
  `learned` khi đạt mục tiêu nghiệm thu. Thiếu file model thì báo lỗi kèm hướng dẫn, không lặng lẽ quay về lõi cũ.

## 6. Tuỳ chọn: tận dụng dataset ảnh trên Kaggle

Nguyên có dataset ảnh phòng thi trên Kaggle. Đang chờ link để xem nhãn và giấy phép. Có hai cách dùng, mỗi
cách đo độ lợi riêng và chỉ giữ lại nếu cải thiện độ chính xác trên video để riêng:

1. **Fine-tune bộ dò vật** (`yolo11x` cho điện thoại và giấy/tài liệu) trên ảnh có hộp vật thể.
2. **Bộ chấm ảnh cắt** (backbone ảnh đóng băng cộng đầu phân loại nhỏ) cho điểm "trông như gian lận" mỗi khung.
   Lấy trung bình và giá trị lớn nhất trong cửa sổ làm đặc trưng cho LightGBM. Không bao giờ dùng riêng lẻ.

Dataset nào có giấy phép chỉ cho phi thương mại hoặc không ghi giấy phép thì chỉ dùng để thử nghiệm, và ghi rõ
trong `meta.json`.

## 7. Thành phần mới và thay đổi

| Thành phần | Việc |
|---|---|
| `multi/cctv/timeline.py` + `scripts/build_timeline.py` | Mỗi ghế một cửa sổ toàn video → `review_offline` → `timeline.json.gz` |
| `multi/cctv/features.py` | Cửa sổ 3 s + `FEATURE_NAMES`, hàm thuần, test bằng khung giả |
| `multi/cctv/learned.py` | Nạp model, dự đoán, trễ hai ngưỡng, gán đỏ/vàng, ra bản ghi sự kiện |
| `scripts/label_sheets.py` | Chọn đoạn cần gán và xuất ảnh ghép |
| `scripts/train_core.py` | Train LightGBM, chấm theo từng video để riêng, chọn ngưỡng, ghi model + `meta.json` |
| `scripts/process_video.py` | `--core learned\|rules`, `--model` |
| `notebooks/kaggle_timeline.ipynb` | Chạy tầng 1 cho mọi video trong dataset, đo thời gian, zip timeline |
| `pyproject.toml` | Thêm `lightgbm` vào nhóm `ai` |

Lõi cũ (`rules.py`, `review_rules.py`) giữ nguyên và vẫn được gọi để làm cờ đặc trưng.

## 8. Lỗi và trường hợp biên

- Ghế không có người trong cửa sổ (đi vắng): bỏ qua cửa sổ đó, không dự đoán.
- Ghế xuất hiện dưới 10 s trên cả video: chưa có mốc nền ổn định, dùng trung vị của tất cả các ghế cùng hàng.
- Video ngắn hơn 3 s: không có cửa sổ, báo cáo rỗng kèm ghi chú.
- Model train trên video có độ phân giải hoặc góc khác hẳn: `meta.json` ghi cỡ khung và chiều cao ghế trung vị
  lúc train; khi lệch quá 2 lần thì báo cáo có cảnh báo "ngoài miền dữ liệu đã train".

## 9. Kiểm thử

- Đơn vị: `features.py` trên khung giả (đầu quay, tay vươn, điện thoại gần tay) ra đúng giá trị; trễ hai
  ngưỡng; ghép sự kiện cặp cho người vươn; khớp nhãn theo hộp ghế.
- Tích hợp: từ cache clip 1 (`tests/fixtures/cctv_clip1_review`) với một model nhỏ train trên dữ liệu giả, chạy
  `--core learned` ra `events.jsonl` đúng dạng và render được.
- Đánh giá: `train_core.py --leave-one-out` in bảng mục tiêu ở mục 5. Đây là bằng chứng nghiệm thu, chạy bằng
  tay, không đưa vào CI.

## 10. Thứ tự làm

1. Notebook tầng 1 cùng bench thời gian. Nguyên chạy trên Kaggle cho mọi video.
2. `features.py` và `label_sheets.py`. Claude gán nhãn, Nguyên duyệt 50 đoạn.
3. `train_core.py` với chấm theo từng video để riêng, rồi `learned.py` và `--core learned`.
4. Vòng gán nhãn thứ hai, chấm lại, quyết định có đổi mặc định không.
5. Tuỳ chọn mục 6 khi có link dataset ảnh.
