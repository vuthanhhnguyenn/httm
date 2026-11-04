# Phase 2e: Bắt đủ sự kiện hơn ở chế độ CCTV (hai tầng) — Thiết kế

- Ngày: 2026-11-04
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-10-29-phase2d-cctv-accuracy-design.md`

## 1. Mục tiêu và điều đã chốt

Nguyên (2026-11-04): cả năm nhóm hành vi đều đang bị sót: (a) dùng điện thoại, (b) chuyền giấy/đồ vật,
(c) quay sang hoặc nhìn bài người khác, (d) xem tài liệu giấu, (e) khác. Được phép dùng code/model có
sẵn trên GitHub để không phải train.

Đã chốt cách chạy (c) "kết hợp": **tầng nhanh** chạy lúc thi trên Quadro M1200, chỉ khoanh *đoạn nghi
vấn* và ưu tiên không bỏ sót; **tầng kỹ** chạy sau buổi thi, không giới hạn tốc độ, xem kỹ từng đoạn
và quyết định sự kiện cuối cùng cho giám thị duyệt.

Hệ thống vẫn chỉ gợi ý để giám thị rà soát, không tự kết luận gian lận.

## 2. Hiện trạng và nguyên nhân sót

| Nguyên nhân | Ảnh hưởng |
|---|---|
| Luật chỉ dựa vào đầu và cổ tay (quay đầu, chụm đầu, tay vào khung ghế bên) | Không có luật cho (b) chuyền đồ, (d) tài liệu giấu, điện thoại dưới bàn |
| Điện thoại dò bằng YOLO COCO trên cả khung 1280×720 | Điện thoại ở CCTV chỉ vài chục pixel, gần như không bao giờ thấy; không có lớp "giấy" |
| `yolo11n-pose` (nano) | Hàng xa: mũi/mắt đặt lên tóc, hướng đầu sai (sự kiện sai 1:00 ở clip 1) |
| Điều kiện chặn báo nhầm chặt: phải ngẩng đầu (`min_head_lift`), mốc 10 s (`home_min_ms`), tay chỉ xét cùng hàng | Chặn được báo nhầm nhưng cũng chặn ca thật; 3 giây đầu clip không có mốc |
| Một clip có nhãn, nhãn không ghi loại hành vi | Không đo được tỉ lệ bắt theo từng hành vi |

## 3. Kiến trúc

```text
video ──► TẦNG NHANH (pipeline CCTV hiện có, ngưỡng nới) ──► suspects.jsonl (đoạn nghi vấn)
                                                              │
video gốc ──────────────────────────────────────────────────► TẦNG KỸ (mỗi đoạn: cắt, phóng to,
                                                              model lớn, luật theo hành vi)
                                                              ──► events.jsonl + ảnh bằng chứng
```

Chế độ CCTV hiện chỉ chạy trên file video (`scripts/analyze_video_multi.py --scene cctv`), nên tầng kỹ
đọc lại đúng đoạn trong file gốc. Bộ đệm video và lưu clip ngắn chỉ cần khi CCTV chạy trực tiếp trong
app; việc đó để sau, không làm trong phase này.

### 3.1 Tầng nhanh: khoanh đoạn nghi vấn

Giữ nguyên `CctvPipeline`, `SeatMapTracker`, `CctvSeatRules`. Thêm chế độ `--suspects` cho
`CctvSeatRules` (cấu hình riêng `CctvSuspectConfig`), mở một *đoạn nghi vấn* khi bất kỳ điều kiện nào
dưới đây đúng trong ≥ 0,5 s:

| Mã | Điều kiện | Hành vi nhắm tới |
|---|---|---|
| `HEAD_AWAY` | lệch hướng đầu so với mốc > 0,6 × ngưỡng hiện tại; không đòi ngẩng đầu | (c) |
| `LEAN` | đầu rời chỗ quen > 0,25 bề rộng khung, mọi hướng | (b), (c) |
| `HAND_OUT` | cổ tay ra ngoài khung ghế mình > 0,15 bề rộng, mọi hàng | (b) |
| `HANDS_MEET` | cổ tay hai ghế cạnh nhau (cùng hàng hoặc trước/sau) cách nhau < 0,5 bề rộng khung | (b) |
| `HEAD_DOWN_HANDS_LOW` | đầu cúi và cả hai cổ tay thấp hơn mép bàn ước lượng của ghế đó | (a), (d) |
| `OBJECT_HINT` | YOLO COCO thấy `cell phone`/`book` với độ tin cậy ≥ 0,15 trong khung ghế | (a), (d) |
| `RULE_EVENT` | bất kỳ sự kiện nào các luật hiện có sẽ mở | tất cả |

- *Mép bàn ước lượng*: trung vị y của cổ tay khi ghế đang viết (đầu cúi, cổ tay trong khung), học dần
  như `RollingBaseline`. Chưa đủ mẫu thì dùng 60 % chiều cao khung tính từ đỉnh.
- Mốc hướng đầu dùng ngay từ 1 s đầu (`min_span_ms` ngắn cho tầng nhanh) để không sót đầu clip.
- Mỗi đoạn: `seat_id`, ghế lân cận liên quan, `start_ms`, `end_ms` (nới thêm 2 s mỗi phía), danh sách
  mã lý do, khung ghế. Các đoạn chồng nhau của cùng ghế được gộp.
- Tầng nhanh vẫn ghi `events.jsonl` như cũ (sự kiện tạm thời) để so với trước.

Ràng buộc tốc độ: tầng nhanh trên M1200 ở `--fps 5` không chậm hơn hiện tại quá 10 % (đang ~10 FPS).
Các điều kiện mới chỉ tính trên keypoint đã có, không thêm model.

### 3.2 Tầng kỹ: xác nhận theo hành vi

Script mới `scripts/review_suspects.py <video> <suspects.jsonl> --out <dir>`. Với mỗi đoạn:

1. Đọc lại video gốc trong đoạn ở 10 khung/giây.
2. Cắt vùng gồm ghế nghi vấn và các ghế lân cận (nới 30 %), phóng to cạnh dài lên 960 px.
3. Chạy trên ảnh cắt, tất cả là model tải sẵn, không train:
   - **Khung xương**: `yolo11x-pose` (Ultralytics, cùng nguồn và license AGPL với model đang dùng).
   - **Vật thể COCO**: `yolo11x` cho `cell phone`, `book`.
   - **Giấy/tài liệu**: YOLOE (Ultralytics, open-vocabulary, gợi ý bằng chữ) với các lớp
     `paper`, `note`, `phone`. Dùng được hay không do bước thử ở mục 6 quyết định.
4. Áp luật theo hành vi (tọa độ đưa về khung hình gốc):

| Sự kiện | Luật (giữ ≥ thời gian) |
|---|---|
| `PHONE_DETECTED` | điện thoại (COCO ≥ 0,35 hoặc YOLOE) có tâm trong khung ghế, gần cổ tay (< 0,3 bề rộng) hoặc dưới mép bàn; ≥ 3 khung trong 1 s |
| `PASSING_OBJECT` (mới) | cổ tay hai ghế gặp nhau (< 0,3 bề rộng) trong ≥ 0,3 s, hoặc vật (giấy/sách/điện thoại) đổi từ khung ghế này sang khung ghế kia trong ≤ 2 s; báo cả hai ghế |
| `LOOK_AT_NEIGHBOR`, `HEAD_TURN_*`, `TURN_AROUND`, `HEADS_TOGETHER` | luật hiện có, chạy trên keypoint của `yolo11x-pose`; giữ quy tắc chỉ báo người nghiêng/quay sang (phase 2d) |
| `DOCUMENT_DETECTED` | giấy/sách không phải tờ bài thi (ngoài vùng mặt bàn trước mặt, hoặc dưới mép bàn, hoặc trong tay đưa lên ngang mặt) ≥ 1 s |
| `LOOK_AT_LAP` (mới) | đầu cúi và cả hai cổ tay dưới mép bàn ≥ 3 s liên tục |

5. Đoạn không đạt luật nào được ghi `dismissed` cùng lý do, để đo tầng kỹ có loại oan ca thật không.
6. Mỗi sự kiện có ảnh bằng chứng (khung giữa sự kiện, vẽ khung ghế, keypoint, vật thể) như webcam mode.

(e) "khác": chưa có luật riêng; đoạn nghi vấn không khớp hành vi nào nhưng có ≥ 2 mã lý do khác nhau ở
tầng nhanh thì ghi `UNCLASSIFIED_SUSPECT` mức thấp để giám thị xem, không cộng điểm.

### 3.3 Không làm trong phase này

- Nhận dạng hành động bằng model video (PoseC3D/NTU-120, SCB-Dataset): dữ liệu khác góc nhìn, giấy phép
  SCB chỉ phi thương mại. Cân nhắc lại khi luật đã đo được trên ≥ 3 clip.
- Bộ đệm video và tầng kỹ chạy trong app desktop.
- Dò mắt/miệng ở CCTV (quá nhỏ).

## 4. Bộ đo

- Nhãn YAML (`data/labels/*.yaml`) thêm trường `behavior: phone | passing | look | notes | other`. Nhãn
  cũ của clip 1 đều gán `look`; Nguyên xem lại `wall-pair` (chị hàng trước giơ giấy lúc 0:53) có nên là
  `passing` không.
- `scripts/eval_cctv.py` báo thêm:
  - tỉ lệ bắt theo từng `behavior`;
  - `--suspects`: tỉ lệ nhãn nằm trong ít nhất một đoạn nghi vấn đúng vùng (độ phủ tầng nhanh), và
    tổng thời lượng đoạn nghi vấn / thời lượng clip (để biết tầng kỹ phải xem bao nhiêu).
- Clip: clip 1 (`EPf9O7UXRoM`), clip 2 (`LUy81HI7wS0`, cần Nguyên gắn nhãn), và nên có **một clip tự
  dàn dựng** (vài người, camera đặt cao như CCTV, mỗi hành vi (a)–(d) diễn 2–3 lần), vì clip công khai
  rất ít cảnh dùng điện thoại hay chuyền giấy.

## 5. Tiêu chí đạt

- Độ phủ tầng nhanh ≥ 95 % nhãn trên các clip có nhãn; thời lượng đoạn nghi vấn ≤ 30 % thời lượng clip.
- Tầng kỹ: báo tỉ lệ bắt theo từng hành vi và số báo sai/phút. Không đặt ngưỡng cứng trước khi có clip
  dàn dựng; mục tiêu là cao hơn bản phase 2d ở mọi hành vi có nhãn, và clip 1 không tệ hơn 5/6.
- Tốc độ tầng nhanh như mục 3.1. Tầng kỹ: ghi thời gian xử lý mỗi đoạn vào `timing.json`, không giới hạn.

## 6. Rủi ro và bước thử trước

- **YOLOE trên ảnh cắt CCTV**: thử đầu tiên trên 10 ảnh cắt từ clip 1 (chị giơ giấy 0:53, bàn có giấy).
  Nếu không thấy giấy, bỏ YOLOE; (d) chỉ còn `LOOK_AT_LAP` và `book` COCO, và ghi rõ giới hạn này.
- **YOLOE cần tải MobileCLIP lúc đặt lớp chữ**: tính sẵn embedding một lần, lưu cùng model, ghi SHA-256
  vào `models/manifest.yaml`, để máy Windows không phải tải gì khi chạy.
- **`yolo11x-pose` trên ảnh cắt có chậm quá không**: offline nên chấp nhận được; nếu một clip 70 s mất
  quá 10 phút trên M1200 thì hạ xuống `yolo11l-pose`.
- **Khớp quá với ít clip**: tham số mới đặt trong `CctvSuspectConfig`/cấu hình tầng kỹ, không viết cứng;
  báo kết quả riêng từng clip.

## 7. Kiểm thử

- Unit test cho từng điều kiện tầng nhanh và từng luật tầng kỹ bằng keypoint/khung tổng hợp (như
  `tests/` hiện có cho `CctvSeatRules`).
- Test gộp đoạn nghi vấn, đổi tọa độ ảnh cắt về khung gốc, và `eval_cctv.py` với `behavior`/`--suspects`.
- Chạy đầu-cuối trên clip 1 trên cloud; so với kết quả phase 2d trong `docs/evaluation-results.md`.
