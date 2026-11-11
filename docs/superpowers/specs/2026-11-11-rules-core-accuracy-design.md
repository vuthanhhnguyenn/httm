# Tối ưu lõi luật CCTV lên 90% đúng — Thiết kế

- Ngày: 2026-11-11
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-11-06-phase3-learned-core-design.md` (PR #43, chưa gộp), tùy chọn `--all-seats`

## 1. Mục tiêu và điều đã chốt

Nguyên (2026-11-11) muốn lõi luật nhận diện đúng khoảng 90–95%, chấp nhận chạy lâu hơn một chút. Đây là BTL.

- Hướng **C** (Nguyên chọn): chỉ sửa luật và cách dùng model sẵn có. **Không** dùng model ngôn ngữ-hình
  ảnh, kể cả API ngoài lẫn model mở.
- Làm cả 6 cải tiến theo thứ tự **6 → 1 → 3 → 2 → 4 → 5** (Nguyên "đồng ý"), mỗi bước có số đo
  trước/sau để đưa vào báo cáo.
- Hệ thống vẫn chỉ gợi ý cho giám thị, không tự kết luận gian lận.
- Video học sinh là dữ liệu cá nhân: ảnh và tờ chấm không vào repo, nhãn chỉ chứa tọa độ.

**Thước đo đích**, trên chế độ `--all-seats`:

- **Bắt được** (recall): ≥ 90% sự kiện đúng có ít nhất một cảnh báo trên đúng học sinh.
- **Đúng** (precision): ≥ 90% cảnh báo là đúng.
- **Thời gian**: không quá 1,5 lần hiện tại.

Hai con số tính riêng trên tập dev (clip 1, đã chấm) và tập giữ lại (bước 6). Chỉ con số trên tập giữ lại
được ghi là kết quả cuối, vì luật được chỉnh khi nhìn tập dev.

## 2. Hiện trạng (clip 1, `--all-seats`, 23 ô Nguyên đã chấm: 9 đúng, 14 nhầm)

| Cách chạy | Bắt được | Báo nhầm (trên 14) | Đỏ |
|---|---|---|---|
| Kaggle trước đây | 5/9 | | |
| `--all-seats` + `require_sustained` | 8/9 | 3/14 | 2, cả hai đúng |

Tổng 28 cảnh báo thô của `--all-seats`, mình phóng to từng cái và phân loại (chưa được Nguyên xác nhận):

| Nhóm lỗi | Số | Nguyên nhân trong code |
|---|---|---|
| LOOK_AT_LAP trên học sinh đang viết | ~8 | `lap_needs_both_wrists=True`: tay không thấy được coi là "dưới bàn", nên người cúi viết che tay bị bắt |
| Cặp (HEADS_TOGETHER, PASSING) trên học sinh đang viết | ~5 | `HEADS_TOGETHER` chỉ dựa vào hướng đầu; `_hands_meet` chỉ cần hai cổ tay gần nhau |
| PHONE là máy tính cầm tay | 2 | máy tính được nhấc khỏi bàn khi bấm vẫn qua ngưỡng `phone_raise_heights` |
| TURN trên người cúi viết | 1 | điểm mũi và tai rung khi đầu cúi sát |
| Gán nhầm người | ~2 | `assign_poses` chọn đầu gần nhất trong ô ghế, đầu hàng sau lọt vào ô |
| Đúng | 4–5 | |

Tức là độ đúng thô vào khoảng 18%. Luật duy trì đã nâng con số lên, nhưng muốn đạt 90% thì phải sửa
từng nhóm lỗi ở gốc.

## 3. Các cải tiến

Mỗi cải tiến là một cờ trong `ReviewRuleConfig` (mặc định bật sau khi đo xong), để tắt riêng từng cái
khi so sánh trước/sau.

### 3.1 Bước 6: tập chấm rộng hơn (làm trước)

Tập 23 ô quá nhỏ: một cảnh báo đổi kết quả là con số dịch 4–7%.

- Script mới `scripts/make_judging_sheet.py`. Đầu vào là `events.jsonl` của một hay nhiều lần chạy (clip 1
  và clip 2, chạy với **tất cả luật nới ra**: tắt `require_sustained`, hạ ngưỡng, để lấy cả ứng viên yếu).
- Đầu ra là tờ chấm ảnh (mỗi ô một khoảnh khắc, khung đỏ quanh học sinh, số thứ tự) cùng
  `judging.yaml`. Mỗi mục có `id`, `video`, `t_ms`, `seat`, tọa độ đầu, `type` và `verdict: null`.
- Tờ chấm ghi vào `/mnt/project-files/rules-core/` (không vào repo). Khoảng 30 ô, Nguyên chấm mất chừng
  15 phút bằng cách trả lời danh sách số ô đúng.
- Nhãn đã chấm lưu vào `data/labels/<video>-judging.yaml` (chỉ tọa độ và kết luận).
- `scripts/score_alerts.py` gom `anyalert.py` hiện nay thành công cụ chính thức. Nó đọc mọi file nhãn,
  ghép theo khoảng cách đầu ≤ 0,06 và cửa sổ [đầu − 2 s, cuối + 2 s], rồi in bắt được, đúng, và bảng
  theo loại.
- Chia tập: clip 1 là dev; các ô của clip 2 là tập giữ lại. Mình **không** nhìn kết quả tập giữ lại khi
  chỉnh luật, chỉ chạy nó sau mỗi bước để ghi số.

Nếu Nguyên chưa chấm kịp, các bước sau vẫn làm trên tập dev, và báo cáo ghi rõ con số chỉ từ dev.

### 3.2 Bước 1: LOOK_AT_LAP chặt hơn

Học sinh đang viết luôn có một tay trên giấy. Luật mới cho mỗi khung:

- **Bị loại** nếu thấy một cổ tay ở trên đường mép bàn, hoặc một cổ tay nằm trên tờ giấy hay quyển
  sách của ghế đó (dùng lại `_on_paper` cho điểm).
- Tay không thấy được **chỉ** được tính là dưới bàn khi khuỷu tay thấy được và nằm dưới mép bàn (tay
  buông xuống). Thay `lap_needs_both_wrists` bằng `lap_hidden_needs_low_elbow: bool = True`.
- Đầu cúi phải kéo dài `lap_hold_ms` (giữ 3 s), và trong khoảng đó cổ tay không được dịch quá
  `lap_still_widths = 0.15` bề rộng ghế (người viết thì tay di chuyển theo dòng chữ).

### 3.3 Bước 3: điện thoại phải được cầm

- Giữ điều kiện hiện có (gần cổ tay, trên mép bàn một khoảng).
- Thêm: điện thoại phải là vật **được nhấc khỏi giấy**. Nếu khung hình của vật nằm trên tờ giấy của ghế
  trong phần lớn thời gian nó xuất hiện (≥ 60%), đó là máy tính, bỏ. Đây là `calculator_on_paper` hiện
  có nhưng tính theo tỷ lệ thời gian thay vì từng khung, rồi bật mặc định.
- Thêm: tỷ lệ cạnh. Máy tính cầm tay đứng thẳng trong khung thường cao gấp ≥ 1,6 lần rộng nhưng điện
  thoại cầm lên nhìn cũng vậy, nên chỉ dùng tỷ lệ cạnh làm **bằng chứng** ghi vào sự kiện, không làm cổng.
- `PHONE_DETECTED` vẫn được miễn luật "ít nhất 1 s" vì đây là loại hiếm và nặng.

### 3.4 Bước 2: sự kiện cặp cần hai tín hiệu

- **HEADS_TOGETHER**: chỉ giữ khi hai đầu cùng quay về phía nhau (đã có) **và** khoảng cách hai đầu nhỏ
  hơn khi ngồi bình thường ít nhất `heads_closer_ratio = 0.8`, so với mốc "đầu ở nhà" của mỗi ghế.
  Hai người cùng cúi viết thì đầu không xích lại gần nhau.
- **PASSING_OBJECT** loại "hands_only": cần cổ tay một bên **vào hẳn trong ô ghế bên kia** ít nhất
  `passing_inside_frames = 2` khung (hiện chỉ áp cho hàng xa). Loại "object_between" và "transfer" giữ
  như cũ.

### 3.5 Bước 4: gán người theo vai, không theo đầu gần nhất

`assign_poses` mới chấm điểm mỗi người với mỗi ghế:

- Phần chung của khung người với ô ghế, chia cho diện tích khung người (≥ 0,5 mới xét).
- Trung điểm hai vai nằm trong nửa trên của ô ghế.
- Người có đầu ở hàng sau (đầu cao hơn đỉnh ô ghế quá `0.1` chiều cao ô) bị loại dù đầu lọt vào ô.

Giữa các ghế giải bằng tham lam theo điểm từ cao xuống (giống hiện tại, nhưng xét điểm thay vì khoảng
cách). Đổi chữ ký giữ nguyên, nên các nơi gọi không phải sửa.

### 3.6 Bước 5: khung xương ổn định hơn

- **Lật ảnh khi suy luận** (test-time augmentation): chạy pose trên ảnh cắt và bản lật ngang, ghép điểm
  theo cặp trái-phải, lấy trung bình có trọng số độ tin. Bật bằng `--pose-flip`, đặt trong
  `review_offline` (`_pose`). Chi phí: pose chạy gấp đôi, ước chừng chậm thêm 30% toàn bộ.
- **Làm mượt theo thời gian**: bộ lọc One Euro cho từng điểm của từng ghế, trước khi vào luật. Không tốn
  thêm thời gian model.
- Bước này làm cuối và chỉ giữ nếu số đo cải thiện. Nếu chậm quá 1,5 lần mà lợi ít, mặc định tắt và ghi
  lại số đo.

## 4. Cách đo mỗi bước

Một lệnh đo chung, dùng lại dòng thời gian đã có (`--reuse-fast --resume`) nên mỗi lần đo chỉ chạy lại
luật, mất vài phút trên CPU:

```
python scripts/process_video.py <video> --all-seats --reuse-fast --device intel:cpu --out <dir>
python scripts/score_alerts.py <dir>/events.jsonl data/labels/<video>-*.yaml
```

Bước 5 đổi khung xương nên phải dựng lại dòng thời gian; bước này đo trên clip 1 rồi clip 2.

Kết quả mỗi bước ghi thành một dòng trong `docs/superpowers/notes/2026-11-11-rules-core-accuracy.md`:
bước, bắt được, đúng, số đỏ, thời gian, so với dòng trước. Bảng này đi thẳng vào báo cáo BTL.

**Luật dừng**: nếu một bước làm tụt "bắt được" quá 1 sự kiện trên tập dev, sửa ngưỡng hoặc bỏ phần đó
trước khi sang bước sau.

## 5. Kiểm thử

- Mỗi luật mới có test đơn vị với khung giả trong `tests/test_cctv_review_rules.py`, theo kiểu test hiện
  có: một ca bị bắt, một ca người viết không bị bắt.
- `assign_poses`: ca đầu hàng sau lọt vào ô ghế.
- TTA: ghép điểm lật đúng cặp trái-phải; ảnh đối xứng cho ra cùng điểm.
- `score_alerts.py` và `make_judging_sheet.py`: test trên dữ liệu giả nhỏ.
- Toàn bộ bộ test và mypy phải xanh trước mỗi lần đẩy.

## 6. Ngoài phạm vi

- Model ngôn ngữ-hình ảnh (Nguyên đã loại).
- Huấn luyện lại lõi học máy (thiếu dữ liệu, đã đo ở phase 3).
- Tốc độ gần thời gian thực.
- Gộp PR #43: chỉ khi Nguyên yêu cầu.

## 7. Rủi ro

- **Tập chấm vẫn nhỏ.** Cả hai clip cộng lại chỉ khoảng 50 ô. Con số 90% trên chừng ấy ô có sai số
  khoảng ±8%. Báo cáo ghi rõ điều này.
- **Quá khớp clip 1.** Đây là lý do có tập giữ lại từ clip 2.
- **Đích 90% có thể không đạt** với luật thuần. Mình ước chừng đạt 80–90% đúng và 8–9/9 bắt được trên
  clip 1 sau cả 6 bước. Con số này chưa đo.
