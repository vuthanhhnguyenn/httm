# Lõi luật v2: gán người theo dõi, góc đầu HHP-Net, bảng chọn loại — Thiết kế

- Ngày: 2026-11-11
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-11-11-rules-core-accuracy-design.md` (PR #43, chưa gộp)
- Nguồn tham khảo: https://claude.ai/artifact/Jvg8oTbFSkGmB8bGS43ndN

## 1. Mục tiêu và điều đã chốt

- Giữ hướng hiện tại: luật có nguồn, chạy CPU, **không** dùng model ngôn ngữ-hình ảnh. Hệ thống chỉ gợi ý cho giám thị.
- Đích: cảnh báo đúng **loại** (chụm đầu, nhìn sang, trao bài, nhìn đùi), không chỉ "có cảnh báo trên đúng người". Đo bằng clip 1 (dev) và clip 2 (giữ lại).
- Thời gian không quá 1,5 lần hiện tại (hiện: dựng timeline 100 phút CPU cho clip 1, luật chạy lại 30 s).
- Dữ liệu học sinh vẫn là dữ liệu cá nhân: ảnh, video, crop không vào repo; nhãn chỉ có toạ độ và kết luận.

## 2. Hiện trạng và gốc lỗi (đã đo, clip 1, 14 nhãn đúng, 27 nhãn sai)

Bắt được 10/14, đúng 5/13 đã chấm, đúng loại 4. Gốc lỗi theo mức ảnh hưởng:

1. **Gán người vào ghế sai theo từng khung.** `assign_poses` chọn người cho ghế trong từng khung riêng rẽ. Khi khung xương hàng trước mất, đầu rơi về đỉnh ô ghế, sát người cúi ở hàng sau: giả "chụm đầu" (giây 42, 44, 68). Cùng lúc, một người phụ nữ bị gán sang ô ghế khác, nên các sự kiện **đúng** ở giây 25–38 chỉ bắt được nhờ lỗi này. Vì vậy luật cùng hàng (5b) làm bắt tụt 10 → 6, và thêm điều kiện "thấy mặt" không đổi gì (đã thử).
2. **Hướng đầu chỉ dựa vào mũi so với vai.** Bỏ sót quay ngang ở giây 24 (yaw vượt ngưỡng 0,6 s, giữ 1,5 s) và 44 (độ lệch 0,24–0,27, ngưỡng 0,35). HHP-Net với ngưỡng 25–30° và hold 0,8 s bắt thêm giây 24 (4/5 so với 3/5).
3. **Loại sự kiện quyết theo luật nào bắn trước**, không theo bằng chứng: trao bài (giây 60–65) bị báo thành nhìn đùi; nhìn sang bị báo thành chụm đầu.

## 3. Giải pháp (theo thứ tự, mỗi bước có cổng)

### P0. Lưu người thô trong cache (nền cho mọi bước sau)

`timeline.json.gz` hiện chỉ giữ khung xương **đã gán ghế**, nên đổi cách gán là phải dựng lại 100 phút. Đổi thành lưu thêm **mọi người thô** mỗi khung (box + 17 điểm, đã quy về toạ độ khung hình). Dựng lại clip 1 và clip 2 **một lần** (~170 phút CPU, chia 3 ghế/lượt, có `--resume`); sau đó mọi thử nghiệm gán và luật chạy lại trong vài phút.

Cổng: phát lại từ người thô bằng cách gán cũ (`assign_by_body=False`) ra đúng kết quả hiện tại (10/14, 14 cảnh báo, sai số dựng < 0,02).

### P1. Gán người theo dõi (sửa gốc lỗi 1)

Mới `person_tracks.py`: nối người giữa các khung bằng IoU của box kết hợp khoảng cách vai (kiểu ByteTrack đơn giản, offline nên nối hai chiều được). Chủ ghế = track có phần chung với ô ghế lớn nhất trong cửa sổ trượt 3 s, đổi chủ chỉ khi track mới hơn hẳn trong ≥ 1,5 s (trễ để khỏi nhảy). Khung mất xương: ghế **không có** khung xương (không điền đỉnh ô), nên cặp chụm đầu không dùng đầu giả. Tham chiếu: SparseTrack (độ sâu giả từ đáy box để tách hàng).

Cổng: bắt được ≥ 10/14 trên clip 1 với luật cùng hàng bật (5b) và tái hiện các sự kiện đúng giây 25–38 bằng đúng người.

### P2. Góc đầu HHP-Net (sửa gốc lỗi 2)

HHP-Net (MIT, 0,4 MB, 5 điểm mặt của khung xương sẵn có, trả yaw/pitch/roll). Chuyển SavedModel sang ONNX (tf2onnx) để chạy bằng OpenVINO/onnxruntime trong `.venv`, không kéo TensorFlow vào repo. Mốc yaw riêng từng ghế (trung vị cửa sổ trượt như `RollingBaseline` hiện có). Luật quay: |yaw − mốc| > θ trong ≥ 0,8 s **và** chỉ số mũi–vai cùng chiều (hai tín hiệu); pitch chỉ dùng để loại cúi sâu. Độ bất định dùng làm trọng số, **không** làm cổng cứng (thử cổng cứng u ≤ 2 mất 3/4 sự kiện đúng). Nhóm tín hiệu quét θ ∈ {20, 25, 30} và hold ∈ {0,8; 1,2}, chọn trên clip 1, báo cáo trên clip 2.

Cổng: bắt các lần quay giây 24, 27, 44 mà không làm số cảnh báo nhầm tăng quá 20%.

### P3. Bảng chọn loại (sửa gốc lỗi 3)

Mỗi cửa sổ ứng viên sinh một tập bằng chứng; loại được chọn theo thứ tự ưu tiên **cố định, có thể giải thích**:

1. PHONE (vật được cầm, đã có)
2. PASSING: cổ tay một người vào hẳn ô ghế người kia ≥ 2 khung, hoặc vật nằm giữa hai cổ tay (đã có); loại này **thắng** nhìn đùi vì hai người ngồi sát cùng cúi (giây 60–65).
3. HEADS_TOGETHER: **cùng hàng**, hai đầu cùng quay vào nhau, khoảng cách < 0,8 × mốc (đã đo).
4. LOOK_AT_NEIGHBOR: yaw hướng về một ghế có người, kể cả khác hàng.
5. LOOK_AT_LAP: cúi + tay xuống dưới (luật forearm-drop đã có) và không có bằng chứng 2–4.
6. TURN_AROUND: thấy gáy.

Nguồn: vector đặc trưng khớp/khoảng cách giữa hai người (Lin 2021; EGCN-IIG), ở đây dùng dạng luật thay vì mạng vì chưa đủ nhãn để học.

Cổng: số "đúng loại" trên clip 1 tăng so với 4 hiện tại, không tụt "bắt được".

### P4. Đo và nhãn

- Thước chính: bắt được / đúng / **đúng loại** (`score_alerts.py`, đã có).
- Tập dev: clip 1 (nhãn từ 4 file). Tập giữ lại: clip 2, **cần Nguyên chấm 6 ô** (tờ `rules-core/clip2/sheet-1.jpg`). Chỉ số cuối cùng là số trên clip 2; mình không nhìn clip 2 khi chỉnh luật.
- Sau mỗi bước ghi một dòng vào `docs/superpowers/notes/2026-11-11-rules-core-accuracy.md` (số liệu cho báo cáo BTL).

### P5. Sau cùng

- Thử mấp máy miệng (MediaPipe, hàng gần) như Nguyên đã xin.
- Nếu clip 2 ≥ 80% đúng loại: tổng hợp báo cáo. Nếu còn dưới 80%: đề xuất mở rộng nhãn hoặc huấn luyện (hướng B trong tài liệu tham khảo).

## 4. Luật dừng

Một bước làm tụt "bắt được" quá 1 sự kiện trên clip 1 thì chỉnh hoặc bỏ phần đó trước khi sang bước sau; mặc định của luật mới chỉ bật khi qua cổng. Mỗi bước là một cờ trong `ReviewRuleConfig` để so trước/sau.

## 5. Kiểm thử

Mỗi bước có test đơn vị với dữ liệu giả (một ca bắt đúng, một ca không bị bắt): theo dõi người (người đổi chỗ, mất xương), chủ ghế có trễ, yaw ↔ ONNX khớp SavedModel (sai số < 0,5°), bảng chọn loại từng hàng ưu tiên. Toàn bộ pytest, ruff, mypy xanh trước mỗi lần đẩy.

## 6. Rủi ro và con số kỳ vọng

- **Tập chấm nhỏ**: clip 1 có 14 nhãn đúng, clip 2 sau khi chấm cũng vài mục; sai số ±8–10%. Báo cáo ghi rõ.
- **Trọng số HHP-Net học trên mặt gần** (BIWI, 300W-LP); mặt hàng xa có thể kém. Giảm rủi ro bằng cách kết hợp với chỉ số mũi–vai.
- **P1 có thể không đủ** nếu khung xương hàng trước mất thật sự (không phải gán sai).
- **Kỳ vọng (chưa đo)**: sau P1–P3, đúng loại trên clip 1 khoảng 60–75%, bắt được 11–12/14. Đích 90% có thể **không đạt** với luật thuần; nếu vậy kết luận hợp lý cho báo cáo là cần dữ liệu có nhãn lớn hơn (các bài báo đạt >90% đều dùng 11–17 nghìn ảnh có nhãn).
- **Thời gian**: P0 mất ~3 giờ CPU một lần; từ P1 trở đi mỗi lần đo vài phút.
