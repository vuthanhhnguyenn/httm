# Phase 2d: Sửa độ chính xác chế độ CCTV — Thiết kế

- Ngày: 2026-10-29
- Trạng thái: chờ Nguyên duyệt
- Nối tiếp: `2026-10-29-phase2c-cctv-seat-map-design.md`

## 1. Hiện trạng (lần chạy cctv3 mới nhất, M1200, `--mode full --fps 5`)

Tốc độ đạt: 10.3 FPS, 24 ghế, không đổi ID. Độ chính xác chưa đạt:

| Nhãn tay | Kết quả |
|---|---|
| 0:02 hai người sau bên trái trao đổi | Sót |
| 0:43 giám thị đi lại, không được có sự kiện | Đạt một phần: giám thị vẫn gây sự kiện gián tiếp (xem B) |
| 0:46 người trước bên phải quay sang | Đúng (#1, 0:47) |
| 0:51 anh áo đỏ #1 quay đầu | Đúng (#1, 0:47–0:56) |
| 0:51 hai chị cạnh tường phải chụm đầu | Sót |

16 sự kiện mở, 3 đúng (#1 nhìn bạn, #1/#22 chụm đầu), 13 sai.

## 2. Nguyên nhân gốc (xem từng khung hình)

**A. Giám thị đứng yên thành ghế.** Giám thị đứng ở cuối phòng (0:30) bị tạo thành ghế #17 vì đứng yên
3 giây, chân bị bàn và người che nên không thấy đầu gối, khung bị mép trên cắt. Khi đi chỗ khác,
ghế #17 "trống" → 2 lần `LEAVING_SEAT` sai (0:41, 0:56).

**B. Người đi ngang bị gán vào ghế đang có.** 0:42 giám thị đi qua chỗ #6 và được gán làm #6
(khung xanh). Luật chạy trên giám thị như thể là thí sinh #6 → `LOOK_AT_NEIGHBOR` #6 và
`HEADS_TOGETHER` #6/#14 sai (0:44).

**C. Hướng đầu tính cả lúc cúi viết.** Chị #10 cúi viết, đầu nghiêng theo trang giấy, chỉ số
quay đầu lệch −0.46…−0.48 suốt 4–5 giây → `LOOK_AT_NEIGHBOR` sai 2 lần. Khi cúi sát giấy, mũi gần
vai nên chỉ số quay đầu không đáng tin.

**D. Chụm đầu "ăn theo".** Khi một ghế bị báo quay đầu sai (C), khoảng cách đầu với ghế bên cạnh
cũng nhỏ đi nên mở luôn `HEADS_TOGETHER` (#10/#6, #10/#13, #12/#24). Luật chỉ so khoảng cách,
không xét đầu có rời chỗ ngồi để chụm vào nhau thật hay không.

**E. Sót ca thật.**
- 0:02: ghế chỉ tạo sau 3 giây ngồi yên, nên 3 giây đầu clip chưa có ghế nào.
- Hai chị cạnh tường (#8/#20): hai khung chồng lên nhau, đầu sát nhau ngay từ đầu nên "khoảng cách
  bình thường" của cặp đã nhỏ; cần xem dữ liệu từng frame để chắc chắn.

**F. Cách làm việc.** Mỗi vòng sửa hiện nay phải chờ Nguyên chạy trên máy rồi gửi video, và mình
chấm bằng mắt. Không có con số đúng/sai/sót cố định nên sửa chỗ này dễ hỏng chỗ khác.

## 3. Hướng sửa

### 3.1 Bộ chấm điểm tự động (làm trước, giải quyết F)

- File nhãn `data/labels/EPf9O7UXRoM.yaml`: mỗi nhãn có thời điểm bắt đầu/kết thúc, vị trí đầu
  (x, y chuẩn hoá) của người liên quan, loại hành vi (`exchange` = nhìn bạn/chụm đầu/trao đổi,
  `turn_around`, `none` cho giám thị). Mình lập từ nhãn tay của Nguyên và khung hình đã xem;
  Nguyên duyệt lại.
- `scripts/eval_cctv.py events.jsonl labels.yaml`: ghép sự kiện với nhãn theo thời gian (±2 s) và vị
  trí ghế. In bảng đúng / sai / sót và danh sách từng sự kiện sai.
- Ghi thêm `debug.jsonl` (bật bằng `--debug`): mỗi frame mỗi ghế có vị trí đầu, chỉ số quay đầu, mốc,
  vai trò; mỗi cặp có tỉ lệ khoảng cách. Để chẩn đoán bằng số thay vì đoán từ ảnh.
- **Cần Nguyên:** tải file gốc `storage\videos\phong-thi\EPf9O7UXRoM.mp4` (bản 2 phút đã cắt) lên
  project files. Có file gốc, mình chạy và chấm ngay trên cloud, mỗi vòng sửa không cần chờ máy
  của Nguyên; máy M1200 chỉ dùng để đo tốc độ cuối cùng.

### 3.2 Sửa A và B: ai là thí sinh

- Ghế chỉ tạo khi người **ngồi**: khung người phải bị bàn cắt (thấp hơn ghế bên cạnh cùng hàng) và
  không cao quá 1.4 lần chiều cao trung vị của các ghế cùng hàng. Giám thị đứng cao hơn hẳn người
  ngồi cùng hàng nên bị loại kể cả khi không thấy chân.
- Gán người vào ghế đã có: từ chối nếu khung cao hơn 1.5 lần khung lúc ghế được tạo, hoặc người đó
  đến từ một track đang là walker. Người đi ngang giữ nguyên là walker.
- `LEAVING_SEAT`: ghế phải có người ngồi ≥ 10 s **và** lúc ghế trống phải có một người đứng dậy/đi ra
  từ đúng chỗ đó (track rời ghế), không chỉ "ghế trống".

### 3.3 Sửa C: chỉ tính quay đầu khi đầu ngẩng

- Chỉ dùng chỉ số quay đầu khi mũi cao hơn đường vai ít nhất 0.3 độ rộng vai (đầu ngẩng). Lúc cúi
  viết, ghế ở trạng thái "đang cúi", không sinh `LOOK_AT_NEIGHBOR` / `HEAD_TURN_*`.
- Giữ ngưỡng theo độ lắc riêng từng ghế (đã làm).

### 3.4 Sửa D: chụm đầu phải là hai đầu cùng rời chỗ về phía nhau

- `HEADS_TOGETHER` cần cả ba: khoảng cách < 0.7 lần bình thường; ít nhất một đầu dịch khỏi vị trí
  ngồi quen ≥ 0.4 độ rộng khung **về phía người kia**; và đầu đó đang ngẩng (theo 3.3).
- Khác hàng: người hàng trước phải quay lại (đã làm).

### 3.5 Sửa E: sót ca thật

- 3 giây đầu: cho phép tạo ghế sau 1 giây ngồi yên trong 5 giây đầu video (camera vừa bật, mọi
  người đã ngồi sẵn). Sau đó giữ 3 giây như cũ.
- Hai chị cạnh tường: dùng `debug.jsonl` xem cặp này có được ghép không và tỉ lệ khoảng cách ra sao,
  rồi sửa theo số liệu. Nếu khoảng cách bình thường của cặp đã quá nhỏ, dùng thêm mốc "đầu dịch khỏi
  chỗ ngồi" ở 3.4 thay cho tỉ lệ khoảng cách.

## 4. Tiêu chí đạt (trên `EPf9O7UXRoM`, chấm bằng `eval_cctv.py`)

1. Bắt được ≥ 4/5 nhãn tay; 0:43 (giám thị) không sinh sự kiện nào, kể cả gián tiếp.
2. Sự kiện sai ≤ 3 trong cả clip (hiện 13).
3. Không `LEAVING_SEAT` nào khi không ai rời chỗ.
4. Tốc độ trên M1200 vẫn ≥ 8 FPS ở `--mode full --fps 5`.

## 5. Kiểm thử

- Unit test cho từng sửa (giám thị đứng sau bàn không thành ghế, người đi ngang không chiếm ghế,
  cúi viết không thành quay đầu, chụm đầu cần dịch khỏi chỗ, tạo ghế sớm đầu video).
- `eval_cctv.py` chạy trên clip thật sau mỗi sửa, ghi bảng số vào `docs/evaluation-results.md`.
- Chỉ có một clip có nhãn, nên có nguy cơ chỉnh quá khớp. Sau khi đạt, cần thêm ít nhất 1 clip phòng
  thi khác để kiểm tra.

## 6. Thứ tự làm

1. Bộ chấm điểm + `debug.jsonl` + file nhãn (cần video gốc).
2. A + B (ai là thí sinh).
3. C (cúi viết).
4. D (chụm đầu).
5. E (đầu video, cặp cạnh tường).
6. Đo lại trên M1200, ghi kết quả.
