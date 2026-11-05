# Thử YOLOE và yolo11x trên ảnh cắt CCTV (phase 2e, Task 1)

- Ngày: 2026-11-04. Clip `EPf9O7UXRoM.mp4`, 10 ảnh cắt phóng cạnh dài lên 960 px.
- `yolo11x` (COCO, lớp `cell phone`/`book`, conf 0,15) và `yoloe-11l-seg` với lớp chữ
  `phone, paper, notebook, book` (`scripts/export_yoloe_prompts.py`, conf 0,1).

| Ảnh | yolo11x | YOLOE |
|---|---|---|
| 0:52–0:54 chị cạnh tường giơ giấy (3 ảnh) | không thấy gì | không thấy gì |
| 0:10 hàng đầu (3 ảnh) | `cell phone` 0,62 (máy tính bỏ túi), `book` 0,26 | `book` tới 0,77 (tờ bài thi), `phone` 0,39 (cùng máy tính bỏ túi) |
| 0:30 hàng đầu (3 ảnh) | `book` ≤ 0,47, `cell phone` 0,45 | `book` ≤ 0,69, `phone` ≤ 0,35 |
| 0:46 phía trước bên phải | không | `book` 0,40 |

## Quyết định

- **YOLOE: không bật mặc định.** Tiêu chí của spec (thấy tờ giấy giơ lên lúc 0:53) không đạt: ảnh 0:53 quá
  mờ (CCTV nén, phóng 3–4 lần), không model nào thấy. Lớp `paper`/`notebook` không bao giờ ra; giấy trên bàn
  đều ra `book`, giống COCO. YOLOE không thêm gì so với `yolo11x`, lại cần gói `clip` lúc xuất model.
  Mục YOLOE nằm trong `configs/review_models.yaml` ở trạng thái tắt, file xuất không đưa vào manifest.
- Với (d) tài liệu giấu: chỉ còn `LOOK_AT_LAP` và `book` của COCO ở vị trí bất thường (spec mục 6).
- **Báo nhầm cần chú ý:** máy tính bỏ túi trên bàn bị cả hai model gọi là điện thoại (0,39–0,62). Luật
  `PHONE_DETECTED` tầng kỹ đòi gần cổ tay hoặc dưới mép bàn nên vẫn có thể bắt nhầm khi đang bấm máy tính.
- Ảnh vẽ khung: project files `phase2e/spike-yoloe/`.
