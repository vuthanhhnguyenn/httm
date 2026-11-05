# Phase 2c: Bản đồ ghế + cắt ảnh cho chế độ CCTV — Thiết kế

- Ngày: 2026-10-29 · Trạng thái: đã duyệt (Nguyên chọn hướng B) · Nối tiếp: phase2b CCTV

## Vấn đề (clip EPf9O7UXRoM, bản 2b)
1. Hai dãy cuối gần như không có khung (đầu cao ~40 px) → sót 0:02 và 0:51 cạnh tường.
2. Một người thành hai track (#32/#1) → tự "chụm đầu" với chính mình; ID đổi liên tục.
3. Giám thị đứng yên ở cửa thành thí sinh, đi xuống thì bị `LEAVING_SEAT`.
4. Khung chồng nhau (#25/#2) làm luật cạnh bên/chụm đầu không áp được.

## Thiết kế
- **Cắt ảnh:** mỗi `--tile-every` frame (mặc định 2), chạy thêm YOLO-pose trên 2 ô phủ 60% phía trên
  khung hình (chồng nhau 15%), ghép với kết quả cả khung. Hàng xa được phóng to ~2 lần.
- **Gộp trùng:** hai người có điểm đầu cách nhau < 0.5 bề rộng khung nhỏ hơn là một; giữ khung tin cậy
  cao hơn. Áp cho cả trùng trong một lượt và trùng giữa các ô.
- **Bản đồ ghế (`SeatMapTracker`)**: ghế = vị trí đầu của một người ngồi yên ≥ 3 s (đầu dịch < 0.35 bề
  rộng khung), không đứng (khung cao/rộng < 2.2 và khung xương không đứng). Ghế học suốt video; ghế mới
  không được nằm trong bán kính ghế cũ. Mỗi frame, người được gán vào ghế gần nhất trong bán kính 0.6
  bề rộng ghế (ghép tham lam theo khoảng cách). Người không thuộc ghế nào = **người ngoài** (giám thị):
  vẽ xám, không chạy luật. Mã số ghế cố định suốt video.
- **Rời chỗ:** ghế đã có người ≥ 80% thời gian trong 30 s gần nhất rồi trống liên tục ≥ 5 s.
- **Láng giềng** lấy từ bản đồ ghế (khoảng cách tâm đầu), không từ khung người, nên khung chồng nhau
  không ảnh hưởng; luật "tay sang chỗ bên" chỉ xét ghế cùng hàng (|dy| < 0.5 bề rộng ghế).
- Luật sự kiện giữ nguyên phase 2b, chỉ đổi nguồn vai trò/ghế.

## Kiểm thử
Unit test: gộp trùng, ghép ô về toạ độ khung, học ghế, người ngoài, giám thị đứng yên, rời chỗ, mã ghế
ổn định khi mất/có lại người. Chạy lại clip của Nguyên trên cloud, xem frame ở 0:02, 0:09, 0:43, 0:46, 0:51.
