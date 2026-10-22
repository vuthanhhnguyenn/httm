# Quyền riêng tư và dữ liệu phiên thi

Tài liệu này mô tả **sản phẩm làm gì với dữ liệu**, **mặc định an toàn**, và **những quyết định tổ chức
phải chốt trước khi pilot**. Đây là tài liệu kỹ thuật, không thay thế tư vấn pháp lý.

## 1. Dữ liệu được tạo ra

| Dữ liệu | Nơi lưu | Mặc định |
|---|---|---|
| Sự kiện (`events.jsonl`): loại, thời điểm, mức độ, độ tin cậy, điểm rủi ro | `storage/sessions/<id>/` | Luôn bật |
| `session.json`: cấu hình, policy, phiên bản model, hồ sơ hiệu chỉnh | `storage/sessions/<id>/` | Luôn bật |
| Ảnh bằng chứng JPEG | `storage/sessions/<id>/evidence/` | **Tắt** (`evidence_frame_enabled: false`) |
| Clip video bằng chứng | `storage/sessions/<id>/evidence/` | **Tắt** (`evidence_clip_enabled: false`) |
| Video camera thô | Không ghi ra đĩa | Không bao giờ |
| Dữ liệu gửi ra ngoài máy | Không có | Không có |

Sản phẩm chạy cục bộ: hình ảnh camera được xử lý trong bộ nhớ và không được tải lên máy chủ.

## 2. Mặc định an toàn

- Ảnh/clip bằng chứng **tắt** cho tới khi tổ chức bật rõ ràng trong `configs/default.yaml`.
- Tín hiệu thử nghiệm (hướng mắt, nói chuyện) **tắt** cho tới khi có báo cáo holdout đạt cổng ở
  [evaluation-methodology.md](evaluation-methodology.md).
- Không có telemetry, không có tài khoản đám mây, không có upload.
- Mọi tệp nằm dưới `storage/` (ngoài Git theo `.gitignore`).

## 3. Lưu trữ và xoá

Hiện tại:

- `evidence.retention_days: 30` trong `configs/default.yaml`; policy có thể siết chặt hơn
  (`min(evidence.retention_days, policy.retention_days)`).
- `EvidenceRecorder.purge_expired()` chạy khi **bắt đầu phiên mới**, xoá ảnh/clip quá hạn.
- Xoá thủ công: xoá thư mục `storage/sessions/<session_id>/`. Không có bản sao ở nơi khác.

Hạn chế đã biết:

- Chưa có nút "xoá phiên" trong UI; hiện phải xoá thư mục.
- Chưa có export có kiểm soát (ví dụ xuất gói bàn giao cho hội đồng).
- Purge chỉ chạy khi mở phiên mới, không chạy theo lịch nền.

## 4. Trước khi pilot: quyết định của tổ chức

Phải chốt bằng văn bản và ghi vào policy của cơ sở:

1. **Căn cứ pháp lý và sự đồng ý**: ai đồng ý, đồng ý bằng hình thức nào, có thể rút lại ra sao.
2. **Thời hạn lưu**: ảnh bằng chứng, sự kiện, báo cáo. Đề xuất: ảnh ≤ 30 ngày, sự kiện ≤ 1 năm học.
3. **Phạm vi truy cập**: ai được xem ảnh bằng chứng, có nhật ký truy cập không, có cần mã hoá ổ đĩa.
4. **Quy trình yêu cầu của thí sinh**: xem, xuất, xoá, khiếu nại; thời hạn phản hồi.
5. **Xử lý sự cố**: máy bị mất/mang đi sửa, ổ đĩa hỏng, sao lưu.
6. **Ca thi giấy và thiết bị cá nhân**: có cho phép không, có ghi hình không.
7. **Thông báo cho thí sinh**: nội dung hiển thị trước khi bắt đầu phiên (camera đang phân tích gì, lưu gì).

## 5. Yêu cầu tối thiểu trước khi dùng thật

- [ ] Có văn bản đồng ý hiển thị trước khi phiên bắt đầu.
- [ ] Chốt thời hạn lưu và bật purge theo lịch (không chỉ khi mở phiên mới).
- [ ] Có chức năng xoá/export một phiên từ UI.
- [ ] Giới hạn quyền truy cập thư mục `storage/` ở mức hệ điều hành.
- [ ] Ghi nhật ký ai đã xem/sửa bằng chứng.
- [ ] Diễn tập quy trình khiếu nại: thí sinh phủ nhận sự kiện, người rà soát xem bằng chứng.

Cho tới khi hoàn tất danh sách trên, sản phẩm chỉ nên dùng **thử nghiệm nội bộ có giám sát**, với người
tham gia đã biết rõ và đồng ý.
