# Model và giấy phép

Mã nguồn của dự án không tự cấp quyền phân phối cho các model bên thứ ba. Mọi model được đưa vào
`models/` phải có một entry trong `models/manifest.yaml` với nguồn, phiên bản, giấy phép, SHA-256,
class map và metric baseline.

Ultralytics/YOLO là một release gate: trước khi phát hành hoặc phân phối hệ thống, nhóm phải xác
nhận giấy phép AGPL-3.0 tương thích với cách triển khai, hoặc có giấy phép Enterprise hợp lệ. Không
được tải model hoặc video nhạy cảm tự động trong CI. CI chỉ kiểm tra manifest và mã nguồn; binary
model được cung cấp qua kênh đã được phê duyệt và kiểm tra checksum trước khi chạy.

