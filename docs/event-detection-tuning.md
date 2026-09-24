# Cảnh báo điện thoại và hướng đầu

Bản chỉnh này tập trung vào quyết định tạo sự kiện, không thay model YOLO và không
coi tăng FPS là tăng độ chính xác. Các ngưỡng bên dưới là lựa chọn kỹ thuật ban đầu,
cần đánh giá trên video thực tế trước khi dùng để quyết định về thí sinh.

## Điện thoại: hai đường xác nhận

- Bình thường: giữ ngưỡng nhận diện 0,55; tracker cần ít nhất hai quan sát đủ mạnh.
  Bộ đếm thời gian bắt đầu từ lần đầu có track tạm, nhưng chỉ cảnh báo sau khi
  track đã xác nhận và quan sát đủ 0,6 giây. Track đã xác nhận có thể tiếp tục ở mức 0,35.
- Nhanh: cùng một track đã xác nhận xuất hiện trong ít nhất hai lượt suy luận mới,
  mỗi lượt có confidence từ 0,85, kéo dài tối thiểu 200 ms. Khoảng cách giữa hai
  mẫu không vượt quá 350 ms; mẫu yếu hoặc mất đối tượng làm lại xác nhận nhanh.
- Hai đường tạo chung một loại sự kiện, không cộng hai cảnh báo cho cùng tín hiệu.
  Nhánh nhanh đã mở được giữ khi track còn hợp lệ, không đóng/mở chỉ vì confidence
  giảm dưới 0,85. Mất đối tượng quá 600 ms sẽ đóng theo luật hiện có.
- Điện thoại được phép trong chính sách không sinh cảnh báo ở cả hai đường.

Ở chuỗi test 5 FPS với confidence 0,90, tracker xác nhận tại 200 ms, nhánh nhanh
mở tại 400 ms tính từ lần YOLO thấy điện thoại đầu tiên. Đây không phải cam kết
400 ms tính từ lúc đưa điện thoại vào webcam: còn thời gian camera/suy luận và
có thể YOLO chưa thấy vật thể. Một khung đơn lẻ không đủ tạo cảnh báo nhanh.

Các tham số nằm trong `behavior.phone`: `fast_confidence`, `fast_duration_seconds`,
`minimum_duration_seconds`. Không hạ đồng loạt confidence chỉ để thấy nhiều khung.
Nếu không có khung điện thoại, thay đổi luật thời gian không thể khắc phục YOLO
bỏ sót; cần dữ liệu điện thoại nhỏ, che khuất và đồ vật dễ nhầm để đánh giá/fine-tune.

## Quay đầu: dùng mốc nhìn thẳng và diễn biến theo thời gian

1. Giám thị nhấn **Hiệu chỉnh hướng đầu** khi thí sinh nhìn giữa màn hình, giữ yên
   khoảng 2 giây. Chưa hiệu chỉnh hoặc hiệu chỉnh thất bại thì không tạo sự kiện
   quay đầu/cúi đầu. Phát hiện điện thoại và các luật khác vẫn hoạt động.
2. Ma trận MediaPipe được chuyển thành yaw/pitch/roll. Bộ lọc trung vị ba mẫu loại
   góc nhảy đơn lẻ, sau đó làm mượt theo thời gian. Lọc tạo thêm độ trễ khoảng một
   lượt ở chuyển động rõ; không tăng độ phân giải hay thêm model nặng.
3. Lệch ngang ít nhất 22 độ so với mốc trong 0,3 giây tạo sự kiện quay đầu.
   Nếu lệch mạnh ít nhất 32 độ, đường xác nhận nhanh dùng 0,2 giây. Bộ lọc góc
   và nhịp AI vẫn tạo thêm độ trễ. Ngưỡng thoát thấp hơn 5 độ để tránh đóng/mở
   liên tục ngay biên. Cúi đầu dùng ngưỡng pitch 15 độ trong 0,4 giây; cúi sâu
   từ 28 độ dùng xác nhận nhanh 0,2 giây. Hai nhánh cùng tạo một sự kiện
   LOOK_DOWN, kể cả khi đồng thời quay ngang, không cần nhìn thấy tài liệu.
   Đây là dấu hiệu cúi nhìn xuống cần xem lại, chưa xác định được vị trí ngăn bàn
   hoặc hành vi chép bài. Roll
   không được tính thành quay trái/phải.
4. Bộ đếm “quay đầu nhiều lần” chỉ được đếm lượt tiếp theo sau khi thấy đầu trở về
   vùng giữa ±12 độ liên tục ít nhất 300 ms. Mất mặt hoặc dao động quanh ngưỡng
   không được coi là một lượt quay mới. Vẫn có thể cảnh báo một lần quay kéo dài.
5. Mất mặt, chất lượng thấp hoặc gián đoạn quan sát không được dùng để hoàn tất
   xác nhận. Khi không còn thấy mặt, cảnh báo riêng có thể mở sau 1 giây quan sát
   hợp lệ. Hiệu chỉnh cũng không được nối mẫu qua gián đoạn hơn 600 ms.

Trái/phải theo ảnh chưa lật gương. Đây là hướng đầu, không phải hướng mắt hay
bằng chứng chắc chắn người dùng đang xem tài liệu. Chỉ số chất lượng mặt hiện dùng
kích thước mặt và tính hợp lệ ma trận, chưa phải xác suất sai số góc đã hiệu chuẩn.

## Thử thực tế

Chạy `scripts/run_gpu.ps1`, bắt đầu phiên rồi hiệu chỉnh đầu. Thử riêng từng cảnh:

| Tình huống | Kỳ vọng |
| --- | --- |
| Điện thoại rõ, giữ trước camera 1–2 giây | Có khung; confidence cao lặp lại sẽ dùng xác nhận nhanh |
| Điện thoại nhỏ/xa, bị che | Ghi nhận việc bỏ sót; không coi việc chưa cảnh báo là đạt |
| Cốc, điều khiển, ví cầm cạnh mặt | Không cảnh báo điện thoại; cần thống kê các lần nhầm |
| Nhìn thẳng, nghiêng đầu, dịch người nhẹ | Không sinh sự kiện quay trái/phải chỉ vì roll |
| Quay rõ sang một bên, giữ khoảng 1–2 giây | Có một sự kiện quay đầu sau ngưỡng xác nhận |
| Cúi sâu nhìn xuống dưới bàn hoặc vừa quay vừa cúi | Có sự kiện cúi đầu khi góc mặt vẫn đo được |
| Cúi quá sâu/quay ra sau làm mất mặt | Dùng cảnh báo mất mặt; không suy đoán góc từ mẫu mất tracking |
| Lướt đầu nhanh rồi về giữa | Không thành sự kiện kéo dài |
| Giữ lệch lâu, mất mặt ngắn rồi thấy lại | Không tự tăng số lần quay đầu do mất tracking |

Nên đo ít nhất 20 lần đưa điện thoại lên và 20 lần quay mỗi hướng; ghi tỷ lệ bỏ sót,
báo nhầm và độ trễ từ hành động tới sự kiện. Thêm vài phút nhìn thẳng/làm bài bình
thường để đếm báo nhầm/phút. Chưa có tập video gán nhãn để xác nhận chất lượng thực tế.
Ứng dụng không tự ghi hình trong bước thử này.

Lưu ý kiểm tra cấu hình: `configs/default.yaml` và `configs/exam_policy.yaml`
đều phải qua schema. Chính sách vùng tài liệu cho phép chưa được hỗ trợ; không thêm
`allowed_document_zones` vì app hiện chưa thực thi khóa này.

## Nguồn và phạm vi

- [MediaPipe Face Landmarker Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python): ma trận hướng mặt, tracking ở chế độ video, làm mượt khi dùng một mặt.
- [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/): ý nghĩa confidence, kích thước đầu vào và thiết bị suy luận.

Hai nguồn mô tả đầu ra/model, không xác nhận các ngưỡng hành vi là chuẩn chống gian
lận. Bộ luật ở đây là heuristic có regression test; quyết định cuối cùng cần giám thị.
