# Sửa độ ổn định và kế hoạch tăng tốc AI

Cập nhật ngày 24/09/2026. Máy được kiểm tra: Quadro M1200 4 GB, driver 528.79;
Python của dự án đang dùng PyTorch `2.14.0+cpu`, `torch.cuda.is_available() = False`.
Ảnh người dùng cung cấp ghi preview 26,2 FPS, AI 5,0 FPS, độ trễ p95 515 ms.
Đây là số liệu trước bản sửa, không phải benchmark sau sửa.

## 1. Các thay đổi đã đưa vào code

- Sửa phép tách ma trận hướng đầu: yaw quanh Y, pitch quanh X, roll quanh Z.
  Code cũ gọi góc quay quanh Z là yaw và gọi góc quanh Y là pitch.
- Loại scale/shear khỏi ma trận; ma trận thiếu/hỏng không còn được coi là đang nhìn thẳng.
- Làm mượt góc theo thời gian, bỏ qua mẫu chất lượng thấp. Có nút **Hiệu chỉnh hướng đầu**:
  nhìn thẳng màn hình, giữ đầu yên khoảng 2 giây để lấy mốc trung tính bằng trung vị.
  Không tự học góc đang quay thành tư thế bình thường. Mốc chỉ tồn tại trong phiên.
- Dùng riêng thời gian quay đầu và cúi đầu; có khoảng chênh 5° giữa ngưỡng bắt đầu và kết thúc.
  `ABNORMAL_HEAD_MOVEMENT` đếm nhiều lượt quay đầu đã kéo dài đủ thời gian trong cửa sổ cấu hình.
- Điện thoại: ghép cùng lớp theo độ chồng khung, cần hai phát hiện >= 0,55 để xác nhận track.
  Sau xác nhận, cho phép tiếp tục với kết quả mới >= 0,35; phải tái xác nhận độ tin cậy cao
  trong 1,5 giây. Event vẫn cần duy trì 1 giây. Hụt tối đa 600 ms không lập tức xóa ứng viên.
  Khung không còn được phát hiện không được dùng để kích hoạt event.
- Áp dụng ngưỡng riêng cho người (0,60). Model vẫn chỉ xét người, điện thoại, sách.
- Overlay có làm mượt vị trí, giữ khung hụt ngắn tối đa 350 ms theo timestamp nguồn và
  giới hạn tổng tuổi kết quả 800 ms. Khung tạm giữ có nét đứt và nhãn riêng.
  Việc giữ khung chỉ ở UI, không đưa dữ liệu cũ vào bộ đánh giá sự kiện.
- Báo tiến trình nạp từng analyzer để phân biệt AI đang khởi tạo với AI đang xử lý.

Các thay đổi xử lý lỗi logic và độ ổn định; chưa có bộ video gán nhãn của webcam này
để khẳng định precision/recall đã đạt một mức cụ thể. Mốc 0,55/0,35 là điểm bắt đầu
cần đánh giá, không phải ngưỡng bảo đảm đúng. Hướng đầu không tương đương hướng mắt.
LEFT/RIGHT hiện theo chiều của ảnh webcam không lật gương.

## 2. Cách dùng và đánh giá bản sửa

Đóng cửa sổ app cũ, chạy lại từ thư mục `E:\httm`:

```powershell
.\.venv\Scripts\python.exe apps/desktop/main.py
```

Nhấn **Bắt đầu giám sát**, đợi analyzer sẵn sàng, nhìn thẳng rồi nhấn
**Hiệu chỉnh hướng đầu**. Giữ đầu yên đến khi thông báo xác nhận hiệu chỉnh xuất hiện.

Đánh giá trên cùng điều kiện trước/sau: 30 giây nhìn thẳng; nghiêng đầu nhưng không quay;
quay trái/phải và giữ 2 giây; cúi/ngẩng; đưa điện thoại vào rồi lấy ra, đổi mặt trước/sau,
che một phần; thử đồ không phải điện thoại như ví, điều khiển, chuột.
Kiểm tra khung vàng “đang xác nhận” khác với event đã được core xác nhận.
Ghi số lần báo nhầm, bỏ sót và thời gian từ khi hành vi bắt đầu đến khi event xuất hiện.

Sau khi người dùng cho phép kiểm thử, 34 test đã qua: hướng đầu, theo dõi điện thoại,
overlay, runtime mất mặt, dựng UI Qt ngoài màn hình, config, event và integration liên quan.
Test mới nằm ở `tests/test_ai_stability.py`. Không mở webcam trong các test này.

## 3. Lộ trình hiệu năng

| Bước | Công việc | Điều kiện chấp nhận |
| --- | --- | --- |
| 1 | Đo baseline CPU sau khi model khởi động xong, cùng video/ảnh | Có p50/p95 mỗi analyzer, FPS toàn pipeline, số timeout; không suy FPS từ tốc độ vẽ khung |
| 2 | Tạo môi trường GPU riêng, thử YOLO FP32 với CUDA phù hợp M1200 | Thực thi inference thành công, health ghi `cuda:0`, không có lỗi kiến trúc CUDA |
| 3 | So sánh YOLO 416/512/640, CPU threads 2/4 và GPU trên cùng đầu vào | Chọn cấu hình theo cả độ trễ và tỷ lệ nhận điện thoại nhỏ; không tăng resolution mặc định trước khi đo |
| 4 | Tách lịch xử lý: YOLO khoảng 10–15 Hz, mặt 10–15 Hz, tay/tư thế 3–5 Hz | Mỗi analyzer có timestamp/FPS riêng; không tính kết quả tái dùng là frame AI mới, không làm rule hiểu “bỏ lượt” là “mất mặt” |
| 5 | Nếu GPU không cải thiện đủ, benchmark ONNX Runtime/OpenVINO CPU | Giữ baseline để so sánh; chỉ chọn backend khi kết quả và thời gian đo tốt hơn |
| 6 | Thu thập/gán nhãn dữ liệu điện thoại và đồ gây nhầm, fine-tune model nhỏ | Chia train/validation/test theo người/phiên; báo precision/recall và false alarms/phút trên test chưa dùng để chỉnh ngưỡng |

Mục tiêu cần đo: preview 25–30 FPS; AI chính hướng tới 10–15 FPS; p95 dưới 200 ms.
Đây là mục tiêu, chưa phải kết quả đã đạt trên M1200. Hiện pipeline chờ đủ các analyzer
trong mỗi lượt; YOLO tăng tốc riêng có thể chưa làm FPS chung tăng tương ứng.
Mục 4 đã có lịch giảm tần suất tư thế/bàn tay và số đo riêng (xem mục 5); chưa có worker
độc lập cho từng analyzer. Mục 5–6 trong bảng vẫn là kế hoạch.

Script đo môi trường (không mở webcam):

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_ai.py
```

Khi có ảnh thử cục bộ chứa cả mặt và điện thoại, có thể đo như sau, đổi đường dẫn ảnh:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_ai.py --image C:\duong-dan\anh-thu.jpg --device cpu --image-size 416 --frames 30 --threads 4
```

Script đo thời gian toàn pipeline và từng analyzer trên ảnh tĩnh lặp lại. Số này không
thay thế benchmark webcam có chuyển động. Trước khi đo, dừng phiên app để tránh tranh CPU/GPU.
Giữ cùng ảnh, số frame, model và điều kiện đo khi so sánh.

## 4. Thử GPU trên Quadro M1200

NVIDIA liệt kê M1200 thuộc compute capability 5.0 (Maxwell). Đã cài PyTorch
`2.7.1+cu118` / torchvision `0.22.1+cu118` vào `.venv-gpu` riêng, không thay `.venv` CPU.
Probe xác nhận `torch.cuda.is_available() == True`, wheel có `sm_50`, và YOLO thực sự
chạy trên `cuda:0`. Driver máy là 528.79; không dùng FP16 mặc định trên card này.

Sau khi sửa một ký tự bị đảo trong SHA-256 của model pose trong manifest (đối chiếu
với file gốc MediaPipe), cả bốn analyzer đều báo `ready`. Test trên **15 frame xám tổng hợp**
ở `image_size=416`, cùng `.venv-gpu` và cùng các model:

| Thiết bị YOLO | Pipeline FPS | p50 | p95 |
| --- | ---: | ---: | ---: |
| Quadro M1200 (`cuda:0`) | 9,57 | 102,5 ms | 117,2 ms |
| CPU | 4,10 | 222,6 ms | 351,3 ms |

Đây là phép đo tải suy luận trên ảnh không có người/điện thoại, không phải FPS webcam
hay kiểm tra độ chính xác. Mục tiêu 15 AI FPS chưa được xác nhận trên máy này.

Các lệnh dưới đã được thực hiện để tạo môi trường GPU. Dùng chúng nếu cần dựng lại
`.venv-gpu`; nên đặt `UV_CACHE_DIR` trên ổ E vì ổ C ít dung lượng:

```powershell
cd E:\httm
$env:UV_CACHE_DIR = "E:\httm\.uv-cache"
$env:UV_PROJECT_ENVIRONMENT = ".venv-gpu"
uv sync --frozen --group desktop --no-default-groups --extra opencv-python
Remove-Item Env:UV_PROJECT_ENVIRONMENT
uv pip install --no-deps --python .venv-gpu/Scripts/python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
Remove-Item Env:UV_CACHE_DIR
.\.venv-gpu\Scripts\python.exe scripts/diagnose_ai.py
.\.venv-gpu\Scripts\python.exe scripts/diagnose_ai.py --synthetic --device 0 --frames 15
```

Kết quả phải báo CUDA khả dụng, GPU M1200 và `object_detector.device: cuda:0`.
Lỗi `no kernel image`, driver hoặc CUDA/cuDNN nghĩa là môi trường GPU chưa sẵn sàng.
Để chạy app với GPU, đóng app/phiên cũ rồi dùng:

```powershell
.\scripts\run_gpu.ps1
```

Không dùng `uv run` mặc định cho môi trường GPU: thao tác đồng bộ theo lockfile hiện tại
có thể thay torch CUDA vừa cài. `run_gpu.ps1` ép `PROCTORING_AI_DEVICE=0` trong phiên app;
không cần thay `configs/default.yaml`, nên có thể quay lại `.venv` CPU như trước.
Trên Task Manager, NVIDIA là GPU 1 vì GPU 0 là Intel; với CUDA nó là `cuda:0`.
Để cài đặt tái tạo dễ hơn về sau, có thể tạo profile phụ thuộc GPU có lockfile riêng.
Chỉ YOLO chuyển sang CUDA theo lựa chọn này; MediaPipe vẫn dùng cấu hình CPU hiện tại.

## 5. Nhịp AI riêng và nhật ký tiếng Việt (24/09/2026)

Không ép MediaPipe sang GPU trên Windows: tài liệu BaseOptions và bản thư viện đang
cài đều nêu GPU delegate Python giới hạn ở Ubuntu. Giữ YOLO CUDA; khuôn mặt vẫn chạy
mỗi lượt, tư thế cách ít nhất 250 ms, bàn tay cách ít nhất 200 ms. Hai khoảng này cấu
hình được trong `detection`; đặt 0 để trở về mọi lượt. Tần suất thực tế phụ thuộc độ
trễ pipeline, không đảm bảo đúng 4/5 FPS.

Lượt không tới lịch có trạng thái `skipped`, không chứa quan sát cũ và không tính là
lượt suy luận mới. Mỗi analyzer có FPS, timestamp quan sát và độ trễ riêng. Lỗi không
bị che bằng trạng thái bỏ lượt; lượt kế tiếp sẽ thử lại. Khung tay/tư thế chỉ được
tạm giữ ở lớp hiển thị tối đa 350 ms, dùng nét đứt và không gửi lại vào bộ luật.

Đo lần này trên M1200, YOLO 416, 60 frame xám/mỗi cấu hình, bỏ 3 lượt khởi động:

| Lịch xử lý | Pipeline FPS | p50 | p95 |
| --- | ---: | ---: | ---: |
| Tất cả analyzer mỗi lượt | 8,54 | 110,1 ms | 147,3 ms |
| Tư thế 250 ms, tay 200 ms | 9,91 | 107,6 ms | 133,0 ms |

Lần đo này tăng khoảng 16% thông lượng. Đây chỉ là ảnh tổng hợp không có người,
không chứng minh độ chính xác hay FPS webcam đạt 15. Pipeline vẫn chờ analyzer
đến lịch chạy trong lượt đó; tách worker độc lập là bước tiếp theo nếu cần giảm
độ trễ hơn nữa, phải ghép timestamp và kiểm tra luật dùng nhiều loại quan sát.

Lệnh đối chiếu (dừng phiên app trước khi đo):

```powershell
.\.venv-gpu\Scripts\python.exe scripts/diagnose_ai.py --synthetic --device 0 --frames 60 --every-frame
.\.venv-gpu\Scripts\python.exe scripts/diagnose_ai.py --synthetic --device 0 --frames 60
```

Nhật ký giao diện đã chuyển tên sự kiện, mức rủi ro, giải thích và giờ hiển thị sang
tiếng Việt/giờ địa phương của máy. JSONL giữ mã sự kiện gốc để không phá định dạng
máy đọc và thêm nội dung tiếng Việt. Thông báo lỗi ứng dụng dùng tiếng Việt; traceback
gốc ghi vào `storage/runtime-errors.log`. Không tắt cảnh báo thư viện ở terminal.

Kiểm chứng: 50 test regression cho lịch AI, FPS, khung tạm giữ, tiếng Việt, core và
tích hợp đã qua. Không mở webcam trong lượt kiểm thử này.

GPU giúp tốc độ tính toán, không tự sửa nhận nhầm điện thoại hay góc đầu. Cần thử trên
video có gán nhãn và hiệu chỉnh hướng đầu khi đổi vị trí ngồi/camera. Tư thế có khung
xương chưa đồng nghĩa mọi hành vi đều có luật đánh giá hoàn chỉnh; ví dụ bộ luật
hiện chưa suy luận nói chuyện từ chuyển động môi trong runtime desktop.

## Tài liệu tham khảo

- [MediaPipe BaseOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions): giới hạn hỗ trợ GPU delegate của Python.

- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python): ma trận biến đổi, VIDEO tracking, timestamp và smoothing một khuôn mặt.
- [MediaPipe Face Geometry](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md): hệ tọa độ và biến đổi từ mặt chuẩn sang mặt quan sát.
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/): kết hợp detection và tracking, các ngưỡng ghép kết quả. Bản sửa dùng bộ ghép IoU nhỏ riêng, không tuyên bố đã tích hợp ByteTrack đầy đủ.
- [Ultralytics predict](https://docs.ultralytics.com/modes/predict/): cấu hình confidence, device và kích thước inference.
- [NVIDIA legacy GPUs](https://developer.nvidia.com/cuda/gpus/legacy): M1200, compute capability 5.0.
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/): cặp torch/torchvision và CUDA 11.8 cho Windows.
- [PyTorch thay đổi hỗ trợ Maxwell](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).
- [CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/): phiên bản driver Windows tương ứng.
