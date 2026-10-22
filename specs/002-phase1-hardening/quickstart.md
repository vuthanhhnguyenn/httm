# Quickstart: kiểm chứng Phase 1 hoàn thiện

## 1. Môi trường

```powershell
uv sync --no-default-groups --group dev --extra opencv-python
uv run --no-sync python scripts/download_model.py
uv run --no-sync ruff check .
uv run --no-sync mypy apps/desktop/src packages/proctoring_core/src packages/proctoring_ai/src packages/proctoring_runtime/src scripts
uv run --no-sync pytest
```

Môi trường GPU `.venv-gpu` không chạy `uv sync` (xem README).

## 2. Bộ dữ liệu OEP

1. Tải `OEP_database.tar.gz` (11,8 GB) từ trang chính thức `cvlab.cse.msu.edu/oep-dataset.html` về
   `E:\datasets\oep`, rồi chỉ giải nén video webcam và nhãn:
   `tar xzf OEP_database.tar.gz -C extracted --wildcards "OEP database/*/*1.avi" "OEP database/*/gt.txt"`.
2. Chép `tests/video_samples/oep_set.example.yaml` và `oep_mapping.example.yaml` vào `E:\datasets\oep\`,
   đổi tên bỏ `.example`.
3. Trích dẫn: Y. Atoum, L. Chen, A. X. Liu, S. D. H. Hsu, X. Liu, "Automated Online Exam Proctoring",
   IEEE Transactions on Multimedia, 19(7):1609–1624. Dữ liệu chỉ dùng đánh giá nội bộ, không đưa vào Git.

Mỗi thư mục gồm `gt.txt` (dòng `MMSS MMSS code`), video webcam `<username>1.avi`, wearcam
`<username>2.avi` (không dùng) và âm thanh `.wav` (không dùng).

## 3. Clip tự quay (bổ sung ca OEP không có)

Quay bằng chính webcam sẽ dùng khi thi, 720p, người trong clip đã đồng ý. Mỗi clip 60–120 giây, ghi
nhãn vào CSV `start,end,label` cùng tên.

| Clip | Kiểu bài thi | Kịch bản | Nhãn |
|---|---|---|---|
| `paper-normal` | paper | Làm bài giấy bình thường, cúi viết nhiều | (không nhãn) |
| `computer-normal` | computer | Gõ phím, đọc màn hình, thỉnh thoảng liếc bàn phím | (không nhãn) |
| `phone-use` | computer | Cầm điện thoại 3 lần × 10 giây, có lúc che bằng tay | `PHONE_DETECTED` |
| `hard-negatives` | computer | Cầm bộ bài, ví, điều khiển, sổ nhỏ, mỗi thứ 10 giây | (không nhãn — báo nhầm) |
| `head-gaze` | computer | Quay đầu trái/phải 5 giây; liếc mắt ra ngoài màn hình 5 giây không quay đầu | `HEAD_TURN_*`, `GAZE_*` |
| `talking` | computer | Nói 3 đoạn × 6 giây; ngáp 1 lần; nhai kẹo 10 giây | `POSSIBLE_TALKING` (chỉ đoạn nói) |
| `hands` | paper | Chống cằm 30 giây; giữ tay lên tai 6 giây; viết 30 giây | `SUSPICIOUS_HAND_ACTIVITY` (đoạn tai) |
| `leave-block` | computer | Rời ghế 10 giây; che camera 6 giây; người thứ hai đứng sau 6 giây | `PERSON_MISSING`, `CAMERA_BLOCKED`, `MULTIPLE_PERSON_DETECTED` |

## 4. Chạy đánh giá

```powershell
uv run --no-sync python scripts/evaluate.py --set E:\datasets\oep\oep_set.yaml --save-events
uv run --no-sync python scripts/evaluate.py --set E:\datasets\oep\oep_set.yaml --baseline storage\evaluation\baseline\report.json
```

Kết quả: `storage/evaluation/<thời điểm>/report.md` và `report.json`. Báo cáo mốc của nhánh này lưu ở
`storage/evaluation/baseline/` (ngoài Git) và tóm tắt số liệu chép vào `docs/evaluation-results.md`.

## 5. Kiểm tra thủ công trên ứng dụng

1. `uv run --no-sync python apps/desktop/main.py`; chọn camera và **Thi máy**; bắt đầu.
2. Nhìn thẳng → thấy "Đề xuất hiệu chỉnh"; nhấn chấp nhận; làm theo các điểm nhìn.
3. Liếc mắt ra ngoài màn hình 3 giây không quay đầu → một hàng "Nhìn ra ngoài màn hình (Thử nghiệm)".
4. Nói liên tục 4 giây → "Có thể đang nói (Thử nghiệm)"; ngáp → không có hàng mới.
5. Rời ghế 5 giây → một hàng "Vắng người" với tín hiệu phụ; không có hàng "Không thấy mặt" riêng.
6. Nhấp đúp một sự kiện → thấy ảnh đúng frame và khung bao; bấm "Báo nhầm" → hàng hiển thị đã rà soát.
7. Dừng; bắt đầu lại với **Thi giấy**; cúi viết 30 giây → không có cảnh báo cúi đầu.
