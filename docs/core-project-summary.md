# Tóm tắt kiến trúc

Smart Exam Proctoring là ứng dụng desktop chạy cục bộ: một webcam, một thí sinh. AI chỉ tạo tín hiệu cần
rà soát; giám thị kết luận. Tài liệu nguồn: [spec 001](../specs/001-smart-exam-proctoring/spec.md),
[spec 002](../specs/002-phase1-hardening/spec.md), [hiến pháp](../.specify/memory/constitution.md).

## Bốn tầng

| Tầng | Gói | Trách nhiệm | Không được làm |
|---|---|---|---|
| Giao diện | `apps/desktop` | Cửa sổ PySide6, panel, hộp thoại hiệu chỉnh và rà soát; bọc `SessionEngine` trong `QThread` | Chứa luật hay ngưỡng |
| Điều phối | `packages/proctoring_runtime` | Nạp cấu hình, vòng lặp camera → AI → luật, lưu `events.jsonl`/bằng chứng, phát lại video | Phụ thuộc Qt |
| Phân tích | `packages/proctoring_ai` | Camera/video, YOLO, MediaPipe (mặt, tư thế, tay), đặc trưng mắt/miệng, ghép tay, chất lượng ảnh | Quyết định hành vi |
| Nghiệp vụ | `packages/proctoring_core` | Quan sát có kiểu, luật theo thời gian, thứ bậc tín hiệu, hiệu chỉnh (vùng ngồi, hướng mắt), sự kiện, risk, chấm điểm đánh giá | Gọi mô hình, I/O |

## Luồng một khung hình

```text
Frame (webcam hoặc video) ─▶ ObservationPipeline ─▶ ObservationBundle (dict từ analyzer)
   ─▶ Observation.from_bundle (kiểu rõ ràng) ─▶ BehaviorEngine
        ├─ luật: presence, objects (episode), head, safety (vùng ngồi, che camera), gaze, talking, hands
        └─ SignalArbiter: tín hiệu gốc giữ lại tín hiệu hệ quả ─▶ RuleSignal
   ─▶ EventAggregator (một sự kiện mỗi episode) ─▶ RiskEngine (+ trần nhóm tín hiệu hỗ trợ)
   ─▶ CorrelationEngine ─▶ events.jsonl + UI
```

## Cấu hình

`configs/default.yaml` được kiểm tra bằng schema Pydantic (`proctoring_core.config.schema`, khóa lạ bị từ
chối). Mọi ngưỡng nằm ở đây; `exam_modes.paper|computer` ghi đè `behavior` theo kiểu bài thi;
`behavior.hierarchy` khai báo thứ bậc tín hiệu.

## Kiểm chứng

- `pytest`: luật tất định, hợp đồng JSONL, giao diện offscreen, và **golden** (`tests/golden`): kịch bản
  tổng hợp phát qua pipeline và luật thật; mọi thay đổi hành vi phải cập nhật golden có chủ đích.
- `scripts/evaluate.py`: phát lại video có nhãn (OEP + clip tự quay) và báo cáo theo loại sự kiện.
