"""Vietnamese presentation text; stable event codes remain unchanged in storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

EVENT_NAMES = {
    "HEAD_TURN_LEFT": "Quay đầu sang trái",
    "HEAD_TURN_RIGHT": "Quay đầu sang phải",
    "LOOK_DOWN": "Cúi đầu nhìn xuống",
    "ABNORMAL_HEAD_MOVEMENT": "Quay đầu nhiều lần",
    "PHONE_DETECTED": "Phát hiện điện thoại",
    "DOCUMENT_DETECTED": "Phát hiện sách/tài liệu",
    "MULTIPLE_PERSON_DETECTED": "Có nhiều người trong khung hình",
    "PERSON_MISSING": "Không thấy người trong khung hình",
    "LEAVING_SEAT": "Có dấu hiệu rời chỗ ngồi",
    "CAMERA_BLOCKED": "Camera có thể bị che",
    "FACE_NOT_VISIBLE": "Không thấy rõ khuôn mặt",
    "SUSPICIOUS_HAND_ACTIVITY": "Cử động tay cần rà soát",
    "POSSIBLE_TALKING": "Có dấu hiệu nói chuyện",
}

CORRELATION_NAMES = {
    "CORRELATION_PHONE_MULTIPLE_PERSON": "Điện thoại và nhiều người cùng xuất hiện",
    "CORRELATION_HEAD_AND_HAND": "Quay đầu kèm hoạt động tay",
    "CORRELATION_LOOK_DOWN_AND_HAND": "Cúi đầu kèm hoạt động tay",
}


def event_name(code: str) -> str:
    return EVENT_NAMES.get(code, CORRELATION_NAMES.get(code, "Tín hiệu cần rà soát"))


def level_name(code: str) -> str:
    return {"NORMAL": "Bình thường", "LOW": "Thấp", "MEDIUM": "Trung bình",
            "HIGH": "Cao", "CRITICAL": "Rất cao"}.get(code, "Chưa xác định")


def event_description(item: dict[str, Any]) -> str:
    confidence = max(0.0, min(1.0, float(item.get("confidence", 0))))
    text = f"{event_name(str(item.get('type', '')))}. Độ tin cậy tín hiệu: {confidence:.0%}."
    if item.get("action") == "correlation":
        return text + " Hai sự kiện đã xác nhận xuất hiện gần nhau. Giám thị cần đối chiếu ảnh và bối cảnh trước khi kết luận."
    duration = item.get("metrics", {}).get("duration_ms")
    if isinstance(duration, (float, int)):
        text += f" Thời gian ghi nhận: {duration / 1000:.1f} giây."
    if item.get("type") == "DOCUMENT_DETECTED":
        text += " Model hiện nhận diện sách; chưa nhận diện tin cậy mọi loại giấy/tài liệu."
    if item.get("metrics", {}).get("fast_confirmation"):
        text += " Xác nhận nhanh qua nhiều khung hình có độ tin cậy cao."
    if str(item.get("type", "")).startswith("HEAD_TURN"):
        text += " Góc đầu được so với tư thế nhìn thẳng đã hiệu chỉnh; không phải phép đo hướng mắt."
    if any(str(code).startswith("CORRELATION_") for code in item.get("risk_reason_codes", [])):
        text += " Điểm tăng thêm vì nhiều tín hiệu xuất hiện gần nhau; cần giám thị đối chiếu ảnh và bối cảnh."
    return text + " Đây là tín hiệu hỗ trợ; giám thị cần kiểm tra trước khi kết luận."


def local_clock(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return "--:--:--"


def health_reason(reason: str) -> str:
    if not reason:
        return ""
    if "SHA-256" in reason:
        return "Tệp model không khớp mã kiểm tra; cần kiểm tra lại model."
    if "MODEL_MISSING" in reason or "FileNotFoundError" in reason or "NOT_CONFIGURED" in reason:
        return "Chưa tìm thấy model hoặc đường dẫn cấu hình chưa đúng."
    if "TIMEOUT" in reason:
        return "Bộ phân tích xử lý quá thời gian cho phép."
    if reason == "DISABLED":
        return "Bộ phân tích chưa được bật hoặc chưa nạp thành công."
    if reason == "SCHEDULED_INTERVAL":
        return "Chờ lượt xử lý kế tiếp để giảm tải CPU; không phải mất đối tượng."
    if "INFERENCE_ALREADY_RUNNING" in reason:
        return "Lượt trước chưa xong; bỏ khung mới để tránh dồn hàng đợi."
    return "Không xử lý được; cần kiểm tra cấu hình model và môi trường AI."
