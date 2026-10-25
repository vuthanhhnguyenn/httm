"""Choose GPU execution for each model where the runtime supports it, CPU otherwise.

Every fallback is logged with its cause so an operator can tell why a model is on CPU.
"""

from __future__ import annotations

import logging
import platform
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("proctoring_ai.acceleration")

AUTO_DEVICE = "auto"


def cuda_status() -> tuple[bool, str]:
    """Return whether PyTorch can run on CUDA here, and a short explanation."""

    try:
        import torch
    except ImportError:
        return False, "PyTorch chưa được cài"
    if torch.version.cuda is None:
        return False, f"PyTorch {torch.__version__} là bản CPU, không có CUDA"
    if not torch.cuda.is_available():
        return False, f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) không thấy GPU NVIDIA nào"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    architectures = torch.cuda.get_arch_list()
    if architectures and f"sm_{major}{minor}" not in architectures:
        return False, (f"{name} (sm_{major}{minor}) không nằm trong các kiến trúc PyTorch "
                       f"{torch.__version__} hỗ trợ: {', '.join(architectures)}")
    return True, f"{name} · CUDA {torch.version.cuda}"


def resolve_torch_device(requested: str | None) -> str:
    """Map ``None``/``"auto"`` to the first CUDA GPU when usable, else ``"cpu"``.

    Any other value is an explicit choice and is returned unchanged.
    """

    if requested not in (None, "", AUTO_DEVICE):
        return str(requested)
    available, detail = cuda_status()
    if available:
        logger.info("YOLO dùng GPU cuda:0 (%s)", detail)
        return "0"
    logger.warning("YOLO chạy CPU vì không dùng được CUDA: %s", detail)
    return "cpu"


def mediapipe_gpu_supported() -> tuple[bool, str]:
    # MediaPipe Tasks for Python only ships a GPU delegate on Linux (OpenGL ES).
    system = platform.system()
    if system == "Linux":
        return True, ""
    return False, f"MediaPipe Tasks Python chưa hỗ trợ GPU trên {system}"


def create_mediapipe_task[T](name: str, mp: Any, create: Callable[[Any], T]) -> tuple[T, str]:
    """Create a MediaPipe task on the GPU delegate when possible, falling back to CPU.

    ``create`` receives a ``BaseOptions.Delegate`` value and returns the task.
    Returns the task and the device it runs on (``"gpu"`` or ``"cpu"``).
    """

    delegate = mp.tasks.BaseOptions.Delegate
    supported, reason = mediapipe_gpu_supported()
    if supported:
        try:
            task = create(delegate.GPU)
        except Exception as exc:
            reason = f"không khởi tạo được GPU delegate ({type(exc).__name__}: {exc})"
        else:
            logger.info("MediaPipe %s dùng GPU", name)
            return task, "gpu"
    logger.warning("MediaPipe %s chạy CPU: %s", name, reason)
    return create(delegate.CPU), "cpu"


__all__ = [
    "AUTO_DEVICE",
    "create_mediapipe_task",
    "cuda_status",
    "mediapipe_gpu_supported",
    "resolve_torch_device",
]
