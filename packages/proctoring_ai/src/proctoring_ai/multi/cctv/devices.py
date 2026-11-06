"""Where each CCTV model runs: CUDA (the NVIDIA card), ``intel:gpu`` (OpenVINO on the Intel iGPU) or CPU.

The Quadro M1200 is shared by the fast tier's pose and the deep tier's pose; the deep tier's object
detector and the fast tier's phone detector can go to the otherwise idle Intel iGPU. Whatever is
missing (no CUDA, no ``openvino`` package, no exported model, no Intel GPU) falls back to the CPU with a
Vietnamese reason for ``timing.json``, never an error.
"""

from __future__ import annotations

import importlib
from pathlib import Path

DEVICES = ("auto", "cuda", "intel:gpu", "cpu")


def openvino_dir(model_path: Path) -> Path:
    """Where ``YOLO(model).export(format="openvino")`` writes its model."""

    return model_path.parent / f"{model_path.stem}_openvino_model"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken: no CUDA
        return False


def _intel_gpu_problem(model_path: Path) -> str:
    try:
        openvino = importlib.import_module("openvino")
    except ImportError:
        return "chưa cài gói openvino"
    if openvino is None:
        return "chưa cài gói openvino"
    if not openvino_dir(model_path).is_dir():
        return f"chưa xuất model OpenVINO (chạy scripts/download_model.py --export-openvino) cho {model_path.name}"
    try:
        available = list(openvino.Core().available_devices)
    except Exception as exc:
        return f"OpenVINO không liệt kê được thiết bị ({exc})"
    if not any(name == "GPU" or str(name).startswith("GPU.") for name in available):
        return "không có card Intel cho OpenVINO"
    return ""


def resolve_device(wanted: str, model_path: Path) -> tuple[str, Path, str]:
    """(device to pass to Ultralytics, model path to load, reason when it is not what was asked)."""

    if wanted == "cpu":
        return "cpu", model_path, ""
    if wanted == "intel:gpu":
        problem = _intel_gpu_problem(model_path)
        return ("intel:gpu", openvino_dir(model_path), "") if not problem else ("cpu", model_path, f"cpu: {problem}")
    if wanted == "cuda":
        return ("cuda:0", model_path, "") if _cuda_available() else ("cpu", model_path, "cpu: không có CUDA")
    if wanted.isdigit() or wanted.startswith("cuda:"):  # an explicit card index
        return (wanted, model_path, "") if _cuda_available() else ("cpu", model_path, "cpu: không có CUDA")
    return ("cuda:0" if _cuda_available() else "cpu"), model_path, ""


__all__ = ["DEVICES", "openvino_dir", "resolve_device"]
