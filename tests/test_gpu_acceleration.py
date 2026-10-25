from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from proctoring_ai.analyzer_factory import AnalyzerFactory

from proctoring_ai import acceleration


def _fake_mp() -> Any:
    delegate = SimpleNamespace(CPU="CPU", GPU="GPU")
    return SimpleNamespace(tasks=SimpleNamespace(BaseOptions=SimpleNamespace(Delegate=delegate)))


def _fake_torch(monkeypatch, *, cuda: str | None, available: bool, arch: list[str] | None = None,
                capability: tuple[int, int] = (5, 0)) -> None:
    import torch

    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index=0: "Quadro M1200")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=0: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch if arch is not None else ["sm_50", "sm_86"])


def test_auto_device_picks_first_cuda_gpu(monkeypatch, caplog):
    _fake_torch(monkeypatch, cuda="11.8", available=True)
    with caplog.at_level(logging.INFO, logger="proctoring_ai.acceleration"):
        assert acceleration.resolve_torch_device(None) == "0"
        assert acceleration.resolve_torch_device("auto") == "0"
    assert "Quadro M1200" in caplog.text


@pytest.mark.parametrize(("cuda", "available", "arch", "expected"), [
    (None, False, [], "bản CPU"),
    ("11.8", False, [], "không thấy GPU"),
    ("12.8", True, ["sm_75", "sm_86"], "sm_50"),
])
def test_auto_device_falls_back_to_cpu_and_logs_why(monkeypatch, caplog, cuda, available, arch, expected):
    _fake_torch(monkeypatch, cuda=cuda, available=available, arch=arch)
    with caplog.at_level(logging.WARNING, logger="proctoring_ai.acceleration"):
        assert acceleration.resolve_torch_device(None) == "cpu"
    assert "YOLO chạy CPU" in caplog.text
    assert expected in caplog.text


def test_explicit_device_is_not_changed(monkeypatch):
    _fake_torch(monkeypatch, cuda=None, available=False)
    assert acceleration.resolve_torch_device("cpu") == "cpu"
    assert acceleration.resolve_torch_device("0") == "0"


def test_mediapipe_uses_gpu_delegate_when_it_starts(monkeypatch):
    monkeypatch.setattr(acceleration.platform, "system", lambda: "Linux")
    task, device = acceleration.create_mediapipe_task("face", _fake_mp(), lambda delegate: f"task-{delegate}")
    assert (task, device) == ("task-GPU", "gpu")


def test_mediapipe_falls_back_to_cpu_when_gpu_delegate_fails(monkeypatch, caplog):
    monkeypatch.setattr(acceleration.platform, "system", lambda: "Linux")
    calls: list[str] = []

    def create(delegate: str) -> str:
        calls.append(delegate)
        if delegate == "GPU":
            raise OSError("libGLESv2.so.2: cannot open shared object file")
        return "cpu-task"

    with caplog.at_level(logging.WARNING, logger="proctoring_ai.acceleration"):
        task, device = acceleration.create_mediapipe_task("pose", _fake_mp(), create)
    assert (task, device, calls) == ("cpu-task", "cpu", ["GPU", "CPU"])
    assert "libGLESv2" in caplog.text


def test_mediapipe_does_not_try_gpu_on_windows(monkeypatch, caplog):
    monkeypatch.setattr(acceleration.platform, "system", lambda: "Windows")
    calls: list[str] = []
    with caplog.at_level(logging.WARNING, logger="proctoring_ai.acceleration"):
        _task, device = acceleration.create_mediapipe_task("hands", _fake_mp(), lambda d: calls.append(d))
    assert (device, calls) == ("cpu", ["CPU"])
    assert "Windows" in caplog.text


class _FlakyGpuDetector:
    """Stands in for YoloObjectDetector: warmup fails on CUDA, succeeds on CPU."""

    instances: list[_FlakyGpuDetector] = []

    def __init__(self, _path: Any, *, device: str | None, **_kwargs: Any) -> None:
        self.device = device
        self.closed = False
        self.instances.append(self)

    async def initialize(self) -> None:
        if self.device != "cpu":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    @property
    def active_device(self) -> str:
        return str(self.device)

    def close(self) -> None:
        self.closed = True


def _factory_with_fake_detector(monkeypatch, tmp_path, device: str | None) -> AnalyzerFactory:
    from proctoring_ai.detectors import yolo_detector

    _FlakyGpuDetector.instances = []
    monkeypatch.setattr(yolo_detector, "YoloObjectDetector", _FlakyGpuDetector)
    factory = AnalyzerFactory(tmp_path / "manifest.yaml", detector_device=device)
    monkeypatch.setattr(factory, "_resolve_model", lambda *_a, **_k: (
        tmp_path / "yolo.pt", {"class_map": {0: "person"}, "sha256": "0" * 64}))
    return factory


async def test_auto_detector_retries_on_cpu_when_gpu_warmup_fails(monkeypatch, tmp_path, caplog):
    _fake_torch(monkeypatch, cuda="11.8", available=True)
    factory = _factory_with_fake_detector(monkeypatch, tmp_path, None)
    with caplog.at_level(logging.WARNING, logger="proctoring_ai.analyzer_factory"):
        await factory._initialize_detector()
    assert factory.health()["object_detector"] == "ready"
    assert factory.health_details()["object_detector"]["device"] == "cpu"
    assert "chuyển sang CPU" in caplog.text
    # The pipeline must not push the detector back onto the GPU that failed.
    factory.build_pipeline({"detection": {"device": None}})
    assert factory.detector is not None and factory.detector.device == "cpu"


async def test_explicit_gpu_detector_failure_is_reported_not_hidden(monkeypatch, tmp_path):
    factory = _factory_with_fake_detector(monkeypatch, tmp_path, "0")
    await factory._initialize_detector()
    assert factory.health()["object_detector"] == "unavailable"
    assert "no kernel image" in factory.health_details()["object_detector"]["reason"]


def test_landmarker_device_is_reported_in_health(monkeypatch, tmp_path):
    factory = AnalyzerFactory(tmp_path / "manifest.yaml")
    factory.face_analyzer = SimpleNamespace(device="gpu")  # type: ignore[assignment]
    factory._status["face"] = ("ready", "")
    assert factory.health_details()["face"]["device"] == "gpu"


def test_auto_env_value_means_automatic_choice(tmp_path):
    from proctoring_runtime.bootstrap import load_settings

    settings = load_settings(tmp_path, environ={"PROCTORING_AI_DEVICE": "auto"})
    assert settings.device is None
    assert settings.config.detection.device is None
