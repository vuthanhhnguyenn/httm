from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from proctoring_ai.multi.cctv import devices
from proctoring_ai.multi.cctv.devices import openvino_dir, resolve_device


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "yolo11x.pt"
    path.write_bytes(b"x")
    return path


def _openvino(monkeypatch, available):
    module = types.ModuleType("openvino")

    class Core:
        available_devices = available

    module.Core = Core
    monkeypatch.setitem(sys.modules, "openvino", module)


def test_openvino_dir_name(model):
    assert openvino_dir(model) == model.parent / "yolo11x_openvino_model"


def test_auto_uses_cuda_when_present(monkeypatch, model):
    monkeypatch.setattr(devices, "_cuda_available", lambda: True)
    assert resolve_device("auto", model) == ("cuda:0", model, "")
    monkeypatch.setattr(devices, "_cuda_available", lambda: False)
    assert resolve_device("auto", model) == ("cpu", model, "")


def test_cuda_missing_falls_back(monkeypatch, model):
    monkeypatch.setattr(devices, "_cuda_available", lambda: False)
    device, path, reason = resolve_device("cuda", model)
    assert (device, path) == ("cpu", model) and "CUDA" in reason


def test_intel_gpu_ready(monkeypatch, model):
    _openvino(monkeypatch, ["CPU", "GPU"])
    openvino_dir(model).mkdir()
    assert resolve_device("intel:gpu", model) == ("intel:gpu", openvino_dir(model), "")


def test_intel_gpu_no_export(monkeypatch, model):
    _openvino(monkeypatch, ["CPU", "GPU"])
    device, path, reason = resolve_device("intel:gpu", model)
    assert (device, path) == ("cpu", model) and "--export-openvino" in reason


def test_intel_gpu_no_card(monkeypatch, model):
    _openvino(monkeypatch, ["CPU"])
    openvino_dir(model).mkdir()
    device, path, reason = resolve_device("intel:gpu", model)
    assert (device, path) == ("cpu", model) and "card Intel" in reason


def test_intel_gpu_no_openvino(monkeypatch, model):
    monkeypatch.setitem(sys.modules, "openvino", None)
    openvino_dir(model).mkdir()
    device, path, reason = resolve_device("intel:gpu", model)
    assert (device, path) == ("cpu", model) and "openvino" in reason


def test_explicit_cpu(model):
    assert resolve_device("cpu", model) == ("cpu", model, "")


def test_yolo_detector_loads_exported_model_after_checking_pt(monkeypatch, model):
    import hashlib

    from proctoring_ai.detectors import yolo_detector

    seen = []
    monkeypatch.setattr(yolo_detector, "YOLO", lambda path, task=None: seen.append((path, task)) or object())
    exported = openvino_dir(model)
    detector = yolo_detector.YoloObjectDetector(model, model_sha256=hashlib.sha256(b"x").hexdigest(),
                                                device="intel:gpu", load_path=exported, task="detect")
    detector._load()
    assert seen == [(str(exported), "detect")]


def test_factory_phone_device_falls_back_with_reason(monkeypatch, tmp_path):
    from proctoring_ai import analyzer_factory

    calls = {}

    class _Detector:
        def __init__(self, path, **kwargs):
            calls.update(kwargs, path=path)
            self.active_device = kwargs["device"]

        async def initialize(self):
            return None

    monkeypatch.setattr("proctoring_ai.detectors.yolo_detector.YoloObjectDetector", _Detector)
    monkeypatch.setattr(devices, "_intel_gpu_problem", lambda _path: "không có card Intel cho OpenVINO")
    factory = analyzer_factory.AnalyzerFactory.__new__(analyzer_factory.AnalyzerFactory)
    factory.__dict__.update(detector_image_size=960, detector_device=None, phone_device="intel:gpu",
                            _status={}, detector=None)
    pt = tmp_path / "yolo11n.pt"
    monkeypatch.setattr(factory, "_resolve_model", lambda *a, **k: (pt, {"class_map": {"67": "cell phone"},
                                                                       "sha256": "0" * 64}), raising=False)
    import asyncio

    asyncio.run(factory._initialize_detector())
    assert calls["device"] == "cpu" and calls.get("load_path") is None
    assert factory._status["object_detector"] == ("ready", "cpu: không có card Intel cho OpenVINO")
