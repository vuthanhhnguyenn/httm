from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from proctoring_ai.multi.cctv.crops import CropTransform
from proctoring_ai.multi.cctv.review_detectors import ModelSpec, ReviewDetectors, load_specs


class _Data:
    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class _Boxes:
    def __init__(self, rows):
        self.data = _Data(rows)


class _Result:
    def __init__(self, names, rows):
        self.names, self.boxes, self.keypoints = names, _Boxes(rows), None


class _Model:
    def __init__(self, names, rows):
        self.names, self.rows = names, rows

    def predict(self, **_kwargs):
        return [_Result(self.names, self.rows)]


def _spec(tmp_path: Path, name: str, class_map, sha: str | None = "auto") -> ModelSpec:
    path = tmp_path / f"{name}.pt"
    path.write_bytes(name.encode())
    digest = hashlib.sha256(name.encode()).hexdigest() if sha == "auto" else sha
    return ModelSpec(name, "objects", path, class_map, 0.2, digest)


def test_missing_or_bad_sha_model_skipped_with_reason(tmp_path: Path) -> None:
    good = _spec(tmp_path, "good", {"cell phone": "phone"})
    bad_sha = _spec(tmp_path, "badsha", {"cell phone": "phone"}, sha="0" * 64)
    missing = ModelSpec("missing", "objects", tmp_path / "nope.pt", {"cell phone": "phone"}, 0.2, None)
    model = _Model({0: "cell phone"}, [[10, 10, 50, 50, 0.9, 0]])
    detectors = ReviewDetectors([good, bad_sha, missing], loader=lambda _path: model)
    assert detectors.status["good"] == "ok"
    assert "SHA" in detectors.status["badsha"] and "không có file" in detectors.status["missing"]
    found = detectors.objects(np.zeros((100, 100, 3), np.uint8), CropTransform(0, 0, 1.0, (100, 100)))
    assert [item.label for item in found] == ["phone"]


def test_class_map_and_cross_model_dedup(tmp_path: Path) -> None:
    coco = _spec(tmp_path, "coco", {"cell phone": "phone"})
    other = _spec(tmp_path, "other", {"phone": "phone"})
    models = {"coco.pt": _Model({0: "cell phone", 1: "cup"}, [[10, 10, 50, 50, 0.6, 0], [60, 60, 90, 90, 0.9, 1]]),
              "other.pt": _Model({0: "phone"}, [[12, 12, 50, 50, 0.8, 0]])}
    detectors = ReviewDetectors([coco, other], loader=lambda path: models[Path(path).name])
    found = detectors.objects(np.zeros((100, 100, 3), np.uint8), CropTransform(0, 0, 1.0, (100, 100)))
    assert len(found) == 1
    assert found[0].label == "phone" and found[0].confidence == 0.8 and found[0].source == "other"


def test_load_specs_skips_disabled(tmp_path: Path) -> None:
    config = tmp_path / "review_models.yaml"
    config.write_text("models:\n"
                      "  - {name: a, kind: pose, path: a.pt, min_conf: 0.25}\n"
                      "  - {name: b, kind: objects, path: b.pt, class_map: {book: book}, enabled: false}\n",
                      encoding="utf-8")
    specs = load_specs(config, tmp_path)
    assert [spec.name for spec in specs] == ["a"] and specs[0].path == tmp_path / "a.pt"


class _BatchModel:
    """Answers one result per image; records each call's batch size and device."""

    def __init__(self, names, rows, *, oom_above: int | None = None, oom_on_cuda: bool = False):
        self.names, self.rows, self.calls = names, rows, []
        self.oom_above, self.oom_on_cuda = oom_above, oom_on_cuda

    def predict(self, **kwargs):
        source = kwargs["source"]
        images = source if isinstance(source, list) else [source]
        device = kwargs.get("device")
        if (self.oom_above is not None and len(images) > self.oom_above) or (self.oom_on_cuda and device != "cpu"):
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        self.calls.append((len(images), device))
        return [_Result(self.names, self.rows) for _ in images]


def _crops(n):
    return [np.zeros((100, 100, 3), np.uint8)] * n, [CropTransform(0, 0, 1.0, (100, 100))] * n


def test_one_predict_per_model_per_batch(tmp_path: Path) -> None:
    model = _BatchModel({0: "cell phone"}, [[10, 10, 50, 50, 0.9, 0]])
    detectors = ReviewDetectors([_spec(tmp_path, "coco", {"cell phone": "phone"})], loader=lambda _p: model)
    found = detectors.objects_many(*_crops(10))
    assert [size for size, _ in model.calls] == [8, 2]
    assert len(found) == 10 and all(len(items) == 1 for items in found)


def test_oom_halves_batch(tmp_path: Path) -> None:
    model = _BatchModel({0: "cell phone"}, [[10, 10, 50, 50, 0.9, 0]], oom_above=2)
    detectors = ReviewDetectors([_spec(tmp_path, "coco", {"cell phone": "phone"})], loader=lambda _p: model)
    found = detectors.objects_many(*_crops(5))
    assert len(found) == 5 and max(size for size, _ in model.calls) <= 2
    assert detectors.status["coco"] == "ok"


def test_oom_at_batch_one_moves_model_to_cpu(tmp_path: Path) -> None:
    model = _BatchModel({0: "cell phone"}, [[10, 10, 50, 50, 0.9, 0]], oom_on_cuda=True)
    detectors = ReviewDetectors([_spec(tmp_path, "coco", {"cell phone": "phone"})], device="cuda:0",
                                loader=lambda _p: model)
    found = detectors.objects_many(*_crops(3))
    assert len(found) == 3 and all(device == "cpu" for _, device in model.calls)
    assert detectors.devices["coco"] == "cpu" and "hết VRAM" in detectors.reasons["coco"]


def test_spec_device_resolved_with_reason(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from proctoring_ai.multi.cctv import review_detectors

    spec = replace(_spec(tmp_path, "coco", {"cell phone": "phone"}), device="intel:gpu")
    loaded = []
    monkeypatch.setattr(review_detectors, "resolve_device", lambda wanted, path: ("cpu", path, "cpu: không có card Intel"))
    detectors = ReviewDetectors([spec], loader=lambda path: loaded.append(path) or _BatchModel({0: "x"}, []))
    assert detectors.status["coco"] == "ok" and detectors.devices["coco"] == "cpu"
    assert detectors.reasons["coco"] == "cpu: không có card Intel" and loaded == [spec.path]


def test_load_specs_reads_device(tmp_path: Path) -> None:
    config = tmp_path / "review_models.yaml"
    config.write_text("models:\n  - {name: a, kind: objects, path: a.pt, class_map: {book: book}, device: intel:gpu}\n"
                      "  - {name: b, kind: pose, path: b.pt}\n", encoding="utf-8")
    assert [spec.device for spec in load_specs(config, tmp_path)] == ["intel:gpu", "auto"]


def test_device_override_beats_config(tmp_path: Path, monkeypatch) -> None:
    from proctoring_ai.multi.cctv import devices

    monkeypatch.setattr(devices, "_cuda_available", lambda: True)
    spec = _spec(tmp_path, "m", {"cell phone": "phone"})
    spec = ModelSpec(spec.name, spec.kind, spec.path, spec.class_map, spec.min_conf, spec.sha256, "intel:gpu")
    model = _Model({0: "cell phone"}, [])
    detectors = ReviewDetectors([spec], loader=lambda _path: model, device_override={"m": "cuda:1"})
    assert detectors.devices["m"] == "cuda:1"
    assert "intel" not in detectors.reasons.get("m", "")


def test_half_only_on_cuda(tmp_path: Path) -> None:
    seen: list[dict] = []

    class _Recorder(_Model):
        def predict(self, **kwargs):
            seen.append(kwargs)
            return super().predict(**kwargs)

    spec = _spec(tmp_path, "m", {"cell phone": "phone"})
    image, transform = np.zeros((100, 100, 3), np.uint8), CropTransform(0, 0, 1.0, (100, 100))
    for device in ("cpu", "0"):
        seen.clear()
        detectors = ReviewDetectors([spec], device=device, loader=lambda _p: _Recorder({0: "cell phone"}, []),
                                    half=True)
        detectors.objects(image, transform)
        assert ("half" in seen[0]) == (device == "0")
        assert seen[0].get("half", True) is True
    seen.clear()
    ReviewDetectors([spec], device="0", loader=lambda _p: _Recorder({0: "cell phone"}, [])).objects(image, transform)
    assert "half" not in seen[0]


def test_kinds_of_loaded_models_only(tmp_path: Path) -> None:
    objects = _spec(tmp_path, "coco", {"cell phone": "phone"})
    pose = _spec(tmp_path, "pose", {})
    pose = ModelSpec(pose.name, "pose", pose.path, {}, 0.2, pose.sha256)
    missing = ModelSpec("missing", "objects", tmp_path / "nope.pt", {}, 0.2, None)
    detectors = ReviewDetectors([objects, pose, missing], loader=lambda _path: _Model({}, []))
    assert detectors.kinds == {"coco": "objects", "pose": "pose"}
