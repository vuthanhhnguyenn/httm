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
