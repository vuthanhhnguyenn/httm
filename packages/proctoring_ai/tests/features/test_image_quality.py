from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from proctoring_ai.features.image_quality import ImageQualityExtractor  # noqa: E402


def _scene(seed: int) -> object:
    rng = np.random.default_rng(seed)
    image = np.full((72, 128, 3), 110, dtype=np.uint8)
    image[20:50, 40:80] = 200  # an object with edges
    noise = rng.integers(-3, 4, size=image.shape)
    return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)


def test_features_separate_covered_lens_from_scene() -> None:
    extractor = ImageQualityExtractor()
    scene = extractor.extract(_scene(1))
    covered = ImageQualityExtractor().extract(np.full((72, 128, 3), 8, dtype=np.uint8))

    assert scene["edge_density"] > covered["edge_density"]
    assert scene["contrast"] > covered["contrast"]
    assert covered["brightness"] < 0.06


def test_only_exactly_repeated_frames_count_as_frozen() -> None:
    noisy = ImageQualityExtractor(frozen_after=4)
    assert not any(noisy.extract(_scene(seed))["frozen"] for seed in range(10))

    stalled = ImageQualityExtractor(frozen_after=4)
    frame = _scene(3)
    flags = [stalled.extract(frame)["frozen"] for _ in range(6)]
    assert flags == [False, False, False, False, True, True]
