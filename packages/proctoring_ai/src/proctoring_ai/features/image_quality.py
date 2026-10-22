"""Cheap per-frame image statistics used to tell a covered/failed camera from behaviour."""

from __future__ import annotations

from typing import Any


class ImageQualityExtractor:
    """Brightness, contrast, edge density and blur on a downscaled grey frame.

    It also flags a *frozen* camera: when ``frozen_after`` consecutive analysed frames are
    practically identical, the device is probably stalled and rules must not treat the
    stale picture as fresh evidence. Only near-exact repeats count: a still candidate still
    produces sensor noise well above ``frozen_delta``.
    """

    def __init__(self, *, frozen_after: int = 16, frozen_delta: float = 0.02, edge_threshold: float = 40.0) -> None:
        self.frozen_after = max(2, frozen_after)
        self.frozen_delta = frozen_delta
        self.edge_threshold = edge_threshold
        self._previous: Any = None
        self._identical = 0

    def extract(self, image: Any) -> dict[str, float | str | bool]:
        try:
            import cv2
            import numpy as np

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(cv2.mean(gray)[0] / 255.0)
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(float)
            probabilities = histogram / max(1.0, histogram.sum())
            nonzero = probabilities[probabilities > 0]
            entropy = float(-np.sum(nonzero * np.log2(nonzero)))
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            _, deviation = cv2.meanStdDev(laplacian)
            blur = float(deviation[0, 0] ** 2)
            contrast = float(gray.std())
            edge_density = float(np.count_nonzero(np.abs(laplacian) > self.edge_threshold) / max(1, gray.size))
            small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            if self._previous is not None and float(np.mean(np.abs(small - self._previous))) < self.frozen_delta:
                self._identical += 1
            else:
                self._identical = 0
            self._previous = small
            return {"status": "ready", "brightness": brightness, "entropy": entropy, "blur": blur,
                    "contrast": contrast, "edge_density": edge_density, "occlusion_estimate": 0.0,
                    "frozen": self._identical >= self.frozen_after}
        except Exception:
            return {"status": "not_run", "reason": "IMAGE_QUALITY_UNAVAILABLE"}


__all__ = ["ImageQualityExtractor"]
