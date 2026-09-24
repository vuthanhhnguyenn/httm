from __future__ import annotations

from typing import Any


class ImageQualityExtractor:
    def extract(self, image: Any) -> dict[str, float | str]:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = float(cv2.mean(gray)[0] / 255.0)
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(float)
            probabilities = histogram / max(1.0, histogram.sum())
            nonzero = probabilities[probabilities > 0]
            entropy = float(-np.sum(nonzero * np.log2(nonzero)))
            _, deviation = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_64F))
            blur = float(deviation[0, 0] ** 2)
            return {"status": "ready", "brightness": brightness, "entropy": entropy, "blur": blur, "occlusion_estimate": 0.0}
        except Exception:
            return {"status": "not_run", "reason": "IMAGE_QUALITY_UNAVAILABLE"}
