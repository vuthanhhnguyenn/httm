"""Head yaw and pitch from the five face points of a body skeleton, with HHP-Net (Cantarini et al., WACV 2022,
MIT; github.com/cantarinigiorgio/HHP-Net).

The 0.4 MB network was trained on faces from 300W-LP and BIWI; it reads the nose, eyes and ears (x, y,
confidence each), centred and scaled per axis, and returns yaw, pitch and roll in degrees with an uncertainty for
each. ``models/pretrained/hhp_net.onnx`` is its SavedModel exported with tf2onnx; it runs on OpenVINO (CPU).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .head import Keypoints

MODEL_PATH = Path(__file__).resolve().parents[6] / "models" / "pretrained" / "hhp_net.onnx"
MIN_POINTS = 3
POINT_MIN_CONF = 0.3


def hhp_input(kps: Keypoints | None) -> np.ndarray | None:
    """The network's input (15 values) for one skeleton, or None with fewer than ``MIN_POINTS`` usable face
    points. Points under ``POINT_MIN_CONF`` count as missing (0, 0, 0)."""

    if kps is None or len(kps) < 5:
        return None
    pts = [(float(p[0]), float(p[1]), float(p[2])) if p[2] >= POINT_MIN_CONF else (0.0, 0.0, 0.0) for p in kps[:5]]
    valid = [p for p in pts if p[2] > 0.0]
    if len(valid) < MIN_POINTS:
        return None
    cx, cy = sum(p[0] for p in valid) / len(valid), sum(p[1] for p in valid) / len(valid)
    mx, my = max(abs(p[0] - cx) for p in valid), max(abs(p[1] - cy) for p in valid)
    out: list[float] = []
    for x, y, c in pts:
        if c > 0.0:
            out += [(x - cx) / mx if mx else 0.0, (y - cy) / my if my else 0.0, c]
        else:
            out += [0.0, 0.0, 0.0]
    return np.asarray(out, dtype=np.float32).reshape(1, 15, 1)


@lru_cache(maxsize=1)
def _compiled(path: str) -> Any:
    from openvino import Core  # type: ignore[import-untyped]

    core = Core()
    return core.compile_model(core.read_model(path), "CPU")


def hhp_yaw_pitch(kps: Keypoints | None, model_path: Path = MODEL_PATH) -> tuple[float, float, float] | None:
    """(yaw, pitch, yaw uncertainty) in degrees; the sign follows the network (BIWI convention), so
    callers compare against the seat's own baseline, never an absolute angle."""

    batch = hhp_input(kps)
    if batch is None:
        return None
    result = _compiled(str(model_path))(batch)
    yaw, pitch = np.asarray(result[0]).reshape(-1), np.asarray(result[1]).reshape(-1)
    return float(yaw[0]), float(pitch[0]), float(yaw[1])
