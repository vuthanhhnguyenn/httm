"""Vietnamese text on BGR images (OpenCV's own fonts have no diacritics)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

_FONTS = ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/Library/Fonts/Arial Unicode.ttf")


def font_paths() -> list[str]:
    """System fonts first, then matplotlib's bundled DejaVuSans (always present on Kaggle images)."""

    import importlib.util
    from pathlib import Path

    paths = list(_FONTS)
    try:
        spec = importlib.util.find_spec("matplotlib")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        bundled = Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if bundled.is_file():
            paths.append(str(bundled))
    return paths


@lru_cache(maxsize=16)
def _font(size: int) -> Any:
    from PIL import ImageFont

    for path in font_paths():
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def put_text(image: Any, text: str, origin: tuple[int, int], size: int = 18,
             colour: tuple[int, int, int] = (255, 255, 0)) -> Any:
    """Vietnamese text on a BGR image; returns a new image."""

    from PIL import Image, ImageDraw

    font = _font(size)
    canvas = Image.fromarray(image[:, :, ::-1].copy())
    draw = ImageDraw.Draw(canvas)
    x, y = origin
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour, font=font)
    return np.asarray(canvas)[:, :, ::-1].copy()


@lru_cache(maxsize=256)
def label_image(text: str, size: int, colour: tuple[int, int, int]) -> Any:
    """Pre-rendered BGR label (black background); cached so per-frame cost is a slice copy."""

    from PIL import Image, ImageDraw

    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    canvas = Image.new("RGB", (int(right) + 4, int(bottom) + 2), (0, 0, 0))
    ImageDraw.Draw(canvas).text((2, 0), text, fill=(colour[2], colour[1], colour[0]), font=font)
    return np.asarray(canvas)[:, :, ::-1].copy()


def blit(frame: Any, label: Any, x: int, y: int) -> None:
    """Copy ``label`` onto ``frame`` at (x, y), clipped to the frame; in place."""

    height, width = frame.shape[:2]
    lh, lw = label.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + lw), min(height, y + lh)
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = label[y0 - y:y1 - y, x0 - x:x1 - x]


__all__ = ["blit", "label_image", "put_text"]
