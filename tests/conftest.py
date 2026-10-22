from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "apps" / "desktop" / "src",
    ROOT / "packages" / "proctoring_core" / "src",
    ROOT / "packages" / "proctoring_ai" / "src",
    ROOT / "packages" / "proctoring_runtime" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
