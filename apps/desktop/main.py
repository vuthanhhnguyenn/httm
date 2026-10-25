"""Launch the local native proctoring application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for source_root in (
    REPOSITORY_ROOT / "packages" / "proctoring_core" / "src",
    REPOSITORY_ROOT / "packages" / "proctoring_ai" / "src",
    REPOSITORY_ROOT / "packages" / "proctoring_runtime" / "src",
    REPOSITORY_ROOT / "apps" / "desktop" / "src",
):
    sys.path.insert(0, str(source_root))

from proctoring_desktop.window import MainWindow  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main() -> int:
    # Shows which model runs on GPU or CPU, and why a GPU was not used.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    window = MainWindow(REPOSITORY_ROOT)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
