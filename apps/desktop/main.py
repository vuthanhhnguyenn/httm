"""Launch the local native proctoring application."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for source_root in (
    REPOSITORY_ROOT / "packages" / "proctoring_core" / "src",
    REPOSITORY_ROOT / "packages" / "proctoring_ai" / "src",
    REPOSITORY_ROOT / "apps" / "desktop" / "src",
):
    sys.path.insert(0, str(source_root))

from PySide6.QtWidgets import QApplication

from proctoring_desktop.window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(REPOSITORY_ROOT)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
