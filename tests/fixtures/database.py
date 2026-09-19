from __future__ import annotations

import tempfile
from pathlib import Path


class TemporaryDatabase:
    """Temporary SQLite path fixture; the API database adapter owns the connection lifecycle."""

    def __init__(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="proctoring-test-")
        self.root = Path(self._directory.name)
        self.path = self.root / "test.sqlite3"
        self.url = f"sqlite+aiosqlite:///{self.path.as_posix()}"

    def close(self) -> None:
        self._directory.cleanup()

    def __enter__(self) -> "TemporaryDatabase":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

