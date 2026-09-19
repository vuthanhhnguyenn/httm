from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "apps" / "api" / "src", ROOT / "packages" / "proctoring_core" / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from proctoring_api.settings import Settings  # noqa: E402


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        auth_mode="development",
        storage_root=tmp_path / "storage",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'test.sqlite3').as_posix()}",
    )

