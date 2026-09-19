from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "packages" / "proctoring_core" / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.fixtures.clock import FakeClock  # noqa: E402


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()

