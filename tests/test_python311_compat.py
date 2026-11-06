"""Kaggle images may still run Python 3.11 and the notebooks do not install the project, so the code must parse there."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([*ROOT.glob("packages/*/src/**/*.py"), *ROOT.glob("scripts/*.py")])


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_parses_as_python_311(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))
