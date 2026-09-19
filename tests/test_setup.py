from pathlib import Path


def test_repository_setup_has_python_manifest() -> None:
    assert (Path(__file__).parents[1] / "pyproject.toml").is_file()

