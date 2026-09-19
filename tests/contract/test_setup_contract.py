from pathlib import Path

import pytest


@pytest.mark.contract
def test_setup_contract_keeps_frontend_manifest_and_api_config() -> None:
    root = Path(__file__).parents[2]
    assert (root / "apps" / "web" / "package.json").is_file()
    assert (root / "configs" / "exam_policy.example.yaml").is_file()

