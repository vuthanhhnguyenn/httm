"""Validate local runtime and exam-policy YAML files before an app session starts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "packages" / "proctoring_core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from proctoring_core.config.schema import load_config_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="Print canonical validated JSON")
    args = parser.parse_args()
    try:
        config, policy = load_config_files(args.config, args.policy)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Configuration validation failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"config": config.model_dump(mode="json"), "policy": policy.model_dump(mode="json")}, indent=2))
    else:
        print(f"Configuration valid: {args.config}")
        print(f"Policy valid: {args.policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
