"""Replay a local JSONL fixture and print expected event windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def replay(path: Path, manifest: Path) -> list[dict[str, object]]:
    expected = yaml.safe_load(manifest.read_text(encoding="utf-8")).get("windows", {})
    frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"frame_count": len(frames), "fixture": path.stem, "expected": expected.get(path.stem, {"expected_events": []})}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("tests/video_samples/manifest.yaml"))
    args = parser.parse_args()
    for item in replay(args.fixture, args.manifest):
        print(json.dumps(item, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
