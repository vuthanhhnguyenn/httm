"""Download the models declared in models/manifest.yaml and verify every SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

MAX_MODEL_BYTES = 250 * 1024 * 1024


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(manifest_path: Path) -> list[Path]:
    manifest_path = manifest_path.resolve()
    payload: dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("models", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.models must be a list")
    root = manifest_path.parent.resolve()
    resolved: list[tuple[dict[str, Any], Path]] = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("every model manifest entry must be a mapping")
        source = item.get("source")
        url = item.get("download_url")
        checksum = str(item.get("sha256", "")).lower()
        if not isinstance(source, str) or not isinstance(url, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"model {item.get('name', '<unknown>')} needs source, HTTPS URL and SHA-256")
        if urlparse(url).scheme != "https":
            raise ValueError("model downloads must use HTTPS")
        target = (root / source).resolve()
        if not target.is_relative_to(root):
            raise ValueError("model source must remain inside the model directory")
        resolved.append((item, target))

    downloaded: list[Path] = []
    for item, target in resolved:
        expected = str(item["sha256"]).lower()
        if target.is_file():
            if _hash(target) != expected:
                raise ValueError(f"existing model checksum does not match manifest: {target}")
            print(f"OK: verified {item['name']}")
            downloaded.append(target)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            request = Request(str(item["download_url"]), headers={"User-Agent": "smart-exam-proctoring-local"})
            with (
                urlopen(request, timeout=120) as response,
                tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as temporary,
            ):
                temporary_path = Path(temporary.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_MODEL_BYTES:
                        raise ValueError("model download exceeds the 250 MiB safety limit")
                    temporary.write(chunk)
            actual = _hash(temporary_path)
            if actual != expected:
                raise ValueError(f"checksum mismatch for {item['name']}: expected {expected}, got {actual}")
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        print(f"OK: downloaded and verified {item['name']}")
        downloaded.append(target)
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.yaml"))
    args = parser.parse_args()
    download_models(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
