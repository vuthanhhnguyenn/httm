"""Validate the versioned model manifest without downloading model files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("PyYAML is required to validate models/manifest.yaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_path: Path, *, model_root: Path | None = None) -> list[str]:
    manifest = _load_manifest(manifest_path)
    entries = manifest.get("models", manifest.get("artifacts", []))
    if not isinstance(entries, list):
        raise ValueError("manifest.models must be a list")
    errors: list[str] = []
    root = model_root or manifest_path.parent
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"models[{index}] must be a mapping")
            continue
        for field in ("source", "version", "license", "sha256", "class_map", "metric_baseline"):
            if field not in entry:
                errors.append(f"models[{index}] missing {field}")
        class_map = entry.get("class_map")
        if not isinstance(class_map, dict):
            errors.append(f"models[{index}].class_map must be a mapping")
        source = entry.get("source")
        expected = entry.get("sha256")
        if isinstance(source, str) and isinstance(expected, str) and expected and expected != "pending":
            model_path = root / source
            if model_path.is_file() and _sha256(model_path).lower() != expected.lower():
                errors.append(f"checksum mismatch: {model_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=Path("models/manifest.yaml"))
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    errors = verify(args.manifest, model_root=args.model_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
