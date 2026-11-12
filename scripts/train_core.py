"""Train lõi học (phase 3) từ đặc trưng và nhãn, chấm theo từng video để riêng.

    python scripts/train_core.py --features timeline/clip1 timeline/clip2 --labels data/labels/learned \
        --out models/learned_core/v1 [--leave-one-out]

``--features``: thư mục có ``features.csv`` (hoặc chính file csv). ``--labels``: thư mục các file
``<video>.jsonl``. Ngưỡng đỏ/vàng chọn trên xác suất leave-one-out gộp của mọi video (đỏ: đúng ≥ 90 %,
vàng: ≥ 50 %); model cuối train trên toàn bộ dữ liệu. Ghi ``model.txt`` và ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _src in ("packages/proctoring_core/src", "packages/proctoring_ai/src", "packages/proctoring_runtime/src"):
    if str(ROOT / _src) not in sys.path:
        sys.path.insert(0, str(ROOT / _src))

from proctoring_ai.multi.cctv.features import FEATURE_NAMES, FeatureRow, read_csv  # noqa: E402
from proctoring_ai.multi.cctv.labels import load_labels, match  # noqa: E402

CLASSES = ("NORMAL", "LOOK_AT_NEIGHBOR", "HEADS_TOGETHER", "PASSING_OBJECT", "PHONE_DETECTED",
           "DOCUMENT_DETECTED", "LOOK_AT_LAP", "TURN_AROUND")
MIN_TURN_AROUND = 10
RED_PRECISION, YELLOW_PRECISION = 0.9, 0.5
NEVER = 1.01
PARAMS = {"objective": "multiclass", "num_leaves": 15, "min_data_in_leaf": 5, "learning_rate": 0.05,
          "verbose": -1, "deterministic": True, "num_threads": 1}
ROUNDS = 300

Pairs = Sequence[tuple[FeatureRow, str]]


def _merge_rare(pairs: Pairs) -> list[tuple[FeatureRow, str]]:
    turns = sum(1 for _row, label in pairs if label == "TURN_AROUND")
    if 0 < turns < MIN_TURN_AROUND:
        print(f"TURN_AROUND chỉ có {turns} mẫu (< {MIN_TURN_AROUND}): gộp vào LOOK_AT_NEIGHBOR.", flush=True)
        return [(row, "LOOK_AT_NEIGHBOR" if label == "TURN_AROUND" else label) for row, label in pairs]
    return list(pairs)


def _matrix(rows: Sequence[FeatureRow]) -> Any:
    import numpy as np

    return np.array([[row.values[name] for name in FEATURE_NAMES] for row in rows], dtype=float)


def train(pairs: Pairs, *, seed: int = 0) -> tuple[Any, tuple[str, ...]]:
    """A LightGBM booster over ``FEATURE_NAMES`` and the classes it predicts, in ``CLASSES`` order (a class
    with no sample is left out). Samples are weighted by the inverse of their class frequency."""

    import lightgbm as lgb

    pairs = _merge_rare(pairs)
    counts: dict[str, int] = {}
    for _row, label in pairs:
        counts[label] = counts.get(label, 0) + 1
    classes = tuple(name for name in CLASSES if counts.get(name))
    index = {name: i for i, name in enumerate(classes)}
    weights = [len(pairs) / (len(classes) * counts[label]) for _row, label in pairs]
    data = lgb.Dataset(_matrix([row for row, _ in pairs]), label=[index[label] for _row, label in pairs],
                       weight=weights, feature_name=list(FEATURE_NAMES))
    params = {**PARAMS, "num_class": len(classes), "seed": seed}
    if len(classes) < 2:
        raise ValueError("cần ít nhất 2 lớp có nhãn để train")
    return lgb.train(params, data, num_boost_round=ROUNDS), classes


def predict(booster: Any, classes: Sequence[str], rows: Sequence[FeatureRow]) -> list[dict[str, float]]:
    if not rows:
        return []
    out = booster.predict(_matrix(rows))
    return [{name: float(p[i]) for i, name in enumerate(classes)} for p in out]


def _metrics(probs: Sequence[Mapping[str, float]], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    guesses = [max(p, key=lambda k: p[k]) for p in probs]
    for name in CLASSES[1:]:
        tp = sum(1 for g, t in zip(guesses, labels, strict=True) if g == name and t == name)
        said = sum(1 for g in guesses if g == name)
        real = sum(1 for t in labels if t == name)
        out[name] = {"precision": tp / said if said else 0.0, "recall": tp / real if real else 0.0,
                     "support": float(real)}
    return out


def leave_one_out(pairs_by_video: Mapping[str, Pairs]) -> dict[str, dict[str, Any]]:
    """For each video: train on the others, predict it. Per class ``precision``/``recall``/``support``
    at the window level, plus ``_probs`` and ``_labels`` for picking thresholds."""

    result: dict[str, dict[str, Any]] = {}
    for video, held in pairs_by_video.items():
        rest = [pair for other, items in pairs_by_video.items() if other != video for pair in items]
        if len({label for _row, label in rest}) < 2:
            print(f"Bỏ qua lượt để riêng {video}: các video còn lại chỉ có 1 lớp nhãn.", file=sys.stderr)
            continue
        booster, classes = train(rest)
        merged = "TURN_AROUND" not in classes
        labels = ["LOOK_AT_NEIGHBOR" if merged and label == "TURN_AROUND" else label for _row, label in held]
        probs = predict(booster, classes, [row for row, _ in held])
        result[video] = {**_metrics(probs, labels), "_probs": probs, "_labels": labels}
    return result


def pick_thresholds(probs: Sequence[float], truth: Sequence[bool], *, red_precision: float = RED_PRECISION,
                    yellow_precision: float = YELLOW_PRECISION) -> tuple[float, float]:
    """Smallest threshold whose windows at or above it are right at least that often (``NEVER`` if none)."""

    order = sorted(zip(probs, truth, strict=True), key=lambda item: -item[0])

    def lowest(target: float) -> float:
        best, right = NEVER, 0
        for count, (p, ok) in enumerate(order, 1):
            right += ok
            if right / count >= target:
                best = p
        return best

    return lowest(red_precision), lowest(yellow_precision)


def save_model(folder: Path, booster: Any, classes: Sequence[str], thresholds: Mapping[str, tuple[float, float]],
               meta_extra: Mapping[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(folder / "model.txt"))
    meta = {"feature_names": list(FEATURE_NAMES), "classes": list(classes),
            "thresholds": {name: list(value) for name, value in thresholds.items()},
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "licenses": [], **meta_extra}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(features: Sequence[Path], labels_dir: Path) -> dict[str, list[tuple[FeatureRow, str]]]:
    out: dict[str, list[tuple[FeatureRow, str]]] = {}
    for item in features:
        path = item / "features.csv" if item.is_dir() else item
        rows = read_csv(path)
        if not rows:
            continue
        video = rows[0].video
        label_path = labels_dir / f"{video}.jsonl"
        if not label_path.is_file():
            print(f"Bỏ qua {video}: chưa có nhãn {label_path}", file=sys.stderr)
            continue
        pairs, unmatched = match(rows, load_labels(label_path))
        print(f"{video}: {len(pairs)} cửa sổ có nhãn, {unmatched} nhãn không khớp cửa sổ nào", flush=True)
        out[video] = pairs
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train lõi học phase 3.")
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--leave-one-out", action="store_true", help="in bảng chấm từng video để riêng")
    args = parser.parse_args(argv)
    data = _load(args.features, args.labels)
    if not data:
        print("Không có video nào có nhãn.", file=sys.stderr)
        return 1
    every = [pair for items in data.values() for pair in items]
    booster, classes = train(every)
    thresholds: dict[str, tuple[float, float]] = {}
    loo = leave_one_out(data) if len(data) >= 2 else {}
    if loo:
        if args.leave_one_out:
            for video, metrics in loo.items():
                print(f"\n== {video} (để riêng) ==")
                for name in CLASSES[1:]:
                    m = metrics.get(name)
                    if m and m["support"]:
                        print(f"  {name:18s} đúng {m['precision']:.2f}  bắt {m['recall']:.2f}  ({int(m['support'])} cửa sổ)")
        probs = [p for m in loo.values() for p in m["_probs"]]
        truth = [t for m in loo.values() for t in m["_labels"]]
    else:
        print("Không chấm được video nào để riêng: ngưỡng chọn trên chính dữ liệu train (lạc quan).", file=sys.stderr)
        probs, truth = predict(booster, classes, [row for row, _ in every]), [label for _row, label in every]
    for name in classes:
        if name != "NORMAL":
            thresholds[name] = pick_thresholds([p.get(name, 0.0) for p in probs], [t == name for t in truth])
    save_model(args.out, booster, classes, thresholds,
               {"videos": sorted(data), "seat_height_median": sorted(r.seat_box.height for r, _ in every)[len(every) // 2],
                "frame_size": None, "windows": len(every)})
    print(f"\nĐã lưu model vào {args.out}; ngưỡng (đỏ, vàng): {thresholds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
