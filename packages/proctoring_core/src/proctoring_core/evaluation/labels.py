"""Load labelled behaviour intervals and map dataset codes to event types."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..types import EventType


class LabelError(ValueError):
    """A label or mapping file cannot be used; the message names the file and line."""

    def __init__(self, path: Path, line: int | None, message: str) -> None:
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {message}")
        self.path = path
        self.line = line


@dataclass(frozen=True, slots=True)
class RawLabel:
    start_ms: int
    end_ms: int
    code: str
    line: int


@dataclass(frozen=True, slots=True)
class LabelInterval:
    """One labelled behaviour; any of ``event_types`` firing counts as a detection."""

    event_types: frozenset[EventType]
    start_ms: int
    end_ms: int
    source_code: str
    ignore: bool = False

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("label interval must have 0 <= start < end")


@dataclass(frozen=True, slots=True)
class CodeMapping:
    events: frozenset[EventType]
    ignore: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class LabelMapping:
    version: int
    codes: Mapping[str, CodeMapping]
    tolerance_ms: int = 2000
    # Same-code intervals closer than this are one behaviour (datasets often split one call in pieces).
    merge_gap_ms: int = 0


# Up to five digits: the dataset has the odd zero-padded time such as "00903".
_OEP_LINE = re.compile(r"^\s*(\d{3,5})\s+(\d{3,5})\s+(\S+)\s*$")


def _mmss_to_ms(token: str) -> int:
    value = int(token)
    minutes, seconds = divmod(value, 100)
    if seconds >= 60:
        raise ValueError(f"invalid MMSS value {token!r}")
    return (minutes * 60 + seconds) * 1000


def _clock_to_ms(token: str) -> int:
    token = token.strip()
    if ":" in token:
        minutes, seconds = token.split(":", 1)
        return round((int(minutes) * 60 + float(seconds)) * 1000)
    return round(float(token) * 1000)


def parse_oep(path: Path) -> list[RawLabel]:
    """Parse OEP ``gt.txt``: ``MMSS<ws>MMSS<ws>code`` per line; blank lines ignored."""

    labels: list[RawLabel] = []
    for number, line in enumerate(_read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        match = _OEP_LINE.match(line)
        if match is None:
            raise LabelError(path, number, f"expected 'MMSS MMSS code', got {line.strip()!r}")
        try:
            start, end = _mmss_to_ms(match.group(1)), _mmss_to_ms(match.group(2))
        except ValueError as exc:
            raise LabelError(path, number, str(exc)) from exc
        if end <= start:
            # OEP marks one-second behaviours as e.g. "0819 0819"; keep them as 1 s intervals.
            if end == start:
                end = start + 1000
            else:
                raise LabelError(path, number, "end time is before start time")
        labels.append(RawLabel(start, end, match.group(3), number))
    return labels


def parse_csv(path: Path) -> list[RawLabel]:
    """Parse ``start,end,label`` with ``mm:ss[.fff]`` or seconds."""

    labels: list[RawLabel] = []
    text = _read_text(path)
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or not {"start", "end", "label"} <= set(reader.fieldnames):
        raise LabelError(path, 1, "CSV header must contain start,end,label")
    for number, row in enumerate(reader, start=2):
        try:
            start, end = _clock_to_ms(row["start"]), _clock_to_ms(row["end"])
        except (TypeError, ValueError) as exc:
            raise LabelError(path, number, f"invalid time: {exc}") from exc
        if end <= start:
            raise LabelError(path, number, "end time must be after start time")
        labels.append(RawLabel(start, end, str(row["label"]).strip(), number))
    return labels


def load_labels(path: Path, label_format: str) -> list[RawLabel]:
    if label_format == "oep":
        return parse_oep(path)
    if label_format == "csv":
        return parse_csv(path)
    raise LabelError(path, None, f"unknown label format {label_format!r}")


def load_mapping(path: Path) -> LabelMapping:
    payload = yaml.safe_load(_read_text(path)) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("codes"), dict):
        raise LabelError(path, None, "mapping must contain a 'codes' table")
    codes: dict[str, CodeMapping] = {}
    for code, item in payload["codes"].items():
        if not isinstance(item, dict):
            raise LabelError(path, None, f"code {code!r} must be a table")
        try:
            events = frozenset(EventType(name) for name in item.get("events", []))
        except ValueError as exc:
            raise LabelError(path, None, f"code {code!r}: {exc}") from exc
        ignore = bool(item.get("ignore", False))
        if not events and not ignore:
            raise LabelError(path, None, f"code {code!r} needs 'events' or 'ignore: true'")
        codes[str(code)] = CodeMapping(events, ignore, str(item.get("note", "")))
    return LabelMapping(int(payload.get("version", 1)), codes, int(payload.get("tolerance_ms", 2000)),
                        int(payload.get("merge_gap_ms", 0)))


def merge_close(raw: list[RawLabel], gap_ms: int) -> list[RawLabel]:
    """Join same-code labels whose gap is at most ``gap_ms`` (input order is not assumed)."""

    if gap_ms <= 0:
        return list(raw)
    merged: list[RawLabel] = []
    for label in sorted(raw, key=lambda item: (item.code, item.start_ms)):
        last = merged[-1] if merged else None
        if last is not None and last.code == label.code and label.start_ms - last.end_ms <= gap_ms:
            merged[-1] = RawLabel(last.start_ms, max(last.end_ms, label.end_ms), last.code, last.line)
        else:
            merged.append(label)
    return sorted(merged, key=lambda item: (item.start_ms, item.code))


def apply_mapping(raw: list[RawLabel], mapping: LabelMapping | None, path: Path,
                  duration_ms: int | None = None) -> list[LabelInterval]:
    """Resolve codes: an ``EventType`` name maps to itself, other codes use ``mapping``."""

    intervals: list[LabelInterval] = []
    if mapping is not None:
        raw = merge_close(raw, mapping.merge_gap_ms)
    for label in raw:
        if duration_ms is not None and label.start_ms >= duration_ms:
            raise LabelError(path, label.line, f"label starts after video end ({duration_ms} ms)")
        end = min(label.end_ms, duration_ms) if duration_ms is not None else label.end_ms
        if label.code in EventType.__members__:
            intervals.append(LabelInterval(frozenset({EventType(label.code)}), label.start_ms, end, label.code))
            continue
        code = mapping.codes.get(label.code) if mapping is not None else None
        if code is None:
            raise LabelError(path, label.line, f"label code {label.code!r} is not in the mapping")
        intervals.append(LabelInterval(code.events, label.start_ms, end, label.code, code.ignore))
    return intervals


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LabelError(path, None, f"cannot read file: {exc.strerror or exc}") from exc


def mapping_from_dict(payload: Mapping[str, Any]) -> LabelMapping:
    """Build a mapping from an in-memory table (used by tests and generated sets)."""

    codes = {
        str(code): CodeMapping(frozenset(EventType(name) for name in item.get("events", [])),
                               bool(item.get("ignore", False)), str(item.get("note", "")))
        for code, item in payload.get("codes", {}).items()
    }
    return LabelMapping(int(payload.get("version", 1)), codes, int(payload.get("tolerance_ms", 2000)))


__all__ = [
    "CodeMapping",
    "LabelError",
    "LabelInterval",
    "LabelMapping",
    "RawLabel",
    "apply_mapping",
    "load_labels",
    "load_mapping",
    "mapping_from_dict",
    "parse_csv",
    "parse_oep",
]
