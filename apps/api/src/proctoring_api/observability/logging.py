"""Structured logging with explicit redaction of sensitive monitoring data."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"(?:token|authorization|password|secret|credential|bearer|frame|landmark|evidence|evidence_path|absolute_path)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|^/|^\\\\)")


def sanitize_for_log(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): sanitize_for_log(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, str):
        if re.search(r"Bearer\s+\S+", value, re.IGNORECASE):
            return re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", value, flags=re.IGNORECASE)
        if key and "path" in key.lower() and _ABSOLUTE_PATH.search(value):
            return "[PATH_REDACTED]"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {"message", "msg", "args", "exc_info", "exc_text", "stack_info"}:
                continue
            if key in payload or key in logging.LogRecord(None, 0, "", 0, "", (), None).__dict__:
                continue
            payload[key] = sanitize_for_log(value, key=key)
        if record.exc_info:
            payload["exception"] = "[REDACTED]"
        return json.dumps(sanitize_for_log(payload), ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["JsonFormatter", "configure_logging", "get_logger", "sanitize_for_log"]
