"""Application settings with safe environment handling."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["development", "test", "production"] = "development"
    auth_mode: Literal["development", "bearer"] = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    database_url: str = "sqlite+aiosqlite:///./storage/sessions/proctoring.sqlite3"
    storage_root: Path = Path("./storage")
    model_manifest_path: Path = Path("./models/manifest.yaml")
    log_level: str = "INFO"
    evidence_retention_days: int = 30
    jwt_secret: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("api_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("api_port must be between 1 and 65535")
        return value

    @field_validator("evidence_retention_days")
    @classmethod
    def validate_retention(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("evidence_retention_days must be greater than zero")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.app_env == "production" and self.auth_mode == "development":
            raise ValueError("AUTH_MODE=development is not allowed when APP_ENV=production")
        object.__setattr__(self, "storage_root", self.storage_root.expanduser().resolve())
        object.__setattr__(self, "model_manifest_path", self.model_manifest_path.expanduser().resolve())

    def redacted(self) -> dict[str, object]:
        """Return settings safe for structured logs; never include credentials."""

        return {
            "app_env": self.app_env,
            "auth_mode": self.auth_mode,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "database_url": _redact_database_url(self.database_url),
            "storage_root": str(self.storage_root),
            "model_manifest_path": str(self.model_manifest_path),
            "log_level": self.log_level,
            "evidence_retention_days": self.evidence_retention_days,
        }


def _redact_database_url(value: str) -> str:
    if "@" not in value:
        return value
    scheme, location = value.split("@", 1)
    prefix = scheme.rsplit("://", 1)[0]
    return f"{prefix}://***@{location}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]

