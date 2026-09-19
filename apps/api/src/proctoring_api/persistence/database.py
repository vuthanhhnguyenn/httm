"""Async SQLAlchemy database setup with safe SQLite defaults."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ..settings import Settings

SAFE_SQLITE_WAL_VERSION = (3, 51, 3)


class DatabaseConfigurationError(ValueError):
    """Raised when a database URL violates local-storage safety requirements."""


def _sqlite_path(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite"):
        return None
    parsed = urlparse(database_url.replace("+aiosqlite", "", 1))
    if parsed.path in ("", "/:memory:") or parsed.path == "/":
        return None
    raw = unquote(parsed.path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw).expanduser()


def _is_network_path(path: Path) -> bool:
    raw = str(path)
    return raw.startswith(("\\\\", "//"))


def sqlite_wal_supported() -> bool:
    """Only enable WAL on the audited SQLite version or an explicit backport build."""

    approved_backport = os.getenv("PROCTORING_SQLITE_WAL_BACKPORT", "").lower() in {"1", "true", "yes"}
    return sqlite3.sqlite_version_info >= SAFE_SQLITE_WAL_VERSION or approved_backport


def create_database_engine(database_url: str) -> AsyncEngine:
    sqlite_path = _sqlite_path(database_url)
    if sqlite_path is not None and _is_network_path(sqlite_path):
        raise DatabaseConfigurationError("SQLite database must not be stored on a network filesystem")
    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_async_engine(database_url, echo=False, future=True, connect_args=connect_args)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine.sync_engine, "connect")
        def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if sqlite_wal_supported():
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


class Database:
    def __init__(self, database_url: str | Settings) -> None:
        url = database_url.database_url if isinstance(database_url, Settings) else database_url
        self.engine = create_database_engine(url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def health(self) -> str:
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            return "unavailable"
        return "ready"

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session


class UnitOfWork:
    """One transaction boundary. Callers commit explicitly after successful writes."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self.session: AsyncSession | None = None

    async def __aenter__(self) -> "UnitOfWork":
        self.session = self._session_factory()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        if self.session is None:
            return
        if exc_type is not None:
            await self.session.rollback()
        await self.session.close()

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("UnitOfWork is not active")
        await self.session.commit()

    async def rollback(self) -> None:
        if self.session is not None:
            await self.session.rollback()


def create_database(settings: Settings) -> Database:
    return Database(settings)


__all__ = [
    "Database",
    "DatabaseConfigurationError",
    "SAFE_SQLITE_WAL_VERSION",
    "UnitOfWork",
    "create_database",
    "create_database_engine",
    "sqlite_wal_supported",
]
