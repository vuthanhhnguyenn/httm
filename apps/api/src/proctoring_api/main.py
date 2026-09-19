"""FastAPI application bootstrap and lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from . import __version__
from .observability.errors import install_exception_handlers
from .observability.logging import configure_logging, get_logger
from .observability.middleware import RequestIdMiddleware
from .persistence.database import Database, create_database
from .persistence.writer import SingleWriterQueue
from .routes import router
from .settings import Settings, get_settings

logger = get_logger(__name__)


def _storage_status(path: Path) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".health-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return "unavailable"
    return "ready"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    database: Database | None = None
    try:
        database = create_database(settings)
        database_status = await database.health()
    except Exception:
        logger.exception("database_initialization_failed")
        database_status = "unavailable"
    storage_status = _storage_status(settings.storage_root)
    writer = SingleWriterQueue()
    await writer.start()
    app.state.database = database
    app.state.writer = writer
    app.state.health_components = {
        "database": database_status,
        "storage": storage_status,
        "analyzers": {
            "object_detector": "disabled",
            "face_landmarker": "disabled",
            "pose": "disabled",
            "hands": "disabled",
        },
    }
    try:
        yield
    finally:
        await writer.close()
        if database is not None:
            await database.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="Smart Exam Proctoring API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.version = __version__
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    install_exception_handlers(app)
    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
