"""Health endpoint for database, storage and optional analyzers."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/api", tags=["Health"])


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    database: Literal["ready", "degraded", "unavailable"]
    storage: Literal["ready", "degraded", "unavailable"]
    analyzers: dict[str, Literal["ready", "disabled", "degraded", "unavailable"]] = Field(default_factory=dict)


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def get_health(request: Request) -> HealthResponse:
    components = getattr(
        request.app.state,
        "health_components",
        {"database": "unavailable", "storage": "unavailable", "analyzers": {}},
    )
    database = components.get("database", "unavailable")
    storage = components.get("storage", "unavailable")
    analyzers = components.get("analyzers", {})
    if database == "unavailable" or storage == "unavailable":
        status = "unhealthy"
    elif database != "ready" or storage != "ready" or "degraded" in analyzers.values():
        status = "degraded"
    else:
        status = "healthy"
    return HealthResponse(
        status=status,
        version=getattr(request.app.state, "version", "0.1.0"),
        database=database,
        storage=storage,
        analyzers=analyzers,
    )


__all__ = ["HealthResponse", "router"]

