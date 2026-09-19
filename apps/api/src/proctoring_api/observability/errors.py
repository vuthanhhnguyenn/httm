"""Stable API errors and exception mapping."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .logging import sanitize_for_log


class ApiError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _payload(request: Request, *, code: str, message: str, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message, "request_id": _request_id(request)}
    if details:
        payload["details"] = sanitize_for_log(details)
    return payload


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=_payload(request, code=exc.code, message=exc.message, details=exc.details))


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_payload(request, code="VALIDATION_ERROR", message="request validation failed", details=exc.errors()))


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    status_code = exc.status_code
    code = "UNAUTHORIZED" if status_code == 401 else "FORBIDDEN" if status_code == 403 else "HTTP_ERROR"
    safe_message = str(sanitize_for_log(str(exc.detail)))
    return JSONResponse(status_code=status_code, content=_payload(request, code=code, message=safe_message))


async def unhandled_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=_payload(request, code="INTERNAL_ERROR", message="internal server error"))


def install_exception_handlers(app: Any) -> None:
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)


__all__ = ["ApiError", "install_exception_handlers"]
