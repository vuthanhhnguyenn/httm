"""Request correlation middleware that never logs request credentials or bodies."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging import get_logger

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
logger = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                extra={"request_id": request_id, "method": request.method, "status_code": 500},
            )
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        path_template = getattr(route, "path", "unknown")
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path_template": path_template,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
            },
        )
        return response


__all__ = ["RequestIdMiddleware"]
