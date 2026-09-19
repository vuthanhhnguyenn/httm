"""FastAPI authentication dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..settings import Settings, get_settings
from .principal import AuthPrincipal, parse_development_credential

_bearer = HTTPBearer(auto_error=False)


async def get_current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> AuthPrincipal:
    existing = getattr(request.state, "principal", None)
    if isinstance(existing, AuthPrincipal):
        return existing

    if settings.auth_mode == "development":
        if credentials is None:
            principal = AuthPrincipal.development()
        else:
            try:
                principal = parse_development_credential(credentials.credentials)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials") from exc
        request.state.principal = principal
        return principal

    authenticator = getattr(request.app.state, "authenticator", None)
    if not callable(authenticator) or credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    try:
        principal = authenticator(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials") from exc
    if not isinstance(principal, AuthPrincipal):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid principal")
    request.state.principal = principal
    return principal


__all__ = ["get_current_principal"]
