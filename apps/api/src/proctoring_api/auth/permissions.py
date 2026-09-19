"""Explicit role-to-permission mapping."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Callable

from .principal import AuthPrincipal, Role


class Permission(StrEnum):
    SESSION_READ = "session:read"
    SESSION_CREATE = "session:create"
    SESSION_CONTROL = "session:control"
    EVENT_READ = "event:read"
    EVENT_REVIEW = "event:review"
    EVIDENCE_READ = "evidence:read"
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    POLICY_READ = "policy:read"
    POLICY_WRITE = "policy:write"
    AUDIT_READ = "audit:read"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.OPERATOR: frozenset(
        {
            Permission.SESSION_READ,
            Permission.SESSION_CREATE,
            Permission.SESSION_CONTROL,
            Permission.EVENT_READ,
            Permission.EVIDENCE_READ,
            Permission.POLICY_READ,
        }
    ),
    Role.REVIEWER: frozenset(
        {
            Permission.SESSION_READ,
            Permission.EVENT_READ,
            Permission.EVENT_REVIEW,
            Permission.EVIDENCE_READ,
            Permission.POLICY_READ,
        }
    ),
    Role.ADMIN: frozenset(Permission),
}


def has_permission(principal: AuthPrincipal, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[principal.role]


def require_permission(permission: Permission) -> Callable[..., Any]:
    """Return a FastAPI dependency that enforces one permission."""

    from fastapi import Depends, HTTPException, status

    from .dependencies import get_current_principal

    async def dependency(principal: AuthPrincipal = Depends(get_current_principal)) -> AuthPrincipal:
        if not has_permission(principal, permission):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="permission denied")
        return principal

    return dependency


def require_any_role(roles: Iterable[Role]) -> Callable[..., Any]:
    allowed = frozenset(roles)

    from fastapi import Depends, HTTPException, status

    from .dependencies import get_current_principal

    async def dependency(principal: AuthPrincipal = Depends(get_current_principal)) -> AuthPrincipal:
        if principal.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="role denied")
        return principal

    return dependency


__all__ = ["Permission", "ROLE_PERMISSIONS", "has_permission", "require_any_role", "require_permission"]
