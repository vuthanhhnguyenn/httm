"""Authenticated identity used by application services.

The API stores only the subject and role. Credentials remain in the Authorization header and are
never accepted in query strings or resource URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    subject: str
    role: Role
    auth_mode: str = "bearer"

    def __post_init__(self) -> None:
        if not self.subject or len(self.subject) > 200:
            raise ValueError("principal subject must be 1..200 characters")

    @classmethod
    def development(cls, subject: str = "development-user", role: Role = Role.OPERATOR) -> "AuthPrincipal":
        return cls(subject=subject, role=role, auth_mode="development")


def parse_development_credential(value: str) -> AuthPrincipal:
    """Parse the local-only ``dev:<role>:<subject>`` header format."""

    parts = value.split(":", 2)
    if len(parts) != 3 or parts[0] != "dev":
        raise ValueError("invalid development credential")
    try:
        role = Role(parts[1])
    except ValueError as exc:
        raise ValueError("invalid development role") from exc
    return AuthPrincipal.development(subject=parts[2], role=role)


__all__ = ["AuthPrincipal", "Role", "parse_development_credential"]

