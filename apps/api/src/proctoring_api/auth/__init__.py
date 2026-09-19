"""Authentication principal and role-based permissions."""

from .principal import AuthPrincipal, Role
from .permissions import Permission, has_permission, require_permission

__all__ = ["AuthPrincipal", "Permission", "Role", "has_permission", "require_permission"]

