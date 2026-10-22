"""Spatial association helpers used by the behaviour engine."""

from __future__ import annotations

from ..observations import Box


def inside_person(inner: Box | None, outer: Box | None, *, margin: float) -> bool:
    """True when ``inner``'s centre lies in ``outer`` widened by ``margin`` of its size."""

    if inner is None or outer is None:
        return False
    x, y = inner.center
    margin_x, margin_y = outer.width * margin, outer.height * margin
    return (outer.x - margin_x <= x <= outer.x + outer.width + margin_x
            and outer.y - margin_y <= y <= outer.y + outer.height + margin_y)


__all__ = ["inside_person"]
