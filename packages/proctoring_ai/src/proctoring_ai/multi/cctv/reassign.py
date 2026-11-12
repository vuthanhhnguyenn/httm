"""Replay seat assignment from the people a timeline cached (``ReviewFrame.people``), without the models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any
from uuid import UUID

from proctoring_core.observations import Box

from .head import Keypoints
from .review_rules import ReviewFrame, assign_poses

Assigner = Callable[[Mapping[UUID, Box], Sequence[Mapping[str, Any]]], Mapping[UUID, Keypoints]]


def reassign_frames(frames: Sequence[ReviewFrame], assigner: Assigner) -> list[ReviewFrame]:
    """Frames with ``poses`` recomputed from their cached people; a frame cached without people keeps its poses."""

    return [replace(frame, poses=dict(assigner(frame.seats, frame.people))) if frame.people else frame
            for frame in frames]


def assigner_named(name: str) -> Assigner:
    """``old``: the nearest-head rule (``assign_poses(by_body=False)``)."""

    if name == "old":
        return lambda seats, people: assign_poses(seats, people, by_body=False)
    raise ValueError(f"cách gán lại không có: {name}")
