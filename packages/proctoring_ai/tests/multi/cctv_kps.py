"""Synthetic COCO keypoints for CCTV-mode tests."""

from __future__ import annotations


def person(cx: float = 0.5, top: float = 0.3, *, nose_dx: float = 0.0, face: bool = True,
           standing: bool = False, shoulder_w: float = 0.1, wrist: tuple[float, float] | None = None,
           ) -> list[list[float]]:
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    head_y, shoulder_y = top + 0.03, top + 0.08
    if face:
        kps[0] = [cx + nose_dx * shoulder_w, head_y, 0.9]
        kps[1] = [cx + nose_dx * shoulder_w - 0.01, head_y - 0.01, 0.9]
        kps[2] = [cx + nose_dx * shoulder_w + 0.01, head_y - 0.01, 0.9]
    kps[3] = [cx - 0.02, head_y, 0.6]
    kps[4] = [cx + 0.02, head_y, 0.6]
    kps[5] = [cx - shoulder_w / 2, shoulder_y, 0.9]
    kps[6] = [cx + shoulder_w / 2, shoulder_y, 0.9]
    hip_y = shoulder_y + 0.12
    kps[11] = [cx - 0.03, hip_y, 0.8]
    kps[12] = [cx + 0.03, hip_y, 0.8]
    knee_y = hip_y + (0.1 if standing else 0.03)
    kps[13] = [cx - 0.03, knee_y, 0.8]
    kps[14] = [cx + 0.03, knee_y, 0.8]
    if wrist is not None:
        kps[9] = [wrist[0], wrist[1], 0.9]
    return kps
