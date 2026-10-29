from __future__ import annotations

import pytest
from proctoring_core.observations import Box

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from proctoring_ai.multi.crop import CropTransform, SubjectCropper  # noqa: E402

FRAME = (480, 640, 3)


def test_head_crop_is_upper_45_percent_with_margin_and_upscaled() -> None:
    image = np.zeros(FRAME, np.uint8)
    crop = SubjectCropper().crop_head(image, Box(0.25, 0.25, 0.25, 0.5))  # 160×240 px person
    assert crop is not None
    t = crop.transform
    assert (t.x0, t.width) == (160 - 24, 160 + 48)  # 15% of width each side
    assert t.y0 == 120 - 16 and t.height == round(240 * 0.45) + 2 * 16  # 15% of head height
    assert min(crop.image.shape[:2]) >= 256
    assert crop.image.flags["C_CONTIGUOUS"]


def test_small_min_side_does_not_shrink() -> None:
    crop = SubjectCropper(min_side=64).crop_head(np.zeros(FRAME, np.uint8), Box(0.25, 0.25, 0.25, 0.5))
    assert crop is not None
    assert crop.image.shape[:2] == (crop.transform.height, crop.transform.width)


def test_body_crop_covers_whole_person_with_margin() -> None:
    crop = SubjectCropper().crop_body(np.zeros(FRAME, np.uint8), Box(0.25, 0.25, 0.25, 0.5))
    assert crop is not None
    t = crop.transform
    assert (t.x0, t.y0, t.width, t.height) == (144, 96, 192, 288)


def test_landmark_maps_back_to_frame_coordinates() -> None:
    t = CropTransform(x0=100, y0=50, width=200, height=100, frame_width=640, frame_height=480)
    assert t.map_point(0.5, 0.5) == pytest.approx((200 / 640, 100 / 480))
    face = t.map_face({"bbox": {"x": 0, "y": 0, "width": 1, "height": 1}, "face_width": 0.5,
                       "keypoints": {"nose": (0.5, 0.5)}, "frame_width": 300, "frame_height": 150})
    assert face["bbox"] == pytest.approx({"x": 100 / 640, "y": 50 / 480, "width": 200 / 640,
                                          "height": 100 / 480})
    assert face["face_width"] == pytest.approx(0.5 * 200 / 640)
    assert face["keypoints"]["nose"] == pytest.approx((200 / 640, 100 / 480))
    assert (face["frame_width"], face["frame_height"]) == (640, 480)


def test_hand_and_pose_landmarks_use_the_same_mapping() -> None:
    t = CropTransform(x0=100, y0=50, width=200, height=100, frame_width=640, frame_height=480)
    hand = t.map_hand({"bbox": {"x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5},
                       "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1}], "wrist": (0.5, 0.5),
                       "index_tip": (1.0, 1.0), "handedness": "Left"})
    expected = (200 / 640, 100 / 480)
    assert hand["wrist"] == pytest.approx(expected)
    assert (hand["landmarks"][0]["x"], hand["landmarks"][0]["y"]) == pytest.approx(expected)
    assert hand["landmarks"][0]["z"] == 0.1 and hand["handedness"] == "Left"
    assert hand["index_tip"] == pytest.approx((300 / 640, 150 / 480))
    pose = t.map_pose({"bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                       "landmarks": [{"x": 0.5, "y": 0.5, "visibility": 0.9}], "quality": 0.5})
    assert (pose["landmarks"][0]["x"], pose["landmarks"][0]["y"]) == pytest.approx(expected)
    assert pose["landmarks"][0]["visibility"] == 0.9 and pose["quality"] == 0.5


def test_box_touching_frame_edge_is_clamped() -> None:
    crop = SubjectCropper().crop_head(np.zeros(FRAME, np.uint8), Box(0.9, 0.0, 0.2, 0.5))
    assert crop is not None
    t = crop.transform
    assert t.x0 + t.width <= 640 and t.y0 == 0


def test_degenerate_box_returns_none() -> None:
    image = np.zeros(FRAME, np.uint8)
    assert SubjectCropper().crop_head(image, Box(0.5, 0.5, 0.001, 0.001)) is None
    assert SubjectCropper().crop_head(image, Box(1.2, 0.5, 0.1, 0.1)) is None
