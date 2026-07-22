import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT
from src.face_detection import detect_face_landmarks
from src.region_masks import MIN_POLYGON_POINTS, build_region_masks

SAMPLES = PROJECT_ROOT / "data" / "sample_images"

REGION_KEYS = ["forehead", "left_cheek", "right_cheek", "jawline", "combined"]


def _load(name):
    return np.array(Image.open(SAMPLES / name).convert("RGB"))


def _masks_for(name):
    img = _load(name)
    result = detect_face_landmarks(img)
    assert result.success
    return build_region_masks(img.shape, result.landmarks), img.shape


def test_all_regions_present_and_shaped_correctly():
    masks, shape = _masks_for("face_astronaut.png")
    for key in REGION_KEYS:
        assert key in masks
        assert masks[key].shape == shape[:2]


def test_regions_have_reasonable_pixel_counts():
    masks, _ = _masks_for("face_astronaut.png")
    for key in ["forehead", "left_cheek", "right_cheek", "jawline"]:
        count = int((masks[key] > 0).sum())
        assert count > 50, f"{key} has too few pixels: {count}"


def test_regions_do_not_overlap_eyes_lips_significantly():
    from src.region_masks import EYE_INDICES, LIPS_INDICES, NOSE_INDICES, _pts

    img = _load("face_astronaut.png")
    result = detect_face_landmarks(img)
    masks = build_region_masks(img.shape, result.landmarks)

    eye_pts = _pts(result.landmarks, EYE_INDICES).astype(int)
    lips_pts = _pts(result.landmarks, LIPS_INDICES).astype(int)

    combined = masks["combined"]
    nose_pts = _pts(result.landmarks, NOSE_INDICES).astype(int)

    for x, y in np.vstack([eye_pts, lips_pts, nose_pts]):
        assert combined[y, x] == 0, "region mask overlaps an eye/lip/nose landmark"


def test_cheek_masks_are_reasonably_balanced_on_front_facing_face():
    masks, _ = _masks_for("face_astronaut.png")
    left = int((masks["left_cheek"] > 0).sum())
    right = int((masks["right_cheek"] > 0).sum())
    balance = min(left, right) / max(left, right)
    assert balance >= 0.45, f"cheek mask areas too imbalanced: left={left}, right={right}"


def test_masks_align_across_pose_and_lighting_variants():
    variants = [
        "face_astronaut.png",
        "face_variant_flipped.png",
        "face_variant_rotated_bright.png",
        "face_variant_dark.png",
    ]
    for name in variants:
        masks, _ = _masks_for(name)
        for key in ["forehead", "left_cheek", "right_cheek", "jawline"]:
            assert int((masks[key] > 0).sum()) > 30, f"{name}: {key} mask too small"


def test_no_landmarks_does_not_crash():
    masks = build_region_masks((256, 256, 3), None)
    for key in REGION_KEYS:
        assert masks[key].sum() == 0


def test_insufficient_points_does_not_crash():
    sparse = [(10, 10)] * (MIN_POLYGON_POINTS - 1)
    masks = build_region_masks((256, 256, 3), sparse)
    for key in REGION_KEYS:
        assert masks[key].sum() == 0
