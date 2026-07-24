import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT
from src.face_detection import detect_face_landmarks
from src.region_masks import (
    MIN_POLYGON_POINTS,
    build_region_masks,
    refine_masks_for_capture,
)

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


def test_jawline_mask_avoids_central_chin_region():
    img = _load("face_astronaut.png")
    result = detect_face_landmarks(img)
    masks = build_region_masks(img.shape, result.landmarks)
    jawline = masks["jawline"] > 0
    ys, xs = np.where(jawline)
    assert len(xs) > 0
    face_center_x = np.mean([p[0] for p in result.landmarks])
    central = np.abs(xs - face_center_x) < img.shape[1] * 0.04
    assert central.mean() < 0.20


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


def test_capture_refinement_removes_synthetic_glasses_reflection_zone():
    from src.region_masks import EYE_INDICES, _pts

    image = _load("face_astronaut.png")
    result = detect_face_landmarks(image)
    masks = build_region_masks(image.shape, result.landmarks)
    eye_points = _pts(result.landmarks, EYE_INDICES).astype(int)
    x0, x1 = int(eye_points[:, 0].min()), int(eye_points[:, 0].max())
    y0, y1 = int(eye_points[:, 1].min()), int(eye_points[:, 1].max())
    reflected = image.copy()
    reflected[max(y0 - 4, 0) : y1 + 18, x0 : x1 + 1] = (40, 220, 220)
    for x in range(x0, x1 + 1, 5):
        reflected[max(y0 - 4, 0) : y1 + 18, x : x + 2] = (5, 5, 5)

    refined, diagnostics = refine_masks_for_capture(
        reflected, masks, result.landmarks
    )

    assert diagnostics["eyewear_reflection_detected"]
    assert diagnostics["eye_zone_reflection_ratio"] > 0
    assert diagnostics["eyewear_exclusion_applied"]
    assert diagnostics["eyewear_excluded_fraction"] > 0
    assert diagnostics["eyewear_excluded_fraction"] < 0.30
    assert np.count_nonzero(refined["left_cheek"]) <= np.count_nonzero(
        masks["left_cheek"]
    )
    assert np.count_nonzero(refined["right_cheek"]) <= np.count_nonzero(
        masks["right_cheek"]
    )


def test_capture_refinement_does_not_flag_unmodified_face_as_eyewear():
    image = _load("face_astronaut.png")
    result = detect_face_landmarks(image)
    masks = build_region_masks(image.shape, result.landmarks)

    _, diagnostics = refine_masks_for_capture(
        image, masks, result.landmarks
    )

    assert not diagnostics["eyewear_reflection_detected"]
    assert not diagnostics["eyewear_exclusion_applied"]


def test_capture_refinement_reduces_one_cheek_for_angled_pose():
    image = _load("face_astronaut.png")
    result = detect_face_landmarks(image)
    landmarks = list(result.landmarks)
    nose_x, nose_y = landmarks[1]
    # Move the nose toward one eye so landmark geometry identifies a
    # foreshortened side without requiring another test fixture.
    left_eye_x, _ = landmarks[33]
    landmarks[1] = (left_eye_x + 0.2 * (nose_x - left_eye_x), nose_y)
    masks = build_region_masks(image.shape, landmarks)

    refined, diagnostics = refine_masks_for_capture(
        image, masks, landmarks, pose_asymmetry=0.30
    )

    reduced = diagnostics["pose_reduced_region"]
    assert reduced in {"left_cheek", "right_cheek"}
    assert np.count_nonzero(refined[reduced]) < np.count_nonzero(masks[reduced])
