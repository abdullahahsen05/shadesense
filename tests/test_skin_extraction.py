import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.color_correction import apply_mild_color_correction
from src.config import PROJECT_ROOT
from src.face_detection import detect_face_landmarks
from src.region_masks import build_region_masks
from src.skin_extraction import (
    JAWLINE_UNSUPPORTED_WEIGHT,
    MIN_VALID_PIXELS_PER_REGION,
    RegionSkinResult,
    _aggregate_patch_candidates,
    _assert_skimage_lab_scale,
    _bootstrap_patch_uncertainty,
    extract_skin_tone,
    _filter_skin_pixels,
    _lab_distance,
    _stable_patch_medians,
    _to_lab,
)
from src.shade_matcher import match_shades

SAMPLES = PROJECT_ROOT / "data" / "sample_images"


def _extract_for(name):
    img = np.array(Image.open(SAMPLES / name).convert("RGB"))
    corrected, _ = apply_mild_color_correction(img)
    result = detect_face_landmarks(corrected)
    masks = build_region_masks(corrected.shape, result.landmarks)
    return extract_skin_tone(corrected, masks)


def _synthetic_scene(forehead_rgb, left_cheek_rgb, right_cheek_rgb, jawline_rgb):
    """Build a flat-color synthetic image + masks so each region's extracted
    color is exactly the color assigned to it (deterministic, no detector
    involved) — used to test exclusion/down-weight thresholds precisely."""
    h, w = 100, 100
    image = np.zeros((h, w, 3), dtype=np.uint8)
    masks = {name: np.zeros((h, w), dtype=np.uint8) for name in
              ["forehead", "left_cheek", "right_cheek", "jawline"]}

    image[0:20, :] = forehead_rgb
    masks["forehead"][0:20, :] = 255

    image[30:50, 0:50] = left_cheek_rgb
    masks["left_cheek"][30:50, 0:50] = 255

    image[30:50, 50:100] = right_cheek_rgb
    masks["right_cheek"][30:50, 50:100] = 255

    image[60:80, :] = jawline_rgb
    masks["jawline"][60:80, :] = 255

    return image, masks


def test_extraction_succeeds_on_clear_face():
    skin = _extract_for("face_astronaut.png")
    assert skin.success
    assert all(0 <= v <= 255 for v in skin.rgb)
    assert 0.0 <= skin.quality_score <= 1.0


def test_uses_median_not_full_mean():
    skin = _extract_for("face_astronaut.png")
    for region in skin.region_results.values():
        if region.median_rgb is not None:
            assert isinstance(region.median_rgb, tuple)


def test_final_color_is_close_to_cheek_jaw_tones_not_hair():
    skin = _extract_for("face_astronaut.png")
    cheek_jaw_rgbs = [
        r.median_rgb
        for name, r in skin.region_results.items()
        if name != "forehead" and r.median_rgb is not None
    ]
    avg_cheek_jaw = np.mean(cheek_jaw_rgbs, axis=0)
    dist = np.linalg.norm(np.array(skin.rgb) - avg_cheek_jaw)
    assert dist < 30, "Final skin color strayed far from cheek/jaw consensus (possible hair contamination)"


def test_low_pixel_region_flagged_in_warnings_or_reliability():
    skin = _extract_for("face_astronaut.png")
    for region in skin.region_results.values():
        if region.valid_pixel_count < MIN_VALID_PIXELS_PER_REGION:
            assert not region.reliable


def test_color_correction_returns_notes_and_valid_image():
    img = np.array(Image.open(SAMPLES / "face_astronaut.png").convert("RGB"))
    corrected, notes = apply_mild_color_correction(img)
    assert corrected.shape == img.shape
    assert corrected.dtype == np.uint8
    assert len(notes) > 0


def test_lab_conversion_uses_cie_l_star_scale_and_rejects_opencv_scale():
    lab = _to_lab(np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8))
    assert 0.0 <= lab[:, 0].min() <= lab[:, 0].max() <= 100.0
    _assert_skimage_lab_scale(lab)
    import pytest

    with pytest.raises(ValueError):
        _assert_skimage_lab_scale(np.array([[255.0, 128.0, 128.0]]))
    with pytest.raises(ValueError):
        _lab_distance((50.0, 0.0, 0.0), (255.0, 0.0, 0.0))


# --- Exclusion / down-weight metadata (synthetic, deterministic) ---

CHEEK_RGB = (215, 180, 160)
SIMILAR_FOREHEAD_RGB = (210, 178, 158)  # close to cheeks -> should stay included
HAIR_CONTAMINATED_FOREHEAD_RGB = (90, 60, 40)  # far from cheeks -> should be excluded
DARK_JAWLINE_RGB = (140, 110, 95)  # notably darker than cheeks -> should be down-weighted
SIMILAR_JAWLINE_RGB = (212, 179, 159)  # close to cheeks -> full weight


def test_forehead_excluded_when_it_disagrees_strongly_with_cheeks():
    image, masks = _synthetic_scene(
        forehead_rgb=HAIR_CONTAMINATED_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    forehead = skin.region_results["forehead"]
    assert forehead.excluded is True
    assert forehead.exclusion_reason is not None
    assert forehead.role == "excluded"
    assert forehead.quality_label == "excluded"
    assert 0.0 <= forehead.quality_score <= 100.0
    assert "forehead" not in [n.lower() for n in skin.included_region_names]
    assert "forehead" in skin.excluded_region_names


def test_forehead_stays_included_when_it_agrees_with_cheeks():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    forehead = skin.region_results["forehead"]
    assert forehead.excluded is False
    assert forehead.exclusion_reason is None
    assert "forehead" in skin.included_region_names


def test_jawline_not_downweighted_when_darker_but_clean_and_stable():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=DARK_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    jawline = skin.region_results["jawline"]
    assert jawline.excluded is False
    assert jawline.weight_multiplier >= 0.75
    assert jawline.downweight_reason is None
    assert jawline.role == "supporting"
    assert jawline.quality_label != "excluded"
    assert "jawline" in skin.included_region_names


def test_jawline_full_weight_when_similar_to_cheeks():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    jawline = skin.region_results["jawline"]
    assert jawline.weight_multiplier == 1.0
    assert jawline.downweight_reason is None


def test_jawline_not_downweighted_when_lab_color_close_to_cheeks():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=(205, 174, 157),
    )
    skin = extract_skin_tone(image, masks)

    jawline = skin.region_results["jawline"]
    assert jawline.weight_multiplier == 1.0
    assert jawline.downweight_reason is None


def test_jawline_downweighted_when_contains_shadow_patches():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=CHEEK_RGB,
    )
    image[60:80, 0:45] = (65, 42, 32)
    skin = extract_skin_tone(image, masks)

    jawline = skin.region_results["jawline"]
    assert jawline.weight_multiplier <= 0.12
    assert jawline.role == "reduced"
    assert jawline.downweight_reason is not None
    assert "chin/neck shadow, contour, occlusion, or uneven lighting" in jawline.downweight_reason
    assert "facial hair" not in jawline.downweight_reason.lower()


def test_off_undertone_jawline_is_diagnostic_only_even_without_large_variance():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=(170, 95, 120),
    )

    skin = extract_skin_tone(image, masks)
    jawline = skin.region_results["jawline"]

    assert jawline.weight_multiplier <= 0.12
    assert jawline.role == "reduced"
    assert "did not agree with either cheek" in jawline.downweight_reason
    assert skin.patch_voting_diagnostics["region_contributions"].get(
        "jawline",
        0.0,
    ) < 0.10


def test_clean_cheek_region_quality_beats_highlight_contaminated_cheek():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    image[30:50, 50:76] = (245, 242, 235)

    skin = extract_skin_tone(image, masks)
    left = skin.region_results["left_cheek"]
    right = skin.region_results["right_cheek"]

    assert left.quality_score > right.quality_score
    assert right.highlight_patches_rejected > 0 or right.specular_highlight_detected
    assert any("highlight" in warning.lower() for warning in right.quality_warnings)


def test_region_quality_scores_stay_in_zero_to_one_hundred_range():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    for region in skin.region_results.values():
        assert 0.0 <= region.quality_score <= 100.0
        assert region.quality_label in {"excellent", "good", "fair", "poor", "excluded"}
        assert region.role in {"trusted", "supporting", "reduced", "excluded"}


def test_asymmetric_cheek_valid_area_warns_without_excluding_cheeks():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    masks["right_cheek"][:, :] = 0
    masks["right_cheek"][30:36, 50:100] = 255

    skin = extract_skin_tone(image, masks)

    assert skin.cheek_area_balance < 0.45
    assert any("cheek area imbalance" in w.lower() for w in skin.warnings)
    assert skin.region_results["left_cheek"].excluded is False
    assert skin.region_results["right_cheek"].excluded is False
    assert skin.region_results["right_cheek"].weight_multiplier < 1.0
    assert "pose or partial visibility" in skin.region_results["right_cheek"].downweight_reason


def test_best_patch_extraction_rejects_noisy_high_variance_patches():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
    mask = np.ones((80, 80), dtype=np.uint8) * 255
    stable_rgb = np.array([180, 135, 105], dtype=np.uint8)
    image[0:18, 0:18] = stable_rgb
    image[0:18, 24:42] = stable_rgb + np.array([2, 1, 0], dtype=np.uint8)

    patch_rgb, patch_lab, stable_count, _ = _stable_patch_medians(image, mask)

    assert stable_count >= 2
    assert len(patch_lab) >= 2
    assert np.linalg.norm(np.median(patch_rgb, axis=0) - stable_rgb) < 8


def test_best_patch_extraction_falls_back_safely_when_too_few_patches_survive():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    masks["left_cheek"][:, :] = 0
    masks["left_cheek"][30:40, 0:12] = 255

    skin = extract_skin_tone(image, masks)

    left = skin.region_results["left_cheek"]
    assert left.patch_fallback_used is True
    assert left.median_rgb is not None
    assert skin.success


@pytest.mark.parametrize(
    "rgb",
    [
        (245, 220, 200),  # fair
        (190, 145, 112),  # medium
        (150, 98, 70),  # tan
        (70, 45, 34),  # deep
        (38, 24, 18),  # rich-deep
    ],
)
def test_skin_filter_keeps_valid_tones_across_depths(rgb):
    skin_pixels = np.tile(np.array([rgb], dtype=np.uint8), (500, 1))
    shadow = np.tile(np.array([[5, 4, 4]], dtype=np.uint8), (50, 1))
    pixels = np.vstack([skin_pixels, shadow])

    valid_rgb, _ = _filter_skin_pixels(pixels)

    assert len(valid_rgb) >= 450
    assert np.linalg.norm(np.median(valid_rgb, axis=0) - np.array(rgb)) < 8


def test_adaptive_dark_filter_keeps_very_deep_skin_tones():
    very_deep_skin = np.tile(np.array([[38, 24, 18]], dtype=np.uint8), (500, 1))
    true_shadow = np.tile(np.array([[2, 2, 2]], dtype=np.uint8), (40, 1))
    pixels = np.vstack([very_deep_skin, true_shadow])

    valid_rgb, _ = _filter_skin_pixels(pixels)

    assert len(valid_rgb) >= 450
    assert np.median(valid_rgb[:, 0]) >= 35


def test_stable_patch_extraction_rejects_shadow_and_highlight_patches():
    image = np.full((80, 80, 3), (180, 135, 105), dtype=np.uint8)
    mask = np.ones((80, 80), dtype=np.uint8) * 255
    image[0:24, 0:24] = (15, 10, 8)
    image[56:80, 56:80] = (250, 246, 238)
    patch_rgb, _, stable_count, stats = _stable_patch_medians(
        image, mask, region_luminance_bounds=(35.0, 75.0)
    )

    assert stable_count >= 2
    assert stats["shadow_patches_rejected"] > 0
    assert stats["highlight_patches_rejected"] > 0
    assert np.linalg.norm(np.median(patch_rgb, axis=0) - np.array([180, 135, 105])) < 10


def test_deep_skin_with_bright_highlight_patches_does_not_become_too_light():
    image, masks = _synthetic_scene(
        forehead_rgb=(72, 48, 36),
        left_cheek_rgb=(82, 54, 40),
        right_cheek_rgb=(82, 54, 40),
        jawline_rgb=(60, 38, 30),
    )
    image[30:50, 0:22] = (210, 205, 195)
    image[30:50, 50:72] = (210, 205, 195)

    skin = extract_skin_tone(image, masks)

    assert skin.lab[0] < 40
    assert skin.depth_estimate in {"deep", "rich-deep"}
    assert skin.patch_voting_diagnostics["used"] is True
    assert any("bright highlight patches were excluded" in r.lower() for r in skin.extraction_quality_reasons)


def test_rich_deep_skin_with_broad_highlights_does_not_shift_too_light():
    image, masks = _synthetic_scene(
        forehead_rgb=(70, 45, 34),
        left_cheek_rgb=(72, 48, 36),
        right_cheek_rgb=(72, 48, 36),
        jawline_rgb=(48, 31, 24),
    )
    image[0:20, :] = (155, 145, 132)
    image[30:50, 0:25] = (180, 172, 160)
    image[30:50, 50:75] = (180, 172, 160)

    skin = extract_skin_tone(image, masks)

    assert skin.lab[0] < 35
    assert skin.depth_estimate == "rich-deep"
    assert any("shine does not make the foundation target too light" in w.lower() for w in skin.warnings)


def test_rich_deep_highlights_create_deeper_foundation_target_than_measured_tone():
    image, masks = _synthetic_scene(
        forehead_rgb=(70, 45, 34),
        left_cheek_rgb=(72, 48, 36),
        right_cheek_rgb=(72, 48, 36),
        jawline_rgb=(48, 31, 24),
    )
    image[0:20, :] = (155, 145, 132)
    image[30:50, 0:25] = (180, 172, 160)
    image[30:50, 50:75] = (180, 172, 160)

    skin = extract_skin_tone(image, masks)

    assert skin.foundation_target_active is True
    assert skin.foundation_target_lab[0] < skin.lab[0]
    assert skin.foundation_target_rgb != skin.rgb
    assert skin.foundation_target_diagnostics["criteria"]["lower_face_reliable"] is True
    assert "adjusted slightly deeper" in skin.foundation_target_reason.lower()


def test_clean_even_lighting_keeps_foundation_target_equal_to_measured_tone():
    image, masks = _synthetic_scene(
        forehead_rgb=(70, 45, 34),
        left_cheek_rgb=(72, 48, 36),
        right_cheek_rgb=(72, 48, 36),
        jawline_rgb=(68, 45, 34),
    )

    skin = extract_skin_tone(image, masks)

    assert skin.foundation_target_active is False
    assert skin.foundation_target_lab == skin.lab
    assert skin.foundation_target_rgb == skin.rgb


def test_shadow_contaminated_jawline_does_not_force_foundation_target_darker():
    image, masks = _synthetic_scene(
        forehead_rgb=(70, 45, 34),
        left_cheek_rgb=(72, 48, 36),
        right_cheek_rgb=(72, 48, 36),
        jawline_rgb=(20, 12, 10),
    )
    image[0:20, :] = (155, 145, 132)
    image[30:50, 0:25] = (180, 172, 160)
    image[30:50, 50:75] = (180, 172, 160)

    skin = extract_skin_tone(image, masks)

    assert skin.foundation_target_active is False
    assert skin.foundation_target_lab == skin.lab
    assert skin.foundation_target_diagnostics["criteria"]["lower_face_reliable"] is False


def test_depth_safe_foundation_target_prefers_similar_deeper_shade():
    image, masks = _synthetic_scene(
        forehead_rgb=(70, 45, 34),
        left_cheek_rgb=(72, 48, 36),
        right_cheek_rgb=(72, 48, 36),
        jawline_rgb=(48, 31, 24),
    )
    image[0:20, :] = (155, 145, 132)
    image[30:50, 0:25] = (180, 172, 160)
    image[30:50, 50:75] = (180, 172, 160)
    skin = extract_skin_tone(image, masks)

    measured_l, a, b = skin.lab
    target_l = skin.foundation_target_lab[0]
    catalog = pd.DataFrame(
        {
            "shade_id": ["too_light", "deeper"],
            "brand": ["Test", "Test"],
            "product": ["Base", "Base"],
            "shade_name": ["Too Light", "Similar Deeper"],
            "hex": ["#6B554A", "#4F3B32"],
            "r": [107, 79],
            "g": [85, 59],
            "b": [74, 50],
            "lab_l": [measured_l, target_l + 0.3],
            "lab_a": [a, a],
            "lab_b": [b, b],
            "depth": ["deep", "rich-deep"],
        }
    )

    matches = match_shades(np.array(skin.foundation_target_lab), catalog, top_k=2)

    assert skin.foundation_target_active is True
    assert matches[0].shade_name == "Similar Deeper"


def test_highlighted_forehead_is_strongly_downweighted():
    image, masks = _synthetic_scene(
        forehead_rgb=(220, 214, 205),
        left_cheek_rgb=(85, 56, 42),
        right_cheek_rgb=(85, 56, 42),
        jawline_rgb=(58, 38, 30),
    )
    skin = extract_skin_tone(image, masks)
    forehead = skin.region_results["forehead"]

    assert forehead.specular_highlight_detected or forehead.excluded
    assert forehead.excluded or forehead.weight_multiplier <= 0.2


def test_patch_voting_shadow_patch_does_not_pull_final_lab_too_dark():
    image, masks = _synthetic_scene(
        forehead_rgb=(152, 105, 78),
        left_cheek_rgb=(150, 100, 72),
        right_cheek_rgb=(150, 100, 72),
        jawline_rgb=(140, 92, 68),
    )
    image[30:50, 0:22] = (18, 12, 10)
    image[60:80, 0:30] = (20, 13, 10)

    skin = extract_skin_tone(image, masks)

    assert skin.patch_voting_diagnostics["used"] is True
    assert skin.patch_voting_diagnostics["outlier_patches_rejected"] > 0
    assert 33 < skin.lab[0] < 55


def test_valid_darker_jawline_with_low_variance_is_not_heavily_downweighted():
    image, masks = _synthetic_scene(
        forehead_rgb=(88, 60, 45),
        left_cheek_rgb=(88, 60, 45),
        right_cheek_rgb=(88, 60, 45),
        jawline_rgb=(62, 41, 31),
    )
    skin = extract_skin_tone(image, masks)

    jawline = skin.region_results["jawline"]
    assert jawline.weight_multiplier >= 0.75
    assert jawline.downweight_reason is None
    assert any("jawline/lower-cheek patches supported" in r.lower() for r in skin.extraction_quality_reasons)


def test_valid_darker_jawline_patch_can_support_depth_when_reliable():
    image, masks = _synthetic_scene(
        forehead_rgb=(88, 60, 45),
        left_cheek_rgb=(88, 60, 45),
        right_cheek_rgb=(88, 60, 45),
        jawline_rgb=(56, 36, 28),
    )
    skin = extract_skin_tone(image, masks)
    cheek_l = np.mean(
        [
            skin.region_results["left_cheek"].median_lab[0],
            skin.region_results["right_cheek"].median_lab[0],
        ]
    )

    assert skin.patch_voting_diagnostics["used"] is True
    assert skin.region_results["jawline"].role == "supporting"
    assert skin.lab[0] < cheek_l
    assert any("stable diffuse patches" in r.lower() for r in skin.extraction_quality_reasons)


@pytest.mark.parametrize(
    "rgb",
    [
        (245, 220, 200),
        (190, 145, 112),
        (150, 98, 70),
        (70, 45, 34),
        (38, 24, 18),
    ],
)
def test_mid_tone_patch_selection_across_skin_depths(rgb):
    image = np.full((96, 96, 3), rgb, dtype=np.uint8)
    mask = np.ones((96, 96), dtype=np.uint8) * 255
    image[0:24, 0:24] = np.maximum(np.array(rgb) - 30, 0).astype(np.uint8)
    image[72:96, 72:96] = np.minimum(np.array(rgb) + 70, 255).astype(np.uint8)

    patch_rgb, _, stable_count, stats = _stable_patch_medians(
        image, mask, region_luminance_bounds=(5.0, 95.0, 20.0, 80.0)
    )

    assert stable_count >= 2
    assert stats["midtone_patch_count"] >= 1
    assert np.linalg.norm(np.median(patch_rgb, axis=0) - np.array(rgb)) < 20


def test_possible_makeup_highlight_influence_warns_and_downweights_cheek():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    image[30:50, 0:22] = (235, 75, 130)

    skin = extract_skin_tone(image, masks)
    left = skin.region_results["left_cheek"]

    assert left.makeup_influence_detected
    assert left.weight_multiplier < 1.0
    assert any("possible makeup/highlight influence detected" in w.lower() for w in skin.warnings)


def test_patch_voting_rejects_lab_outlier_patch():
    cheek = RegionSkinResult(
        "left_cheek",
        400,
        360,
        0.9,
        (150, 100, 72),
        (45.0, 13.0, 20.0),
        True,
        stable_patch_count=4,
        stable_patch_rgbs=[(150, 100, 72), (152, 101, 73), (149, 99, 71), (60, 20, 20)],
        stable_patch_labs=[(45.0, 13.0, 20.0), (45.4, 13.1, 20.2), (44.8, 12.9, 19.8), (18.0, 28.0, 12.0)],
        stable_patch_quality_scores=[0.95, 0.95, 0.95, 0.9],
        stable_patch_midtone_flags=[True, True, True, True],
        quality_score=95.0,
        role="trusted",
    )
    jawline = RegionSkinResult(
        "jawline",
        400,
        340,
        0.85,
        (142, 94, 68),
        (42.0, 12.0, 18.0),
        True,
        stable_patch_count=3,
        stable_patch_rgbs=[(142, 94, 68), (141, 93, 68), (143, 95, 69)],
        stable_patch_labs=[(42.0, 12.0, 18.0), (41.8, 12.1, 18.1), (42.3, 11.9, 17.9)],
        stable_patch_quality_scores=[0.9, 0.9, 0.9],
        stable_patch_midtone_flags=[True, True, True],
        quality_score=88.0,
        role="supporting",
    )

    _, final_lab, diagnostics = _aggregate_patch_candidates([cheek, jawline])

    assert diagnostics["used"] is True
    assert diagnostics["outlier_patches_rejected"] >= 1
    assert final_lab[0] > 38


def test_patch_voting_falls_back_safely_if_too_few_stable_patches_exist():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    for name in masks:
        masks[name][:] = 0
    masks["left_cheek"][30:42, 0:12] = 255
    masks["right_cheek"][30:42, 50:62] = 255

    skin = extract_skin_tone(image, masks)

    assert skin.success
    assert skin.patch_voting_diagnostics["used"] is False
    assert "fallback" in skin.patch_voting_diagnostics["fallback_reason"].lower()


def test_patch_voting_final_lab_and_rgb_are_valid_ranges():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    assert all(0 <= channel <= 255 for channel in skin.rgb)
    assert 0.0 <= skin.lab[0] <= 100.0
    assert skin.patch_voting_diagnostics["stable_patches_available"] >= skin.patch_voting_diagnostics["stable_patches_used"]
    assert "dominant_region_contribution" in skin.patch_voting_diagnostics


def test_stable_regions_produce_high_region_stability_score():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)
    stability = skin.stability_diagnostics

    assert 0.0 <= stability["stability_score"] <= 100.0
    assert stability["stability_score"] >= 85
    assert stability["stability_label"] == "excellent"
    assert stability["unstable_regions"] == []
    assert "removing any one trusted region" in stability["summary"]


def test_outlier_jawline_is_reduced_before_it_can_create_false_instability():
    image, masks = _synthetic_scene(
        forehead_rgb=HAIR_CONTAMINATED_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=(170, 95, 120),
    )
    skin = extract_skin_tone(image, masks)
    stability = skin.stability_diagnostics

    jawline = skin.region_results["jawline"]
    assert jawline.weight_multiplier <= JAWLINE_UNSUPPORTED_WEIGHT
    assert stability["stability_score"] >= 50
    assert stability["most_influential_region"] == max(
        stability["leave_one_out_delta_e"],
        key=stability["leave_one_out_delta_e"].get,
    )
    assert (
        "diagnostic-only support" in jawline.downweight_reason
        or "did not corroborate" in " ".join(jawline.quality_reasons)
    )


def test_dominant_clean_cheek_reports_limited_support_not_contradiction():
    image, masks = _synthetic_scene(
        forehead_rgb=(225, 190, 170),
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=(245, 220, 205),
        jawline_rgb=(185, 145, 122),
    )
    masks["right_cheek"][:, 62:] = 0

    skin = extract_skin_tone(image, masks)
    stability = skin.stability_diagnostics

    assert stability["support_mode"] in {
        "limited_independent_support",
        "agreement",
    }
    if stability["support_mode"] == "limited_independent_support":
        assert "not direct evidence" in stability["summary"]
    assert "influence_adjusted_leave_one_out_delta_e" in stability


def test_region_reliability_score_reflects_patch_and_valid_pixel_quality():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    left = skin.region_results["left_cheek"]
    assert 0.0 <= left.reliability_score <= 1.0
    assert left.reliability_score > 0.6
    assert left.role == "trusted"
    assert left.quality_score > 60
    assert any("reliability score" in r.lower() for r in skin.extraction_quality_reasons)


def test_extraction_quality_reasons_mention_only_actual_factors():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)
    text = " ".join(skin.extraction_quality_reasons).lower()

    assert "included regions" in text
    assert "stable patch" in text
    assert "not-used regions" not in text
    assert "reduced-weight regions" not in text
    assert "fewer than 3 regions" not in text


def test_excluded_region_status_label_never_says_reliable():
    image, masks = _synthetic_scene(
        forehead_rgb=HAIR_CONTAMINATED_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=SIMILAR_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    forehead = skin.region_results["forehead"]
    assert forehead.reliable is True  # had enough valid pixels
    assert forehead.status_label == "excluded"
    assert "reliable" not in forehead.status_label
    assert forehead.status_reason == forehead.exclusion_reason


def test_included_region_status_label_reflects_downweight():
    image, masks = _synthetic_scene(
        forehead_rgb=SIMILAR_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=CHEEK_RGB,
    )
    image[60:80, 0:45] = (65, 42, 32)
    skin = extract_skin_tone(image, masks)

    jawline = skin.region_results["jawline"]
    assert jawline.status_label == "included (reduced weight)"
    assert jawline.status_reason == jawline.downweight_reason

    left_cheek = skin.region_results["left_cheek"]
    assert left_cheek.status_label == "included"
    assert left_cheek.status_reason is None


def test_included_and_excluded_region_name_lists_are_disjoint_and_complete():
    image, masks = _synthetic_scene(
        forehead_rgb=HAIR_CONTAMINATED_FOREHEAD_RGB,
        left_cheek_rgb=CHEEK_RGB,
        right_cheek_rgb=CHEEK_RGB,
        jawline_rgb=DARK_JAWLINE_RGB,
    )
    skin = extract_skin_tone(image, masks)

    assert set(skin.included_region_names) & set(skin.excluded_region_names) == set()
    assert set(skin.included_region_names) == {"left_cheek", "right_cheek", "jawline"}
    assert set(skin.excluded_region_names) == {"forehead"}


def test_no_reliable_cheek_skips_forehead_jawline_rules_gracefully():
    """With no reliable cheek anchor, forehead/jawline rules can't be
    evaluated and must not crash or spuriously exclude/down-weight."""
    h, w = 100, 100
    image = np.zeros((h, w, 3), dtype=np.uint8)
    masks = {name: np.zeros((h, w), dtype=np.uint8) for name in
              ["forehead", "left_cheek", "right_cheek", "jawline"]}
    image[0:20, :] = HAIR_CONTAMINATED_FOREHEAD_RGB
    masks["forehead"][0:20, :] = 255
    image[60:80, :] = DARK_JAWLINE_RGB
    masks["jawline"][60:80, :] = 255
    # no cheek masks filled in at all

    skin = extract_skin_tone(image, masks)

    forehead = skin.region_results["forehead"]
    jawline = skin.region_results["jawline"]
    assert forehead.excluded is False
    assert jawline.weight_multiplier == 1.0


def test_patch_sampling_scales_with_region_resolution():
    small = np.full((80, 80, 3), (150, 105, 78), dtype=np.uint8)
    large = np.full((240, 240, 3), (150, 105, 78), dtype=np.uint8)
    small_mask = np.full((80, 80), 255, dtype=np.uint8)
    large_mask = np.full((240, 240), 255, dtype=np.uint8)

    _, _, _, small_stats = _stable_patch_medians(small, small_mask, "left_cheek")
    _, _, _, large_stats = _stable_patch_medians(large, large_mask, "left_cheek")

    assert large_stats["patch_size"] > small_stats["patch_size"]
    assert all(evidence.region == "left_cheek" for evidence in small_stats["patch_evidence"])
    assert all(evidence.size == small_stats["patch_size"] for evidence in small_stats["patch_evidence"])


def test_perceptual_consensus_reports_medoid_and_caps_region_influence():
    image, masks = _synthetic_scene(
        forehead_rgb=(155, 111, 82),
        left_cheek_rgb=(150, 105, 78),
        right_cheek_rgb=(148, 104, 77),
        jawline_rgb=(140, 96, 72),
    )

    skin = extract_skin_tone(image, masks)
    diagnostics = skin.patch_voting_diagnostics

    assert diagnostics["used"]
    assert "CIEDE2000 medoid" in diagnostics["consensus_method"]
    assert diagnostics["consensus_medoid_lab"] is not None
    assert diagnostics["outlier_threshold_delta_e"] >= 6.0
    contributions = diagnostics["region_contributions"]
    assert contributions.get("forehead", 0.0) <= 0.15 + 1e-6
    assert contributions.get("jawline", 0.0) <= 0.30 + 1e-6


def test_bootstrap_uncertainty_is_deterministic_and_tracks_patch_spread():
    stable_diag = {
        "retained_patch_labs": [
            (50.0, 12.0, 18.0),
            (50.3, 12.1, 18.1),
            (49.8, 11.9, 17.9),
            (50.1, 12.0, 18.2),
        ],
        "retained_patch_weights": [1.0, 1.0, 1.0, 1.0],
        "retained_patch_regions": [
            "left_cheek",
            "left_cheek",
            "right_cheek",
            "right_cheek",
        ],
    }
    varied_diag = {
        **stable_diag,
        "retained_patch_labs": [
            (44.0, 8.0, 13.0),
            (54.0, 16.0, 23.0),
            (45.0, 9.0, 14.0),
            (55.0, 17.0, 24.0),
        ],
    }

    samples_a, stable = _bootstrap_patch_uncertainty(
        stable_diag, (50.0, 12.0, 18.0)
    )
    samples_b, stable_again = _bootstrap_patch_uncertainty(
        stable_diag, (50.0, 12.0, 18.0)
    )
    _, varied = _bootstrap_patch_uncertainty(
        varied_diag, (50.0, 12.0, 18.0)
    )

    assert len(samples_a) == 96
    assert samples_a == samples_b
    assert stable == stable_again
    assert varied["delta_e_radius_p90"] > stable["delta_e_radius_p90"]
    assert varied["stability_score"] < stable["stability_score"]


def test_foundation_target_shift_is_capped_for_light_medium_skin():
    cheek = (200, 160, 130)
    image, masks = _synthetic_scene(
        forehead_rgb=cheek,
        left_cheek_rgb=cheek,
        right_cheek_rgb=cheek,
        jawline_rgb=(175, 135, 105),
    )
    image[0:20, :] = (245, 235, 225)
    image[30:50, 0:25] = (245, 235, 225)
    image[30:50, 50:75] = (245, 235, 225)

    skin = extract_skin_tone(image, masks)
    diagnostics = skin.foundation_target_diagnostics

    assert skin.depth_estimate == "light-medium"
    assert skin.foundation_target_active
    assert diagnostics["criteria"]["lower_face_reliable"]
    assert diagnostics["maximum_l_adjustment"] == 3.0
    assert diagnostics["l_adjustment"] <= 3.0
    assert "cheek-derived undertone was preserved" in skin.foundation_target_reason
