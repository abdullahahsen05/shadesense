import numpy as np
import pytest
from PIL import Image

from src.color_correction import apply_mild_color_correction
from src.config import PROJECT_ROOT
from src.face_detection import detect_face_landmarks
from src.region_masks import build_region_masks
from src.skin_extraction import (
    MIN_VALID_PIXELS_PER_REGION,
    _assert_skimage_lab_scale,
    extract_skin_tone,
    _filter_skin_pixels,
    _lab_distance,
    _stable_patch_medians,
    _to_lab,
)

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
    assert jawline.weight_multiplier < 1.0
    assert jawline.downweight_reason is not None
    assert "chin/neck shadow, contour, occlusion, or uneven lighting" in jawline.downweight_reason
    assert "facial hair" not in jawline.downweight_reason.lower()


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
    assert any("bright highlight patches were excluded" in r.lower() for r in skin.extraction_quality_reasons)


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
