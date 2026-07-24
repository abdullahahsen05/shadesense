import numpy as np

from src.lighting_quality import analyze_lighting_quality


def test_lighting_quality_scores_clean_image_high():
    image = np.full((100, 100, 3), 140, dtype=np.uint8)
    result = analyze_lighting_quality(image)
    assert result.score > 0.9
    assert result.warnings == []


def test_lighting_quality_flags_underexposure_overexposure_and_cast():
    dark = np.full((100, 100, 3), 35, dtype=np.uint8)
    assert analyze_lighting_quality(dark).underexposed

    bright = np.full((100, 100, 3), 230, dtype=np.uint8)
    assert analyze_lighting_quality(bright).overexposed

    cast = np.zeros((100, 100, 3), dtype=np.uint8)
    cast[:, :, 0] = 170
    cast[:, :, 1] = 95
    cast[:, :, 2] = 95
    result = analyze_lighting_quality(cast)
    assert result.color_cast
    assert result.score < 1.0


def test_lighting_quality_flags_uneven_shadow_contrast():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :50] = 35
    image[:, 50:] = 245
    result = analyze_lighting_quality(image)
    assert result.uneven_lighting
    assert result.strong_shadow_contrast
    assert result.score < 0.8


def test_lighting_quality_flags_broad_glossy_highlights():
    image = np.full((120, 120, 3), (72, 48, 36), dtype=np.uint8)
    image[20:80, 20:100] = (235, 230, 220)

    result = analyze_lighting_quality(image)

    assert result.strong_highlights
    assert result.score < 1.0
    assert any("glossy shine" in warning.lower() or "highlights" in warning.lower() for warning in result.warnings)


def test_face_region_lighting_ignores_bright_background():
    image = np.full((120, 160, 3), 248, dtype=np.uint8)
    image[30:100, 45:115] = (145, 105, 82)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[35:95, 50:110] = 255
    masks = {"left_cheek": mask.copy(), "right_cheek": mask.copy()}

    global_quality = analyze_lighting_quality(image)
    face_quality = analyze_lighting_quality(image, masks=masks)

    assert face_quality.using_face_regions
    assert not face_quality.overexposed
    assert face_quality.score > global_quality.score
    assert "facial skin regions" in face_quality.explanation


def test_face_region_lighting_reports_asymmetric_cheeks():
    image = np.full((100, 140, 3), 120, dtype=np.uint8)
    left = np.zeros((100, 140), dtype=np.uint8)
    right = np.zeros_like(left)
    left[30:75, 20:60] = 255
    right[30:75, 80:120] = 255
    image[left > 0] = (65, 50, 45)
    image[right > 0] = (180, 145, 125)

    quality = analyze_lighting_quality(
        image,
        masks={"left_cheek": left, "right_cheek": right},
    )

    assert quality.uneven_lighting
    assert quality.left_right_gap > 24
    assert set(quality.region_metrics) == {"left_cheek", "right_cheek"}


def test_shadowed_cheek_is_low_signal_without_treating_even_deep_skin_as_bad():
    dark_even = np.full((100, 140, 3), 55, dtype=np.uint8)
    left = np.zeros((100, 140), dtype=np.uint8)
    right = np.zeros_like(left)
    left[25:80, 15:60] = 255
    right[25:80, 80:125] = 255

    even = analyze_lighting_quality(
        dark_even,
        masks={"left_cheek": left, "right_cheek": right},
    )
    assert not even.low_signal

    split = dark_even.copy()
    split[left > 0] = 50
    split[right > 0] = 145
    uneven = analyze_lighting_quality(
        split,
        masks={"left_cheek": left, "right_cheek": right},
    )
    assert uneven.low_signal
    assert uneven.recapture_recommended
    assert uneven.score < even.score


def test_continuous_lighting_subscores_distinguish_severity():
    left = np.zeros((100, 140), dtype=np.uint8)
    right = np.zeros_like(left)
    left[25:80, 15:60] = 255
    right[25:80, 80:125] = 255
    mild = np.full((100, 140, 3), 125, dtype=np.uint8)
    severe = mild.copy()
    mild[left > 0] = 105
    mild[right > 0] = 145
    severe[left > 0] = 55
    severe[right > 0] = 200

    mild_result = analyze_lighting_quality(
        mild, masks={"left_cheek": left, "right_cheek": right}
    )
    severe_result = analyze_lighting_quality(
        severe, masks={"left_cheek": left, "right_cheek": right}
    )

    assert severe_result.uniformity_score < mild_result.uniformity_score
    assert severe_result.score < mild_result.score
    assert "Subscores" in severe_result.explanation
