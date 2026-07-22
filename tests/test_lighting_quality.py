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
