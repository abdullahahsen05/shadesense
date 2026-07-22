from types import SimpleNamespace

import numpy as np

from src.color_correction import apply_mild_color_correction
from src.extraction_selection import choose_extraction_candidate
from src.skin_extraction import SkinToneResult


def _result(rgb, lab, quality=0.8, consistency=0.8, valid_ratio=0.8):
    return SkinToneResult(
        rgb=rgb,
        lab=lab,
        region_results={},
        quality_score=quality,
        region_consistency=consistency,
        avg_valid_pixel_ratio=valid_ratio,
        success=True,
    )


def test_color_correction_does_not_modify_original_image_in_place():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = (70, 45, 35)
    before = image.copy()

    corrected, _ = apply_mild_color_correction(image)

    assert np.array_equal(image, before)
    assert corrected is not image


def test_high_lighting_quality_prefers_original_if_correction_desaturates():
    original = _result((82, 52, 38), (32.0, 12.0, 17.0), quality=0.82, consistency=0.84)
    corrected = _result((72, 64, 60), (34.0, 2.0, 2.0), quality=0.88, consistency=0.90)
    lighting = SimpleNamespace(score=0.96, color_cast=False)

    selection = choose_extraction_candidate(original, corrected, lighting)

    assert selection.selected_source == "original"
    assert "Original image color was preserved" in selection.reason
    assert selection.chroma_preservation_score < 0.72


def test_high_lighting_quality_defaults_to_original_for_small_improvement():
    original = _result((120, 80, 58), (42.0, 10.0, 18.0), quality=0.84, consistency=0.86)
    corrected = _result((122, 82, 60), (43.0, 10.2, 18.2), quality=0.89, consistency=0.93)
    lighting = SimpleNamespace(score=0.97, color_cast=False)

    selection = choose_extraction_candidate(original, corrected, lighting)

    assert selection.selected_source == "original"
    assert selection.selection_mode == "auto"
    assert "Original image color was preserved" in selection.reason


def test_corrected_image_selected_when_it_improves_strong_color_cast():
    original = _result((130, 92, 70), (45.0, 6.0, 18.0), quality=0.70, consistency=0.68, valid_ratio=0.72)
    corrected = _result((124, 88, 67), (44.0, 7.0, 17.0), quality=0.78, consistency=0.78, valid_ratio=0.78)
    lighting = SimpleNamespace(score=0.78, color_cast=True)

    selection = choose_extraction_candidate(original, corrected, lighting)

    assert selection.selected_source == "corrected"
    assert "improved lighting consistency" in selection.reason


def test_force_original_mode_uses_original_extraction():
    original = _result((82, 52, 38), (32.0, 12.0, 17.0), quality=0.70)
    corrected = _result((90, 70, 62), (40.0, 4.0, 6.0), quality=0.95)

    selection = choose_extraction_candidate(original, corrected, selection_mode="force_original")

    assert selection.selected is original
    assert selection.selected_source == "original"
    assert selection.selection_mode == "force_original"


def test_force_corrected_mode_uses_corrected_extraction():
    original = _result((82, 52, 38), (32.0, 12.0, 17.0), quality=0.95)
    corrected = _result((90, 70, 62), (40.0, 4.0, 6.0), quality=0.40)

    selection = choose_extraction_candidate(original, corrected, selection_mode="force_corrected")

    assert selection.selected is corrected
    assert selection.selected_source == "corrected"
    assert selection.selection_mode == "force_corrected"


def test_lab_chroma_shift_guard_rejects_over_corrected_skin_tone():
    original = _result((160, 105, 75), (50.0, 15.0, 24.0), quality=0.80, consistency=0.80)
    corrected = _result((128, 126, 124), (58.0, 1.0, 1.0), quality=0.90, consistency=0.92)
    lighting = SimpleNamespace(score=0.70, color_cast=True)

    selection = choose_extraction_candidate(original, corrected, lighting)

    assert selection.selected_source == "original"
    assert selection.lab_difference > 12.0
    assert selection.chroma_preservation_score < 0.72
