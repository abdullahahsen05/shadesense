"""Choose between original-image and corrected-image skin extraction."""

from dataclasses import dataclass

import numpy as np

from src.skin_extraction import SkinToneResult, extract_skin_tone


HIGH_LIGHTING_SCORE = 0.88
MIN_CLEAR_QUALITY_GAIN = 0.06
MIN_CLEAR_CONSISTENCY_GAIN = 0.08
MIN_CLEAR_VALID_RATIO_GAIN = 0.08
MIN_CLEAR_CONTAMINATION_DROP = 0.08
HIGH_LIGHTING_SIGNIFICANT_QUALITY_GAIN = 0.08
HIGH_LIGHTING_SIGNIFICANT_CONSISTENCY_GAIN = 0.12
MAX_SAFE_LAB_SHIFT = 12.0
MIN_CHROMA_PRESERVATION = 0.72
MIN_HIGH_LIGHTING_CHROMA_PRESERVATION = 0.95
MAX_HIGH_LIGHTING_LAB_SHIFT = 8.0


@dataclass
class ExtractionSelection:
    selected: SkinToneResult
    original: SkinToneResult
    corrected: SkinToneResult
    selected_source: str
    selection_mode: str
    reason: str
    rgb_difference: float
    lab_difference: float
    chroma_preservation_score: float


def _mean_region_shadow_highlight(result: SkinToneResult) -> float:
    regions = [
        region
        for region in result.region_results.values()
        if region.reliable and not region.excluded
    ]
    if not regions:
        return 1.0
    return float(np.mean([region.shadow_highlight_ratio for region in regions]))


def _chroma(lab) -> float:
    lab_arr = np.asarray(lab, dtype=np.float64)
    return float(np.linalg.norm(lab_arr[1:3]))


def choose_extraction_candidate(
    original: SkinToneResult,
    corrected: SkinToneResult,
    lighting_quality=None,
    selection_mode: str = "auto",
) -> ExtractionSelection:
    """Select the safer extraction candidate.

    Correction is only chosen when it clearly improves extraction reliability
    without excessive chroma loss or Lab shift.
    """
    rgb_difference = float(
        np.linalg.norm(np.asarray(original.rgb, dtype=np.float64) - np.asarray(corrected.rgb, dtype=np.float64))
    )
    lab_difference = float(
        np.linalg.norm(np.asarray(original.lab, dtype=np.float64) - np.asarray(corrected.lab, dtype=np.float64))
    )
    original_chroma = _chroma(original.lab)
    corrected_chroma = _chroma(corrected.lab)
    chroma_preservation = 1.0 if original_chroma < 1e-6 else float(np.clip(corrected_chroma / original_chroma, 0.0, 1.0))
    mode = str(selection_mode).lower().replace(" ", "_")

    if mode in {"force_original", "original"}:
        reason = "Original image was used because debug mode forced original extraction."
        return ExtractionSelection(original, original, corrected, "original", "force_original", reason, rgb_difference, lab_difference, chroma_preservation)
    if mode in {"force_corrected", "corrected"}:
        reason = "Corrected image was used because debug mode forced corrected extraction."
        return ExtractionSelection(corrected, original, corrected, "corrected", "force_corrected", reason, rgb_difference, lab_difference, chroma_preservation)

    if not original.success and corrected.success:
        reason = "Corrected image was used because original extraction did not produce a usable skin tone."
        return ExtractionSelection(corrected, original, corrected, "corrected", "auto", reason, rgb_difference, lab_difference, chroma_preservation)
    if original.success and not corrected.success:
        reason = "Original image color was preserved because correction did not produce a usable skin tone."
        return ExtractionSelection(original, original, corrected, "original", "auto", reason, rgb_difference, lab_difference, chroma_preservation)

    quality_gain = corrected.quality_score - original.quality_score
    consistency_gain = corrected.region_consistency - original.region_consistency
    valid_ratio_gain = corrected.avg_valid_pixel_ratio - original.avg_valid_pixel_ratio
    contamination_drop = _mean_region_shadow_highlight(original) - _mean_region_shadow_highlight(corrected)
    correction_improves = (
        quality_gain >= MIN_CLEAR_QUALITY_GAIN
        or consistency_gain >= MIN_CLEAR_CONSISTENCY_GAIN
        or valid_ratio_gain >= MIN_CLEAR_VALID_RATIO_GAIN
        or contamination_drop >= MIN_CLEAR_CONTAMINATION_DROP
    )

    excessive_shift = lab_difference > MAX_SAFE_LAB_SHIFT and chroma_preservation < 0.9
    desaturated = chroma_preservation < MIN_CHROMA_PRESERVATION
    if excessive_shift or desaturated:
        reason = "Original image color was preserved because correction did not improve extraction reliability."
        return ExtractionSelection(original, original, corrected, "original", "auto", reason, rgb_difference, lab_difference, chroma_preservation)

    lighting_score = float(getattr(lighting_quality, "score", 1.0))
    has_color_cast = bool(getattr(lighting_quality, "color_cast", False))
    high_quality_lighting = lighting_score >= HIGH_LIGHTING_SCORE and not has_color_cast
    if high_quality_lighting:
        significant_improvement = (
            quality_gain >= HIGH_LIGHTING_SIGNIFICANT_QUALITY_GAIN
            and consistency_gain >= HIGH_LIGHTING_SIGNIFICANT_CONSISTENCY_GAIN
            and chroma_preservation >= MIN_HIGH_LIGHTING_CHROMA_PRESERVATION
            and lab_difference <= MAX_HIGH_LIGHTING_LAB_SHIFT
        )
        if significant_improvement:
            reason = "Corrected image was used because it improved lighting consistency without excessive color shift."
            return ExtractionSelection(corrected, original, corrected, "corrected", "auto", reason, rgb_difference, lab_difference, chroma_preservation)
        reason = "Original image color was preserved because correction did not improve extraction reliability."
        return ExtractionSelection(original, original, corrected, "original", "auto", reason, rgb_difference, lab_difference, chroma_preservation)

    if correction_improves or (
        has_color_cast
        and (quality_gain >= 0.02 or consistency_gain >= 0.03 or valid_ratio_gain >= 0.03)
    ):
        reason = "Corrected image was used because it improved lighting consistency without excessive color shift."
        return ExtractionSelection(corrected, original, corrected, "corrected", "auto", reason, rgb_difference, lab_difference, chroma_preservation)

    reason = "Original image color was preserved because correction did not improve extraction reliability."
    return ExtractionSelection(original, original, corrected, "original", "auto", reason, rgb_difference, lab_difference, chroma_preservation)


def run_dual_extraction(
    original_rgb: np.ndarray,
    corrected_rgb: np.ndarray,
    masks: dict,
    lighting_quality=None,
    selection_mode: str = "auto",
) -> ExtractionSelection:
    original_result = extract_skin_tone(original_rgb, masks)
    corrected_result = extract_skin_tone(corrected_rgb, masks)
    return choose_extraction_candidate(original_result, corrected_result, lighting_quality, selection_mode)
