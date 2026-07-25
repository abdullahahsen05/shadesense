"""Choose between original-image and corrected-image skin extraction."""

from dataclasses import dataclass, field

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
MAX_AUTOMATIC_UNDERTONE_SHIFT = 7.0
MAX_AUTOMATIC_LIGHTNESS_SHIFT = 8.0
MAX_HIGHLIGHT_BRIGHTENING_SHIFT = 3.0


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
    lightness_shift: float = 0.0
    undertone_shift: float = 0.0
    safety_flags: list[str] = field(default_factory=list)


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
    lightness_shift = float(corrected.lab[0] - original.lab[0])
    undertone_shift = float(
        np.linalg.norm(
            np.asarray(corrected.lab[1:3], dtype=np.float64)
            - np.asarray(original.lab[1:3], dtype=np.float64)
        )
    )
    mode = str(selection_mode).lower().replace(" ", "_")

    def build_selection(
        selected,
        selected_source: str,
        resolved_mode: str,
        reason: str,
        safety_flags: list[str] | None = None,
    ) -> ExtractionSelection:
        return ExtractionSelection(
            selected=selected,
            original=original,
            corrected=corrected,
            selected_source=selected_source,
            selection_mode=resolved_mode,
            reason=reason,
            rgb_difference=rgb_difference,
            lab_difference=lab_difference,
            chroma_preservation_score=chroma_preservation,
            lightness_shift=lightness_shift,
            undertone_shift=undertone_shift,
            safety_flags=list(safety_flags or []),
        )

    if mode in {"force_original", "original"}:
        reason = "Original image was used because debug mode forced original extraction."
        return build_selection(original, "original", "force_original", reason)
    if mode in {"force_corrected", "corrected"}:
        reason = "Corrected image was used because debug mode forced corrected extraction."
        return build_selection(corrected, "corrected", "force_corrected", reason)

    if not original.success and corrected.success:
        reason = "Corrected image was used because original extraction did not produce a usable skin tone."
        return build_selection(corrected, "corrected", "auto", reason)
    if original.success and not corrected.success:
        reason = "Original image color was preserved because correction did not produce a usable skin tone."
        return build_selection(original, "original", "auto", reason)

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

    strong_highlights = bool(getattr(lighting_quality, "strong_highlights", False))
    face_highlight_ratio = float(
        getattr(lighting_quality, "face_highlight_ratio", 0.0)
    )
    safety_flags = []
    if (
        lightness_shift > MAX_HIGHLIGHT_BRIGHTENING_SHIFT
        and (strong_highlights or face_highlight_ratio > 0.02)
    ):
        safety_flags.append(
            "Automatic correction would brighten highlighted facial skin by "
            f"{lightness_shift:.1f} L*."
        )
    if abs(lightness_shift) > MAX_AUTOMATIC_LIGHTNESS_SHIFT:
        safety_flags.append(
            "Automatic correction would change skin lightness by "
            f"{abs(lightness_shift):.1f} L*."
        )
    if undertone_shift > MAX_AUTOMATIC_UNDERTONE_SHIFT:
        safety_flags.append(
            "Automatic correction would change the skin a*/b* undertone by "
            f"{undertone_shift:.1f} Lab units."
        )
    if safety_flags:
        reason = (
            "Original image color was preserved because automatic correction "
            "exceeded conservative skin-tone safety limits. "
            + " ".join(safety_flags)
        )
        return build_selection(
            original,
            "original",
            "auto",
            reason,
            safety_flags=safety_flags,
        )

    excessive_shift = lab_difference > MAX_SAFE_LAB_SHIFT and chroma_preservation < 0.9
    desaturated = chroma_preservation < MIN_CHROMA_PRESERVATION
    if excessive_shift or desaturated:
        reason = "Original image color was preserved because correction did not improve extraction reliability."
        return build_selection(original, "original", "auto", reason)

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
            return build_selection(corrected, "corrected", "auto", reason)
        reason = "Original image color was preserved because correction did not improve extraction reliability."
        return build_selection(original, "original", "auto", reason)

    if correction_improves or (
        has_color_cast
        and (quality_gain >= 0.02 or consistency_gain >= 0.03 or valid_ratio_gain >= 0.03)
    ):
        reason = "Corrected image was used because it improved lighting consistency without excessive color shift."
        return build_selection(corrected, "corrected", "auto", reason)

    reason = "Original image color was preserved because correction did not improve extraction reliability."
    return build_selection(original, "original", "auto", reason)


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
