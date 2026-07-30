"""Shared end-to-end ShadeSense image analysis orchestration.

The Streamlit UI and offline evaluation harness both call this module so a
benchmark result exercises the same CV, color, matching, and confidence logic
that a user sees locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.capture_uncertainty import analyze_capture_uncertainty
from src.color_correction import (
    apply_mild_color_correction,
    correction_settings_for_lighting,
)
from src.confidence import build_quality_report, compute_confidence
from src.config import TOP_K_SHADES
from src.extraction_quality import build_extraction_quality_report
from src.extraction_selection import run_dual_extraction
from src.face_detection import FaceDetectionResult, detect_face_landmarks
from src.image_quality import analyze_image_quality
from src.input_validation import validate_human_subject, validate_image_content
from src.lighting_quality import analyze_lighting_quality
from src.lighting_sensitivity import analyze_lighting_sensitivity
from src.recommendation_readiness import build_recommendation_readiness
from src.region_masks import build_region_masks, refine_masks_for_capture
from src.shade_matcher import match_shades

DEFAULT_MAX_ANALYSIS_SIDE = 1600


@dataclass
class ImageAnalysisResult:
    """Complete analysis state for one decoded RGB image."""

    success: bool
    image_rgb: np.ndarray
    source_shape: tuple[int, ...]
    analysis_scale: float
    image_color_metadata: dict
    face_result: Any
    global_lighting_quality: Any
    provisional_corrected_rgb: np.ndarray
    input_validation: dict = field(default_factory=dict)
    provisional_image_quality: Any = None
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    mask_capture_diagnostics: dict = field(default_factory=dict)
    image_quality: Any = None
    lighting_quality: Any = None
    corrected_rgb: np.ndarray | None = None
    correction_notes: list[str] = field(default_factory=list)
    extraction_selection: Any = None
    skin_result: Any = None
    lighting_sensitivity: Any = None
    capture_uncertainty: Any = None
    extraction_quality_report: dict = field(default_factory=dict)
    readiness_matches: list = field(default_factory=list)
    recommendation_readiness: Any = None
    matches: list = field(default_factory=list)
    quality_report: Any = None
    visualization_rgb: np.ndarray | None = None
    visual_source_label: str = "Original image"
    error: str | None = None


def normalize_analysis_resolution(
    image_rgb: np.ndarray,
    max_side: int = DEFAULT_MAX_ANALYSIS_SIDE,
) -> tuple[np.ndarray, float]:
    """Downscale large camera images deterministically for stable runtime."""
    image_rgb = np.asarray(image_rgb)
    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image_rgb, 1.0
    scale = float(max_side / longest)
    resized = cv2.resize(
        image_rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def analyze_rgb_image(
    image_rgb: np.ndarray,
    catalog_df: pd.DataFrame | None = None,
    *,
    extraction_mode: str = "auto",
    top_k: int = TOP_K_SHADES,
    max_analysis_side: int = DEFAULT_MAX_ANALYSIS_SIDE,
    image_color_metadata: dict | None = None,
) -> ImageAnalysisResult:
    """Run the full single-image pipeline used by ShadeSense AI.

    Args:
        image_rgb: uint8 RGB image array.
        catalog_df: Validated catalog. When omitted, extraction still runs but
            recommendation matching/confidence is skipped.
        extraction_mode: Existing dual-extraction selection mode.
        top_k: Number of final recommendations.
        max_analysis_side: Standardized maximum image side. Larger camera
            images are downscaled with area resampling before CV analysis.
        image_color_metadata: Diagnostics supplied by the image decoder.
    """
    image_rgb = np.asarray(image_rgb)
    source_shape = tuple(image_rgb.shape)
    content_validation = validate_image_content(image_rgb)
    if not content_validation.valid:
        face = FaceDetectionResult(
            success=False,
            landmarks=None,
            face_count=0,
            image_shape=source_shape,
            error=content_validation.message,
        )
        return ImageAnalysisResult(
            success=False,
            image_rgb=image_rgb,
            source_shape=source_shape,
            analysis_scale=1.0,
            image_color_metadata=dict(image_color_metadata or {}),
            face_result=face,
            global_lighting_quality=None,
            provisional_corrected_rgb=image_rgb,
            input_validation={
                "content": content_validation.as_dict(),
                "human_subject": None,
            },
            visualization_rgb=image_rgb,
            error=content_validation.message,
        )

    image_rgb, analysis_scale = normalize_analysis_resolution(
        image_rgb,
        max_analysis_side,
    )
    global_lighting = analyze_lighting_quality(image_rgb)
    provisional_corrected, _ = apply_mild_color_correction(
        image_rgb,
        **correction_settings_for_lighting(global_lighting),
    )
    face = detect_face_landmarks(provisional_corrected)
    subject_validation = validate_human_subject(face, image_rgb.shape)
    validation_diagnostics = {
        "content": content_validation.as_dict(),
        "human_subject": subject_validation.as_dict(),
    }
    if not subject_validation.valid:
        face.success = False
        face.error = subject_validation.message
    result = ImageAnalysisResult(
        success=False,
        image_rgb=image_rgb,
        source_shape=source_shape,
        analysis_scale=analysis_scale,
        image_color_metadata=dict(image_color_metadata or {}),
        face_result=face,
        global_lighting_quality=global_lighting,
        provisional_corrected_rgb=provisional_corrected,
        input_validation=validation_diagnostics,
        visualization_rgb=image_rgb,
        error=face.error if not face.success else None,
    )
    if not face.success:
        return result

    provisional_image_quality = analyze_image_quality(image_rgb, face.landmarks)
    masks = build_region_masks(image_rgb.shape, face.landmarks)
    masks, mask_diagnostics = refine_masks_for_capture(
        image_rgb,
        masks,
        face.landmarks,
        pose_asymmetry=provisional_image_quality.pose_asymmetry,
    )
    image_quality = analyze_image_quality(
        image_rgb,
        face.landmarks,
        masks=masks,
    )
    lighting = analyze_lighting_quality(image_rgb, masks=masks)
    corrected, correction_notes = apply_mild_color_correction(
        image_rgb,
        **correction_settings_for_lighting(lighting),
    )
    selection = run_dual_extraction(
        image_rgb,
        corrected,
        masks,
        lighting,
        extraction_mode,
    )
    skin = selection.selected
    skin.capture_region_diagnostics = mask_diagnostics
    skin.extraction_quality_reasons.append(selection.reason)

    sensitivity_source = (
        image_rgb if selection.selected_source == "original" else corrected
    )
    lighting_sensitivity = analyze_lighting_sensitivity(
        sensitivity_source,
        masks,
        skin,
    )
    skin.lighting_sensitivity_labs = lighting_sensitivity.variant_labs
    skin.lighting_sensitivity_diagnostics = (
        lighting_sensitivity.as_diagnostics()
    )
    skin.warnings.extend(lighting_sensitivity.warnings)

    capture_uncertainty = analyze_capture_uncertainty(
        skin,
        lighting_quality=lighting,
        image_quality=image_quality,
    )
    skin.systematic_uncertainty_diagnostics = (
        capture_uncertainty.as_diagnostics()
    )
    skin.warnings.extend(capture_uncertainty.warnings)

    extraction_quality = build_extraction_quality_report(
        skin,
        image_quality=image_quality,
        lighting_quality=lighting,
        extraction_selection=selection,
        face_result=face,
    )

    readiness_matches = []
    readiness = None
    matches = []
    quality_report = None
    if catalog_df is not None and skin.success:
        matching_lab = (
            skin.foundation_target_lab
            if skin.foundation_target_active
            else skin.lab
        )
        readiness_matches = match_shades(
            np.asarray(matching_lab),
            catalog_df,
            top_k=top_k,
            uncertainty_labs=skin.bootstrap_labs,
            lighting_sensitivity_labs=skin.lighting_sensitivity_labs,
        )
        readiness = build_recommendation_readiness(
            skin,
            extraction_quality,
            lighting,
            matches=readiness_matches,
        )
        # Matching is deterministic; retain a separate final list because the
        # confidence pass mutates ShadeMatch instances.
        matches = match_shades(
            np.asarray(matching_lab),
            catalog_df,
            top_k=top_k,
            uncertainty_labs=skin.bootstrap_labs,
            lighting_sensitivity_labs=skin.lighting_sensitivity_labs,
        )
        quality_report = build_quality_report(
            skin,
            face,
            matches,
            lighting,
        )
        matches = compute_confidence(
            matches,
            quality_report,
            readiness=readiness,
        )

    visualization_rgb = (
        image_rgb if selection.selected_source == "original" else corrected
    )
    result.success = bool(skin.success)
    result.error = None if skin.success else "Skin extraction failed."
    result.provisional_image_quality = provisional_image_quality
    result.masks = masks
    result.mask_capture_diagnostics = mask_diagnostics
    result.image_quality = image_quality
    result.lighting_quality = lighting
    result.corrected_rgb = corrected
    result.correction_notes = correction_notes
    result.extraction_selection = selection
    result.skin_result = skin
    result.lighting_sensitivity = lighting_sensitivity
    result.capture_uncertainty = capture_uncertainty
    result.extraction_quality_report = extraction_quality
    result.readiness_matches = readiness_matches
    result.recommendation_readiness = readiness
    result.matches = matches
    result.quality_report = quality_report
    result.visualization_rgb = visualization_rgb
    result.visual_source_label = (
        "Original image"
        if selection.selected_source == "original"
        else "Corrected image"
    )
    return result
