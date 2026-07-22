"""Formal skin extraction quality scoring.

This score describes how reliable the extracted skin color is. It is separate
from shade match confidence, which describes how clearly that extracted color
matches the catalog.
"""

from __future__ import annotations

import numpy as np


def _label(score: float) -> str:
    if score >= 88:
        return "excellent"
    if score >= 72:
        return "good"
    if score >= 50:
        return "fair"
    return "poor"


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _mean_region_quality(skin_result) -> float:
    regions = [
        region
        for region in getattr(skin_result, "region_results", {}).values()
        if getattr(region, "reliable", False) and not getattr(region, "excluded", False)
    ]
    if not regions:
        return 0.0
    region_quality = float(np.mean([getattr(region, "quality_score", 0.0) for region in regions]))
    valid_ratio = float(np.mean([getattr(region, "valid_ratio", 0.0) for region in regions])) * 100.0
    contamination = float(np.mean([getattr(region, "shadow_highlight_ratio", 0.0) for region in regions])) * 100.0
    return _clip_score(0.70 * region_quality + 0.25 * valid_ratio + 0.05 * (100.0 - contamination))


def _patch_stability_score(skin_result) -> float:
    patch_diag = getattr(skin_result, "patch_voting_diagnostics", {}) or {}
    if not patch_diag.get("used"):
        return 48.0
    used = float(patch_diag.get("stable_patches_used", 0))
    available = max(float(patch_diag.get("stable_patches_available", used)), 1.0)
    outliers = float(patch_diag.get("outlier_patches_rejected", 0))
    highlight = float(patch_diag.get("highlight_patches_rejected", 0))
    shadow = float(patch_diag.get("shadow_patches_rejected", 0))
    midtone = float(patch_diag.get("midtone_patches_used", 0))
    score = 55.0 + min(used, 10.0) * 3.0 + min(midtone, used) * 1.5
    score += 10.0 * min(used / available, 1.0)
    score -= outliers * 4.0 + min(highlight, 10.0) * 2.6 + min(shadow, 8.0) * 1.5
    return _clip_score(score)


def _lighting_safety_score(lighting_quality, extraction_selection) -> float:
    lighting_score = float(getattr(lighting_quality, "score", 1.0)) * 100.0
    chroma_score = float(getattr(extraction_selection, "chroma_preservation_score", 1.0)) * 100.0
    lab_diff = float(getattr(extraction_selection, "lab_difference", 0.0))
    lab_safety = _clip_score(100.0 - max(lab_diff - 4.0, 0.0) * 4.0)
    selected_source = getattr(extraction_selection, "selected_source", "original")
    color_preservation_bonus = 5.0 if selected_source == "original" and lighting_score >= 85 else 0.0
    return _clip_score(0.55 * lighting_score + 0.25 * chroma_score + 0.20 * lab_safety + color_preservation_bonus)


def _quality_sentence(report: dict) -> str:
    subscores = report["subscores"]
    positives = []
    limits = []
    if subscores["region_reliability"] >= 72:
        positives.append("trusted regions had reliable skin pixels")
    if subscores["patch_stability"] >= 72:
        positives.append("stable diffuse patches supported the final swatch")
    if subscores["lighting_safety"] >= 72:
        positives.append("color handling was safe")
    if subscores["region_stability"] < 70:
        limits.append("region stability was limited")
    if subscores["patch_stability"] < 60:
        limits.append("patch evidence was limited")
    if subscores["image_capture"] < 60:
        limits.append("image capture quality was limited")
    if not positives:
        positives.append("the app extracted a usable skin color")
    explanation = "The extraction is reliable because " + ", ".join(positives) + "."
    if limits:
        explanation += " Confidence was limited because " + ", ".join(limits) + "."
    return explanation


def build_extraction_quality_report(
    skin_result,
    image_quality=None,
    lighting_quality=None,
    extraction_selection=None,
    face_result=None,
) -> dict:
    """Combine extraction reliability signals into one demo-facing report."""
    image_capture = _clip_score(float(getattr(image_quality, "overall_score", 70.0)))
    region_reliability = _mean_region_quality(skin_result)
    patch_stability = _patch_stability_score(skin_result)
    lighting_safety = _lighting_safety_score(lighting_quality, extraction_selection)
    color_consistency = _clip_score(float(getattr(skin_result, "region_consistency", 0.0)) * 100.0)
    stability_diag = getattr(skin_result, "stability_diagnostics", {}) or {}
    region_stability = _clip_score(float(stability_diag.get("stability_score", 70.0)))

    face_warnings = getattr(face_result, "warnings", []) if face_result is not None else []
    if face_warnings:
        image_capture = _clip_score(image_capture - min(len(face_warnings) * 5.0, 15.0))

    subscores = {
        "image_capture": image_capture,
        "region_reliability": region_reliability,
        "patch_stability": patch_stability,
        "lighting_safety": lighting_safety,
        "color_consistency": color_consistency,
        "region_stability": region_stability,
    }
    overall = _clip_score(
        0.18 * image_capture
        + 0.22 * region_reliability
        + 0.18 * patch_stability
        + 0.15 * lighting_safety
        + 0.15 * color_consistency
        + 0.12 * region_stability
    )

    warnings: list[str] = []
    warnings.extend(getattr(image_quality, "warnings", []) if image_quality is not None else [])
    warnings.extend(getattr(lighting_quality, "warnings", []) if lighting_quality is not None else [])
    warnings.extend(stability_diag.get("warnings", []))
    if region_reliability < 60:
        warnings.append("Some skin regions had low extraction reliability.")
    if patch_stability < 60:
        warnings.append("Stable patch evidence was limited; region median fallback may be less robust.")
    if color_consistency < 55:
        warnings.append("Skin regions disagreed in color, which can reduce extraction reliability.")

    report = {
        "overall_score": overall,
        "label": _label(overall),
        "subscores": subscores,
        "reasons": [],
        "warnings": list(dict.fromkeys(warnings)),
    }
    report["reasons"] = [
        _quality_sentence(report),
        "Skin Extraction Quality measures input/extraction reliability, while Match confidence measures catalog-match clarity.",
    ]
    return report
