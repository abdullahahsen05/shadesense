"""Deterministic readiness state for presenting shade recommendations."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RecommendationReadiness:
    state: str
    score: float
    confidence_cap: float
    summary: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_recommendation_readiness(
    skin_result,
    extraction_quality_report: dict,
    lighting_quality=None,
) -> RecommendationReadiness:
    extraction_score = float(extraction_quality_report.get("overall_score", 0.0))
    lighting_score = float(getattr(lighting_quality, "score", 1.0))
    uncertainty = getattr(skin_result, "uncertainty_diagnostics", {}) or {}
    radius = float(uncertainty.get("delta_e_radius_p90", 12.0))
    stability = float(uncertainty.get("stability_score", 45.0))
    sensitivity = (
        getattr(skin_result, "lighting_sensitivity_diagnostics", {}) or {}
    )
    sensitivity_score = float(sensitivity.get("score", 100.0))
    sensitivity_radius = float(sensitivity.get("delta_e_p90", 0.0))
    success = bool(getattr(skin_result, "success", False))

    combined = float(
        np.clip(
            0.50 * extraction_score
            + 0.15 * (lighting_score * 100.0)
            + 0.20 * stability
            + 0.15 * sensitivity_score,
            0.0,
            100.0,
        )
    )
    reasons = [
        f"Skin Extraction Quality {extraction_score:.0f}/100.",
        f"Face-aware lighting quality {lighting_score:.0%}.",
        f"Bootstrap uncertainty radius {radius:.1f} Delta E (90th percentile).",
        f"Lighting sensitivity {sensitivity_score:.0f}/100 with {sensitivity_radius:.1f} Delta E variation.",
    ]

    if (
        success
        and extraction_score >= 70.0
        and lighting_score >= 0.65
        and radius <= 6.0
        and sensitivity_score >= 70.0
        and sensitivity_radius <= 3.0
    ):
        return RecommendationReadiness(
            state="ready",
            score=combined,
            confidence_cap=0.93,
            summary="Recommendation evidence is ready for comparison with the catalog.",
            reasons=reasons,
        )
    if (
        success
        and extraction_score >= 50.0
        and radius <= 10.0
        and sensitivity_radius <= 6.0
    ):
        return RecommendationReadiness(
            state="caution",
            score=combined,
            confidence_cap=0.75,
            summary="Recommendations are usable with caution; image or extraction evidence is limited.",
            reasons=reasons,
            warnings=["Consider retaking the photo in soft, even daylight for a more stable ranking."],
        )
    return RecommendationReadiness(
        state="provisional",
        score=combined,
        confidence_cap=0.55,
        summary="Recommendations are provisional because the current image does not provide stable enough color evidence.",
        reasons=reasons,
        warnings=[
            "Retake the photo in soft daylight with both cheeks and the jawline visible. "
            "The Top 3 are shown as provisional candidates, not dependable final matches."
        ],
    )
