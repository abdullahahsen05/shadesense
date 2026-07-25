"""Deterministic readiness state for presenting shade recommendations."""

from dataclasses import dataclass, field

import numpy as np

from src.readiness_calibration import load_readiness_thresholds


@dataclass(frozen=True)
class RecommendationReadiness:
    state: str
    score: float
    confidence_cap: float
    summary: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capture_readiness_score: float = 0.0
    shade_family_stability_score: float = 0.0
    exact_product_stability_score: float = 0.0


def build_recommendation_readiness(
    skin_result,
    extraction_quality_report: dict,
    lighting_quality=None,
    matches: list | None = None,
    thresholds=None,
) -> RecommendationReadiness:
    thresholds = thresholds or load_readiness_thresholds()
    extraction_score = float(extraction_quality_report.get("overall_score", 0.0))
    lighting_score = float(getattr(lighting_quality, "score", 1.0))
    uncertainty = getattr(skin_result, "uncertainty_diagnostics", {}) or {}
    capture_uncertainty = (
        getattr(skin_result, "systematic_uncertainty_diagnostics", {}) or {}
    )
    local_radius = float(uncertainty.get("delta_e_radius_p90", 12.0))
    radius = float(
        capture_uncertainty.get("total_delta_e_radius_p90", local_radius)
    )
    stability = float(uncertainty.get("stability_score", 45.0))
    sensitivity = (
        getattr(skin_result, "lighting_sensitivity_diagnostics", {}) or {}
    )
    sensitivity_score = float(sensitivity.get("score", 100.0))
    sensitivity_radius = float(sensitivity.get("delta_e_p90", 0.0))
    success = bool(getattr(skin_result, "success", False))
    low_signal = bool(getattr(lighting_quality, "low_signal", False))
    capture_regions = (
        getattr(skin_result, "capture_region_diagnostics", {}) or {}
    )
    eyewear_detected = bool(
        capture_regions.get("eyewear_reflection_detected", False)
    )
    rank_score = 100.0
    exact_rank_score = 100.0
    bootstrap_family_top3 = 1.0
    lighting_family_top3 = 1.0
    if matches:
        best = matches[0]
        bootstrap_family_top3 = float(
            getattr(best, "top3_family_stability", None)
            if getattr(best, "top3_family_stability", None) is not None
            else getattr(best, "top3_stability", 1.0) or 0.0
        )
        lighting_family_top3 = float(
            getattr(best, "lighting_top3_family_stability", None)
            if getattr(best, "lighting_top3_family_stability", None) is not None
            else getattr(best, "lighting_top3_stability", 1.0) or 0.0
        )
        bootstrap_family_top1 = float(
            getattr(best, "recommendation_family_stability", None)
            if getattr(best, "recommendation_family_stability", None) is not None
            else getattr(best, "recommendation_stability", 1.0) or 0.0
        )
        lighting_family_top1 = float(
            getattr(best, "lighting_family_stability", None)
            if getattr(best, "lighting_family_stability", None) is not None
            else getattr(best, "lighting_recommendation_stability", 1.0) or 0.0
        )
        rank_score = 100.0 * (
            0.20 * bootstrap_family_top1
            + 0.30 * bootstrap_family_top3
            + 0.20 * lighting_family_top1
            + 0.30 * lighting_family_top3
        )
        exact_rank_score = 100.0 * (
            0.20 * float(getattr(best, "recommendation_stability", 0.0) or 0.0)
            + 0.30 * float(getattr(best, "top3_stability", 0.0) or 0.0)
            + 0.20
            * float(
                getattr(best, "lighting_recommendation_stability", 0.0) or 0.0
            )
            + 0.30
            * float(getattr(best, "lighting_top3_stability", 0.0) or 0.0)
        )

    capture_certainty = float(np.clip(100.0 * (1.0 - radius / 12.0), 0.0, 100.0))
    sensitivity_certainty = float(
        np.clip(100.0 * (1.0 - sensitivity_radius / 8.0), 0.0, 100.0)
    )
    capture_readiness_score = float(
        np.clip(
            0.43 * extraction_score
            + 0.14 * (lighting_score * 100.0)
            + 0.13 * stability
            + 0.11 * capture_certainty
            + 0.10 * sensitivity_score
            + 0.09 * sensitivity_certainty,
            0.0,
            100.0,
        )
    )
    combined = float(
        np.clip(0.95 * capture_readiness_score + 0.05 * rank_score, 0.0, 100.0)
    )
    if eyewear_detected:
        combined = max(combined - 4.0, 0.0)
    reasons = [
        f"Readiness gates: {thresholds.source}.",
        f"Skin Extraction Quality {extraction_score:.0f}/100.",
        f"Face-aware lighting quality {lighting_score:.0%}.",
        f"Bootstrap uncertainty radius {local_radius:.1f} Delta E (local patches).",
        f"Total capture uncertainty radius {radius:.1f} Delta E.",
        f"Shade-family ranking stability {rank_score:.0f}/100.",
        f"Exact-product ranking stability {exact_rank_score:.0f}/100.",
        f"Lighting sensitivity {sensitivity_score:.0f}/100 with {sensitivity_radius:.1f} Delta E variation.",
    ]
    if eyewear_detected:
        reasons.append(
            "Glasses/reflection risk reduced readiness after upper-cheek exclusion."
        )

    if low_signal:
        return RecommendationReadiness(
            state="provisional",
            score=combined,
            confidence_cap=0.55,
            summary=(
                "Recommendations are provisional because facial skin regions "
                "do not contain enough reliable light and color signal."
            ),
            reasons=reasons
            + ["Low-signal face lighting requires a brighter, more even recapture."],
            warnings=[
                "Retake the photo in brighter, even daylight. The Top 3 remain "
                "visible as provisional candidates only."
            ],
            capture_readiness_score=capture_readiness_score,
            shade_family_stability_score=rank_score,
            exact_product_stability_score=exact_rank_score,
        )

    if (
        success
        and combined >= thresholds.ready_score
        and extraction_score >= thresholds.ready_extraction_score
        and lighting_score >= thresholds.ready_lighting_score
        and radius <= thresholds.ready_max_uncertainty
        and sensitivity_radius <= thresholds.ready_max_sensitivity
        and bootstrap_family_top3
        >= thresholds.ready_min_bootstrap_family_top3
        and lighting_family_top3
        >= thresholds.ready_min_lighting_family_top3
    ):
        confidence_cap = float(
            np.clip(
                0.78
                + 0.15
                * (
                    (combined - thresholds.ready_score)
                    / max(90.0 - thresholds.ready_score, 1.0)
                ),
                0.78,
                0.93,
            )
        )
        return RecommendationReadiness(
            state="ready",
            score=combined,
            confidence_cap=confidence_cap,
            summary="Recommendation evidence is ready for comparison with the catalog.",
            reasons=reasons,
            capture_readiness_score=capture_readiness_score,
            shade_family_stability_score=rank_score,
            exact_product_stability_score=exact_rank_score,
        )
    if (
        success
        and combined >= thresholds.caution_score
        and extraction_score >= thresholds.caution_extraction_score
        and radius <= thresholds.caution_max_uncertainty
        and sensitivity_radius <= thresholds.caution_max_sensitivity
        and bootstrap_family_top3
        >= thresholds.caution_min_bootstrap_family_top3
        and lighting_family_top3
        >= thresholds.caution_min_lighting_family_top3
    ):
        confidence_cap = float(
            np.clip(
                0.55
                + 0.20
                * (
                    (combined - thresholds.caution_score)
                    / max(
                        thresholds.ready_score - thresholds.caution_score,
                        1.0,
                    )
                ),
                0.55,
                0.75,
            )
        )
        return RecommendationReadiness(
            state="caution",
            score=combined,
            confidence_cap=confidence_cap,
            summary="Recommendations are usable with caution; image or extraction evidence is limited.",
            reasons=reasons,
            warnings=["Consider retaking the photo in soft, even daylight for a more stable ranking."],
            capture_readiness_score=capture_readiness_score,
            shade_family_stability_score=rank_score,
            exact_product_stability_score=exact_rank_score,
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
        capture_readiness_score=capture_readiness_score,
        shade_family_stability_score=rank_score,
        exact_product_stability_score=exact_rank_score,
    )
