"""Candidate-specific confidence for shade matches.

Capture and extraction reliability are represented by the recommendation
readiness cap. Each candidate is then differentiated within that ceiling using
its uncertainty-aware color fit, shade-family stability, and catalog evidence.
The result is an explainable engineering score, not a calibrated probability.
"""

from dataclasses import dataclass, field

import numpy as np

DELTA_E_TEMPERATURE = 15.0
SEPARATION_SCALE = 5.0
NEAR_TIE_DELTA_E_SPREAD = 1.25
CONFIDENCE_FLOOR = 0.02
CONFIDENCE_CEILING = 0.93  # never claim near-100% certainty

CANDIDATE_WEIGHT_COLOR_FIT = 0.65
CANDIDATE_WEIGHT_STABILITY = 0.25
CANDIDATE_WEIGHT_CATALOG = 0.10
FAMILY_BOOTSTRAP_WEIGHT = 0.70
FAMILY_LIGHTING_WEIGHT = 0.30


@dataclass
class QualityReport:
    region_consistency: float
    valid_pixel_ratio: float
    face_quality: float
    top_match_separation: float
    lighting_quality: float = 1.0
    cheek_area_balance: float = 1.0
    usable_region_count: int = 0
    region_stability: float = 1.0
    highlight_safety: float = 1.0
    extraction_uncertainty: float = 1.0
    uncertainty_radius: float = 0.0
    lighting_sensitivity: float = 1.0
    lighting_sensitivity_radius: float = 0.0
    close_match_tie: bool = False
    warnings: list = field(default_factory=list)


def _face_quality_score(face_warnings: list) -> float:
    """1.0 = clean detection. Penalized for multi-face selection or a small/
    marginal face, both of which make the extracted region less trustworthy."""
    score = 1.0
    for w in face_warnings:
        lower = w.lower()
        if "faces detected" in lower:
            score -= 0.25
        if "small" in lower:
            score -= 0.25
        if "lower-confidence" in lower:
            score -= 0.20
    return float(np.clip(score, 0.2, 1.0))


def _separation_score(matches: list) -> float:
    """How clearly the best match stands out from the second-best.
    A small Delta E gap between rank 1 and rank 2 means the top pick is
    ambiguous. This is reported as a shortlist warning rather than mixed
    directly into every candidate score."""
    if len(matches) < 2:
        return 0.6  # can't measure separation; neutral default
    gap = matches[1].delta_e - matches[0].delta_e
    return float(np.clip(gap / SEPARATION_SCALE, 0.0, 1.0))


def build_quality_report(skin_result, face_result, matches: list, lighting_quality=None) -> QualityReport:
    """Bundle the non-per-shade quality signals used by `compute_confidence`."""
    warnings = []
    face_quality = _face_quality_score(face_result.warnings if face_result else [])
    separation = _separation_score(matches)

    if separation < 0.3 and len(matches) >= 2:
        warnings.append(
            "The top two shade recommendations are very close in color; "
            "treat them as an ambiguous shortlist rather than a decisive order."
        )

    close_match_tie = False
    if len(matches) >= 3:
        top_three = matches[:3]
        delta_spread = max(m.delta_e for m in top_three) - min(m.delta_e for m in top_three)
        close_match_tie = delta_spread <= NEAR_TIE_DELTA_E_SPREAD
        if close_match_tie:
            warnings.append(
            "Close match tie: these shades are nearly identical in catalog color "
            "and should be considered equivalent candidates."
            )

    cheek_area_balance = float(np.clip(getattr(skin_result, "cheek_area_balance", 1.0), 0.0, 1.0))
    if cheek_area_balance < 0.45:
        warnings.append(
            "One cheek has much less valid skin area than the other; capture "
            "readiness may be reduced."
        )
    usable_region_count = int(getattr(skin_result, "usable_region_count", 0))
    if 0 < usable_region_count < 3:
        warnings.append(
            "Fewer than 3 skin regions were usable; capture readiness may be reduced."
        )
    stability_diagnostics = getattr(skin_result, "stability_diagnostics", {}) or {}
    region_stability = float(np.clip(stability_diagnostics.get("stability_score", 100.0) / 100.0, 0.0, 1.0))
    if region_stability < 0.7:
        warnings.extend(stability_diagnostics.get("warnings", []))
    region_results = getattr(skin_result, "region_results", {}) or {}
    highlight_count = sum(int(getattr(region, "highlight_patches_rejected", 0)) for region in region_results.values())
    specular_count = sum(1 for region in region_results.values() if getattr(region, "specular_highlight_detected", False))
    highlight_safety = float(np.clip(1.0 - 0.05 * highlight_count - 0.08 * specular_count, 0.45, 1.0))
    if highlight_safety < 0.9:
        warnings.append(
            "Highlight influence was detected; capture readiness may be reduced "
            "because the extracted tone could skew light."
        )
    lighting_score = float(np.clip(getattr(lighting_quality, "score", 1.0), 0.0, 1.0))
    uncertainty_diagnostics = getattr(skin_result, "uncertainty_diagnostics", {}) or {}
    extraction_uncertainty = float(
        np.clip(uncertainty_diagnostics.get("stability_score", 45.0) / 100.0, 0.0, 1.0)
    )
    uncertainty_radius = float(
        uncertainty_diagnostics.get("delta_e_radius_p90", 12.0)
    )
    if uncertainty_radius > 6.0:
        warnings.append(
            f"Patch bootstrap uncertainty is elevated ({uncertainty_radius:.1f} Delta E radius)."
        )
    if lighting_quality is not None:
        warnings.extend(getattr(lighting_quality, "warnings", []))
    sensitivity_diagnostics = (
        getattr(skin_result, "lighting_sensitivity_diagnostics", {}) or {}
    )
    lighting_sensitivity = float(
        np.clip(sensitivity_diagnostics.get("score", 100.0) / 100.0, 0.0, 1.0)
    )
    lighting_sensitivity_radius = float(
        sensitivity_diagnostics.get("delta_e_p90", 0.0)
    )
    if lighting_sensitivity_radius > 3.0:
        warnings.append(
            "Recommendations changed under small simulated exposure or white-balance variations."
        )

    return QualityReport(
        region_consistency=getattr(skin_result, "region_consistency", 0.0),
        valid_pixel_ratio=getattr(skin_result, "avg_valid_pixel_ratio", 0.0),
        lighting_quality=lighting_score,
        face_quality=face_quality,
        top_match_separation=separation,
        cheek_area_balance=cheek_area_balance,
        usable_region_count=usable_region_count,
        region_stability=region_stability,
        highlight_safety=highlight_safety,
        extraction_uncertainty=extraction_uncertainty,
        uncertainty_radius=uncertainty_radius,
        lighting_sensitivity=lighting_sensitivity,
        lighting_sensitivity_radius=lighting_sensitivity_radius,
        close_match_tie=close_match_tie,
        warnings=warnings,
    )


def delta_e_to_color_fit(
    delta_e: float,
    temperature: float = DELTA_E_TEMPERATURE,
) -> float:
    """Map uncertainty-aware Delta E to a documented 0-1 color-fit score."""
    safe_temperature = max(float(temperature), 1e-6)
    return float(
        np.clip(
            np.exp(-max(float(delta_e), 0.0) / safe_temperature),
            0.0,
            1.0,
        )
    )


def _normalized_weighted_score(
    factors: list[tuple[str, float, float | None]],
) -> tuple[float, dict[str, float]]:
    """Combine available evidence only and normalize its configured weights."""
    available = [
        (name, weight, float(np.clip(value, 0.0, 1.0)))
        for name, weight, value in factors
        if value is not None
    ]
    total_weight = sum(weight for _, weight, _ in available)
    if total_weight <= 0:
        return 0.0, {}
    normalized = {
        name: weight / total_weight for name, weight, _ in available
    }
    score = sum(normalized[name] * value for name, _, value in available)
    return float(np.clip(score, 0.0, 1.0)), normalized


def _candidate_stability(match) -> tuple[float | None, str]:
    """Prefer Top-3 shade-family stability; fall back explicitly to exact SKU."""
    family_score, family_weights = _normalized_weighted_score(
        [
            (
                "bootstrap_family_top3",
                FAMILY_BOOTSTRAP_WEIGHT,
                getattr(match, "top3_family_stability", None),
            ),
            (
                "lighting_family_top3",
                FAMILY_LIGHTING_WEIGHT,
                getattr(match, "lighting_top3_family_stability", None),
            ),
        ]
    )
    if family_weights:
        return family_score, "shade_family"

    exact_score, exact_weights = _normalized_weighted_score(
        [
            (
                "bootstrap_exact_top3",
                FAMILY_BOOTSTRAP_WEIGHT,
                getattr(match, "top3_stability", None),
            ),
            (
                "lighting_exact_top3",
                FAMILY_LIGHTING_WEIGHT,
                getattr(match, "lighting_top3_stability", None),
            ),
        ]
    )
    if exact_weights:
        return exact_score, "exact_product_fallback"
    return None, "unavailable"


def compute_confidence(
    matches: list,
    quality_report: QualityReport,
    temperature: float = DELTA_E_TEMPERATURE,
    readiness=None,
) -> list:
    """Compute candidate-specific heuristic confidence under a capture cap.

    Capture and extraction quality are represented once by ``readiness`` and
    limit the score range. Candidate evidence determines where each shade sits
    within that range: 65% color fit, 25% stability, and 10% catalog evidence.
    Missing factors are omitted and the remaining weights are normalized.
    """
    del quality_report  # retained in the public signature for compatibility
    readiness_cap = min(
        CONFIDENCE_CEILING,
        float(getattr(readiness, "confidence_cap", CONFIDENCE_CEILING)),
    )

    for match in matches:
        effective_delta_e = getattr(match, "distribution_delta_e", None)
        if effective_delta_e is None:
            effective_delta_e = match.delta_e
        color_fit = delta_e_to_color_fit(effective_delta_e, temperature)
        stability, stability_source = _candidate_stability(match)
        catalog_raw = getattr(match, "catalog_quality_score", None)
        catalog_quality = (
            float(np.clip(catalog_raw, 0.0, 1.0))
            if catalog_raw is not None
            else None
        )
        candidate_evidence, normalized_weights = _normalized_weighted_score(
            [
                ("color_fit", CANDIDATE_WEIGHT_COLOR_FIT, color_fit),
                ("shade_family_stability", CANDIDATE_WEIGHT_STABILITY, stability),
                ("catalog_evidence", CANDIDATE_WEIGHT_CATALOG, catalog_quality),
            ]
        )
        candidate_confidence = float(
            np.clip(
                readiness_cap * candidate_evidence,
                CONFIDENCE_FLOOR,
                readiness_cap,
            )
        )

        match.color_fit_score = color_fit
        match.shade_family_stability_score = stability
        match.confidence_stability_source = stability_source
        match.candidate_confidence = candidate_confidence
        # Preserve the original field as a backward-compatible alias.
        match.confidence = candidate_confidence
        match.confidence_breakdown = {
            "candidate_evidence": candidate_evidence,
            "candidate_confidence": candidate_confidence,
            "color_fit_score": color_fit,
            "shade_family_stability_score": stability,
            "stability_source": stability_source,
            "catalog_evidence_score": catalog_quality,
            "distribution_delta_e": float(effective_delta_e),
            "readiness_cap": readiness_cap,
            "normalized_weights": normalized_weights,
        }
    return matches
