"""Deterministic natural-language explanations for shade recommendations."""


def _closeness_phrase(delta_e: float) -> str:
    if delta_e < 2.0:
        return "an excellent perceptual color match"
    if delta_e < 5.0:
        return "a close perceptual color match"
    if delta_e < 10.0:
        return "a reasonably close perceptual color match"
    return "the closest available match, though not a tight one"


def build_explanation(
    match,
    skin_result,
    quality_report,
    rank: int,
    matches: list,
    readiness=None,
) -> str:
    """Build a deterministic, factor-grounded explanation for one shade match.

    Mentions: closeness of the Lab/Delta E distance, undertone agreement,
    and any quality caveats (lighting, region disagreement, top-pick
    ambiguity) pulled from the quality report / skin extraction warnings.
    """
    parts = []

    parts.append(
        f"This shade is {_closeness_phrase(match.delta_e)} to the extracted skin tone "
        f"(Delta E {match.delta_e:.1f} using CIEDE2000 on cheek/forehead/jawline color)."
    )
    if getattr(match, "color_fit_score", None) is not None:
        stability = getattr(match, "shade_family_stability_score", None)
        stability_text = (
            f"{stability:.0%} shade-family stability"
            if stability is not None
            else "stability unavailable"
        )
        parts.append(
            f"Its candidate evidence includes {match.color_fit_score:.0%} color fit, "
            f"{stability_text}, and "
            f"{getattr(match, 'catalog_quality_score', 0.5):.0%} catalog evidence."
        )
        if (
            getattr(match, "confidence_stability_source", "")
            == "exact_product_fallback"
        ):
            parts.append(
                "Exact-product Top-3 stability was used as an explicit fallback "
                "because shade-family stability was unavailable."
            )

    if match.undertone:
        parts.append(f"Its listed undertone is {match.undertone}.")

    if getattr(match, "depth_penalty", 0.0) > 0:
        parts.append(
            "Depth was used only as a small tie-breaker because nearby catalog colors "
            "had very similar Delta E scores."
        )

    if getattr(match, "depth_sanity_note", None):
        parts.append(match.depth_sanity_note)

    if getattr(match, "product_variants", None):
        parts.append(
            "This shade appears in multiple product formats in the catalog; the closest "
            "matching variant is shown."
        )

    if getattr(match, "recommendation_stability", None) is not None:
        parts.append(
            f"It remained in the Top 3 for {match.top3_stability:.0%} of deterministic "
            "patch-bootstrap samples."
        )

    if getattr(match, "catalog_quality_score", 1.0) < 0.75:
        parts.append(
            "Catalog metadata for this candidate is limited, so its candidate "
            "confidence is reduced."
        )

    if quality_report.region_consistency < 0.5:
        parts.append(
            "Capture readiness is reduced because the forehead, cheek, and jawline "
            "regions did not agree closely in color (possible uneven lighting, "
            "shadows, or occlusion)."
        )

    if quality_report.valid_pixel_ratio < 0.5:
        parts.append(
            "Fewer valid skin pixels than ideal survived filtering, reducing capture readiness."
        )

    if quality_report.face_quality < 0.8:
        parts.append(
            "Face detection quality was reduced (e.g. a small or multiply-detected face), "
            "which reduces capture readiness."
        )

    if getattr(quality_report, "cheek_area_balance", 1.0) < 0.45:
        parts.append(
            "One cheek contributed much less valid skin area than the other, so capture "
            "readiness is reduced slightly."
        )

    if getattr(quality_report, "uncertainty_radius", 0.0) > 6.0:
        parts.append(
            "Patch-bootstrap uncertainty is elevated, indicating that retained facial "
            "regions did not produce a tightly stable color estimate."
        )

    if getattr(quality_report, "close_match_tie", False):
        parts.append(
            "Close match tie: these shades are nearly identical in catalog color and "
            "should be considered equivalent candidates."
        )
    elif rank <= 2 and len(matches) >= 2 and quality_report.top_match_separation < 0.3:
        parts.append(
            "The top two recommendations are very close in color, so the exact ranking "
            "between them is uncertain."
        )

    if getattr(readiness, "state", "ready") == "provisional":
        parts.append(
            "This is a provisional candidate; retake the photo in soft, even daylight "
            "before treating it as a dependable match."
        )

    return " ".join(parts)
