"""Human-readable skin extraction summary for the Streamlit result view."""


def _format_region_list(names: list[str]) -> str:
    if not names:
        return "none"
    return ", ".join(name.replace("_", " ").title() for name in names)


def _reliability_label(score: float) -> str:
    if score >= 0.78:
        return "Good"
    if score >= 0.55:
        return "Moderate"
    return "Reduced"


def build_skin_extraction_summary(skin_result, lighting_quality, extraction_selection) -> str:
    """Return a compact, demo-friendly explanation of extraction reliability."""
    included = _format_region_list(getattr(skin_result, "included_region_names", []))
    excluded = _format_region_list(getattr(skin_result, "excluded_region_names", []))
    reduced = [
        name
        for name, region in skin_result.region_results.items()
        if not region.excluded and region.weight_multiplier < 1.0
    ]
    reduced_text = _format_region_list(reduced)

    highlight_regions = [
        name
        for name, region in skin_result.region_results.items()
        if region.highlight_patches_rejected > 0 or region.shadow_patches_rejected > 0
    ]
    highlight_text = (
        f"Highlight/shadow handling was active in {_format_region_list(highlight_regions)}."
        if highlight_regions
        else "No major highlight/shadow patch rejection was needed."
    )

    reliability = _reliability_label(float(getattr(skin_result, "quality_score", 0.0)))
    source = getattr(extraction_selection, "selected_source", "auto")
    source_text = "original" if source == "original" else "corrected"
    lighting_score = float(getattr(lighting_quality, "score", 1.0))
    lighting_text = f"Lighting quality was {lighting_score:.0%}."

    confidence_reason = "the trusted regions were consistent"
    if getattr(skin_result, "region_consistency", 1.0) < 0.55:
        confidence_reason = "region colors disagreed, so confidence is reduced"
    elif getattr(skin_result, "usable_region_count", 0) < 3:
        confidence_reason = "fewer than three regions were usable, so confidence is reduced"
    elif reduced:
        confidence_reason = "some regions were reduced due to reliability checks"

    return (
        f"Extraction reliability: {reliability}. "
        f"Trusted regions used: {included}. "
        f"Excluded regions: {excluded}. "
        f"Reduced-weight regions: {reduced_text}. "
        f"{lighting_text} {highlight_text} "
        f"Final extraction source: {source_text}. "
        f"Final depth estimate: {skin_result.depth_estimate or 'unknown'}. "
        f"The system trusts this result because {confidence_reason}. "
        f"{getattr(extraction_selection, 'reason', '')}"
    ).strip()
