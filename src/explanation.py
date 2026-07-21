"""Deterministic natural-language explanations for shade recommendations."""


def _closeness_phrase(delta_e: float) -> str:
    if delta_e < 2.0:
        return "an excellent perceptual color match"
    if delta_e < 5.0:
        return "a close perceptual color match"
    if delta_e < 10.0:
        return "a reasonably close perceptual color match"
    return "the closest available match, though not a tight one"


def build_explanation(match, skin_result, quality_report, rank: int, matches: list) -> str:
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

    if match.undertone:
        parts.append(f"Its listed undertone is {match.undertone}.")

    if quality_report.region_consistency < 0.5:
        parts.append(
            "Confidence is reduced because the forehead, cheek, and jawline regions "
            "did not agree closely in color (possible uneven lighting, shadows, or occlusion)."
        )

    if quality_report.valid_pixel_ratio < 0.5:
        parts.append(
            "Fewer valid skin pixels than ideal survived filtering, adding some uncertainty."
        )

    if quality_report.face_quality < 0.8:
        parts.append(
            "Face detection quality was reduced (e.g. a small or multiply-detected face), "
            "which adds some uncertainty."
        )

    if rank <= 2 and len(matches) >= 2 and quality_report.top_match_separation < 0.3:
        parts.append(
            "The top two recommendations are very close in color, so the exact ranking "
            "between them is uncertain."
        )

    return " ".join(parts)
