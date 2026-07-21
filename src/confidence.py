"""Confidence scoring for shade matches.

Combines: match distance (Delta E), region consistency, valid pixel ratio,
face detection quality, and Top1-vs-Top2 separation into an interpretable
0-1 confidence per shade, per the weighting in docs/02_TECHNICAL_ARCHITECTURE.md:

    match_distance_score     50%
    region_consistency       20%
    valid_pixel_ratio        10%
    face_detection_quality   10%
    top_match_separation     10%
"""

from dataclasses import dataclass, field

import numpy as np

DELTA_E_TEMPERATURE = 15.0
SEPARATION_SCALE = 5.0
CONFIDENCE_FLOOR = 0.02
CONFIDENCE_CEILING = 0.93  # never claim near-100% certainty

WEIGHT_MATCH_DISTANCE = 0.5
WEIGHT_REGION_CONSISTENCY = 0.2
WEIGHT_VALID_PIXEL_RATIO = 0.1
WEIGHT_FACE_QUALITY = 0.1
WEIGHT_TOP_SEPARATION = 0.1


@dataclass
class QualityReport:
    region_consistency: float
    valid_pixel_ratio: float
    face_quality: float
    top_match_separation: float
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
    return float(np.clip(score, 0.2, 1.0))


def _separation_score(matches: list) -> float:
    """How clearly the best match stands out from the second-best.
    A small Delta E gap between rank 1 and rank 2 means the top pick is
    ambiguous, which should reduce confidence in all recommendations."""
    if len(matches) < 2:
        return 0.6  # can't measure separation; neutral default
    gap = matches[1].delta_e - matches[0].delta_e
    return float(np.clip(gap / SEPARATION_SCALE, 0.0, 1.0))


def build_quality_report(skin_result, face_result, matches: list) -> QualityReport:
    """Bundle the non-per-shade quality signals used by `compute_confidence`."""
    warnings = []
    face_quality = _face_quality_score(face_result.warnings if face_result else [])
    separation = _separation_score(matches)

    if separation < 0.3 and len(matches) >= 2:
        warnings.append(
            "The top two shade recommendations are very close in color; "
            "confidence is reduced because the best pick is ambiguous."
        )

    return QualityReport(
        region_consistency=getattr(skin_result, "region_consistency", 0.0),
        valid_pixel_ratio=getattr(skin_result, "avg_valid_pixel_ratio", 0.0),
        face_quality=face_quality,
        top_match_separation=separation,
        warnings=warnings,
    )


def compute_confidence(matches: list, quality_report: QualityReport, temperature: float = DELTA_E_TEMPERATURE) -> list:
    """Set a 0-1 `confidence` on each ShadeMatch and return the list.

    Only the match-distance term varies per shade; the other four factors
    describe overall extraction/detection quality and apply equally to
    every candidate in the list.
    """
    for match in matches:
        closeness = float(np.exp(-match.delta_e / temperature))
        raw_confidence = (
            WEIGHT_MATCH_DISTANCE * closeness
            + WEIGHT_REGION_CONSISTENCY * quality_report.region_consistency
            + WEIGHT_VALID_PIXEL_RATIO * quality_report.valid_pixel_ratio
            + WEIGHT_FACE_QUALITY * quality_report.face_quality
            + WEIGHT_TOP_SEPARATION * quality_report.top_match_separation
        )
        match.confidence = float(np.clip(raw_confidence, CONFIDENCE_FLOOR, CONFIDENCE_CEILING))
    return matches
