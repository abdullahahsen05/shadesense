"""Robust consensus across two or three independently analysed photos."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000, lab2rgb

from src.confidence import build_quality_report, compute_confidence
from src.multicapture_consensus import (
    CaptureEvidence,
    build_multicapture_consensus,
)
from src.recommendation_readiness import RecommendationReadiness
from src.shade_matcher import match_shades


@dataclass(frozen=True)
class CaptureConsensusEvidence:
    capture_index: int
    lab: tuple[float, float, float]
    weight: float
    distance_from_medoid: float
    included: bool
    readiness_state: str


@dataclass
class MultiPhotoConsensusResult:
    success: bool
    consensus_lab: tuple[float, float, float] | None = None
    consensus_rgb: tuple[int, int, int] | None = None
    retained_indices: list[int] = field(default_factory=list)
    rejected_indices: list[int] = field(default_factory=list)
    evidence: list[CaptureConsensusEvidence] = field(default_factory=list)
    agreement_delta_e_p90: float = 0.0
    uncertainty_labs: list[tuple[float, float, float]] = field(
        default_factory=list
    )
    matches: list = field(default_factory=list)
    readiness: RecommendationReadiness | None = None
    reference_index: int | None = None
    warnings: list[str] = field(default_factory=list)
    explanation: str = ""


def _target_lab(analysis) -> np.ndarray:
    skin = analysis.skin_result
    value = (
        skin.foundation_target_lab
        if skin.foundation_target_active and skin.foundation_target_lab
        else skin.lab
    )
    return np.asarray(value, dtype=np.float64)


def _capture_weight(analysis) -> float:
    extraction = float(
        (analysis.extraction_quality_report or {}).get("overall_score", 0.0)
    )
    readiness = analysis.recommendation_readiness
    readiness_score = float(getattr(readiness, "score", extraction))
    lighting = float(getattr(analysis.lighting_quality, "score", 0.0)) * 100.0
    low_signal = bool(getattr(analysis.lighting_quality, "low_signal", False))
    weight = 0.45 * extraction + 0.35 * readiness_score + 0.20 * lighting
    if low_signal:
        weight *= 0.35
    return float(np.clip(weight / 100.0, 0.05, 1.0))


def _balanced_uncertainty_samples(
    analyses: list,
    retained_indices: list[int],
    consensus_lab: np.ndarray,
    count: int = 96,
) -> list[tuple[float, float, float]]:
    pools = []
    capture_centres = []
    for index in retained_indices:
        skin = analyses[index].skin_result
        centre = _target_lab(analyses[index])
        capture_centres.append(centre)
        samples = np.asarray(skin.bootstrap_labs or [centre], dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 3:
            samples = centre.reshape(1, 3)
        pools.append(samples)
    result = []
    for sample_index in range(count):
        pool_index = sample_index % len(pools)
        pool = pools[pool_index]
        source_index = (
            sample_index // len(pools)
        ) * max(len(pool) - 1, 1) // max(count // len(pools), 1)
        source = pool[min(source_index, len(pool) - 1)]
        # Preserve within-capture patch variation while centring the
        # distribution on the multi-capture consensus.
        adjusted = consensus_lab + (source - capture_centres[pool_index])
        result.append(tuple(float(value) for value in adjusted))
    return result


def _consensus_readiness(
    analyses: list,
    retained: list[int],
    agreement: float,
) -> RecommendationReadiness:
    states = [
        getattr(analyses[index].recommendation_readiness, "state", "provisional")
        for index in retained
    ]
    capture_scores = [
        float(
            getattr(
                analyses[index].recommendation_readiness,
                "capture_readiness_score",
                0.0,
            )
        )
        for index in retained
    ]
    if len(retained) >= 2 and agreement <= 4.0 and "provisional" not in states:
        state, cap = "ready", 0.93
    elif len(retained) >= 2 and agreement <= 7.0:
        state, cap = "caution", 0.75
    else:
        state, cap = "provisional", 0.55
    score = float(
        np.clip(
            np.mean(capture_scores)
            + (8.0 if len(retained) >= 2 else 0.0)
            - 3.0 * max(agreement - 3.0, 0.0),
            0.0,
            100.0,
        )
    )
    return RecommendationReadiness(
        state=state,
        score=score,
        confidence_cap=cap,
        summary=(
            f"{len(retained)} consistent captures were combined; "
            f"cross-photo variation is {agreement:.1f} Delta E."
        ),
        reasons=[
            "Each photo was processed independently before consensus.",
            "A weighted CIEDE2000 medoid identified the central capture.",
            "Gross capture outliers were excluded only when at least three photos were available.",
        ],
        warnings=(
            [
                "Cross-photo color disagreement remains high. Retake the photos "
                "in the same soft, even daylight."
            ]
            if state == "provisional"
            else []
        ),
        capture_readiness_score=score,
        shade_family_stability_score=0.0,
        exact_product_stability_score=0.0,
    )


def build_multi_photo_consensus(
    analyses: list,
    catalog_df: pd.DataFrame,
    *,
    top_k: int = 3,
) -> MultiPhotoConsensusResult:
    """Combine valid photos and rank shades from the retained evidence."""
    valid_pairs = [
        (index, analysis)
        for index, analysis in enumerate(analyses)
        if analysis is not None
        and analysis.success
        and analysis.skin_result is not None
        and analysis.skin_result.success
    ]
    if not valid_pairs:
        return MultiPhotoConsensusResult(
            success=False,
            warnings=["No uploaded photo produced a usable skin extraction."],
        )

    original_indices = [index for index, _ in valid_pairs]
    valid_analyses = [analysis for _, analysis in valid_pairs]
    labs = np.asarray([_target_lab(item) for item in valid_analyses])
    weights = np.asarray([_capture_weight(item) for item in valid_analyses])
    low_signal = [
        bool(getattr(item.lighting_quality, "low_signal", False))
        for item in valid_analyses
    ]
    low_level = build_multicapture_consensus(
        [
            CaptureEvidence(
                capture_id=str(index),
                lab=tuple(float(value) for value in labs[index]),
                extraction_score=float(
                    (
                        valid_analyses[index].extraction_quality_report or {}
                    ).get("overall_score", 0.0)
                ),
                lighting_score=float(
                    getattr(valid_analyses[index].lighting_quality, "score", 0.0)
                ),
                uncertainty_radius=float(
                    (
                        valid_analyses[index]
                        .skin_result.systematic_uncertainty_diagnostics
                        or {}
                    ).get("total_delta_e_radius_p90", 12.0)
                ),
                low_signal=low_signal[index],
            )
            for index in range(len(valid_analyses))
        ]
    )
    if not low_level.success:
        return MultiPhotoConsensusResult(
            success=False,
            warnings=low_level.warnings,
        )
    retained_local_indices = [
        int(value) for value in low_level.included_capture_ids
    ]
    rejected_local_indices = [
        int(value) for value in low_level.excluded_capture_ids
    ]
    retained_local = np.zeros(len(labs), dtype=bool)
    retained_local[retained_local_indices] = True
    consensus_lab = np.asarray(low_level.lab)
    agreement = low_level.uncertainty_radius_p90
    distances = np.asarray(
        [
            low_level.delta_e_by_capture[str(index)]
            for index in range(len(valid_analyses))
        ]
    )
    retained_indices = [
        original_indices[index] for index in retained_local_indices
    ]
    rejected_indices = [
        original_indices[index] for index in rejected_local_indices
    ]
    uncertainty_labs = _balanced_uncertainty_samples(
        analyses,
        retained_indices,
        consensus_lab,
    )
    matches = match_shades(
        consensus_lab,
        catalog_df,
        top_k=top_k,
        uncertainty_labs=uncertainty_labs,
    )
    readiness = _consensus_readiness(analyses, retained_indices, agreement)
    reference_index = max(
        retained_indices,
        key=lambda index: _capture_weight(analyses[index]),
    )
    quality = build_quality_report(
        analyses[reference_index].skin_result,
        analyses[reference_index].face_result,
        matches,
        analyses[reference_index].lighting_quality,
    )
    quality.extraction_uncertainty = float(
        np.clip(1.0 - agreement / 12.0, 0.0, 1.0)
    )
    quality.uncertainty_radius = max(quality.uncertainty_radius, agreement)
    matches = compute_confidence(matches, quality, readiness=readiness)
    readiness = replace(
        readiness,
        shade_family_stability_score=float(
            100.0
            * (
                getattr(matches[0], "top3_family_stability", 0.0)
                if matches
                else 0.0
            )
        ),
        exact_product_stability_score=float(
            100.0
            * (getattr(matches[0], "top3_stability", 0.0) if matches else 0.0)
        ),
    )
    rgb = lab2rgb(consensus_lab.reshape(1, 1, 3)).reshape(3)
    consensus_rgb = tuple(int(round(value)) for value in np.clip(rgb * 255, 0, 255))
    evidence = []
    for local_index, original_index in enumerate(original_indices):
        readiness_state = getattr(
            valid_analyses[local_index].recommendation_readiness,
            "state",
            "provisional",
        )
        evidence.append(
            CaptureConsensusEvidence(
                capture_index=original_index,
                lab=tuple(float(value) for value in labs[local_index]),
                weight=float(weights[local_index]),
                distance_from_medoid=float(distances[local_index]),
                included=bool(retained_local[local_index]),
                readiness_state=readiness_state,
            )
        )
    warnings = []
    if rejected_indices:
        warnings.append(
            "One inconsistent capture was excluded from the final consensus."
        )
    return MultiPhotoConsensusResult(
        success=True,
        consensus_lab=tuple(float(value) for value in consensus_lab),
        consensus_rgb=consensus_rgb,
        retained_indices=retained_indices,
        rejected_indices=rejected_indices,
        evidence=evidence,
        agreement_delta_e_p90=agreement,
        uncertainty_labs=uncertainty_labs,
        matches=matches,
        readiness=readiness,
        reference_index=reference_index,
        warnings=warnings + readiness.warnings,
        explanation=(
            f"Combined {len(retained_indices)} of {len(valid_pairs)} usable "
            "captures with quality weighting after perceptual outlier checks."
        ),
    )
