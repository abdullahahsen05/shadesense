"""Robust consensus across multiple independent face captures."""

from dataclasses import dataclass, field

import numpy as np
from skimage.color import deltaE_ciede2000


@dataclass(frozen=True)
class CaptureEvidence:
    capture_id: str
    lab: tuple[float, float, float]
    extraction_score: float = 70.0
    lighting_score: float = 0.70
    uncertainty_radius: float = 6.0
    low_signal: bool = False


@dataclass(frozen=True)
class MultiCaptureConsensus:
    success: bool
    lab: tuple[float, float, float] | None = None
    anchor_capture_id: str | None = None
    included_capture_ids: list[str] = field(default_factory=list)
    excluded_capture_ids: list[str] = field(default_factory=list)
    excluded_low_signal_capture_ids: list[str] = field(default_factory=list)
    excluded_perceptual_outlier_capture_ids: list[str] = field(
        default_factory=list
    )
    delta_e_by_capture: dict[str, float] = field(default_factory=dict)
    uncertainty_radius_p90: float = 12.0
    repeatability_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    method: str = "weighted CIEDE2000 medoid with weighted-MAD capture rejection"


def _capture_weight(capture: CaptureEvidence) -> float:
    extraction = float(np.clip(capture.extraction_score / 100.0, 0.05, 1.0))
    lighting = float(np.clip(capture.lighting_score, 0.05, 1.0))
    uncertainty = 1.0 / (1.0 + max(capture.uncertainty_radius, 0.0) / 6.0)
    signal = 0.20 if capture.low_signal else 1.0
    return max(extraction * lighting * uncertainty * signal, 0.01)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _medoid_index(labs: np.ndarray, weights: np.ndarray) -> int:
    grid_a = np.repeat(labs[:, None, :], len(labs), axis=1)
    grid_b = np.repeat(labs[None, :, :], len(labs), axis=0)
    distances = deltaE_ciede2000(grid_a, grid_b)
    costs = np.sum(distances * weights[None, :], axis=1)
    return int(np.argmin(costs))


def build_multicapture_consensus(
    captures: list[CaptureEvidence],
) -> MultiCaptureConsensus:
    """Return an observed capture medoid after whole-photo outlier rejection."""
    usable = [
        capture
        for capture in captures
        if len(capture.lab) == 3 and np.all(np.isfinite(capture.lab))
    ]
    if len(usable) < 2:
        return MultiCaptureConsensus(
            success=False,
            warnings=["At least two usable captures are required for consensus."],
        )

    labs = np.asarray([capture.lab for capture in usable], dtype=np.float64)
    weights = np.asarray([_capture_weight(capture) for capture in usable])
    initial_index = _medoid_index(labs, weights)
    initial_lab = labs[initial_index]
    initial_distances = deltaE_ciede2000(
        labs,
        np.repeat(initial_lab[None, :], len(labs), axis=0),
    )
    distance_median = _weighted_median(initial_distances, weights)
    mad = _weighted_median(np.abs(initial_distances - distance_median), weights)
    robust_sigma = 1.4826 * mad
    threshold = max(3.0, distance_median + 2.5 * robust_sigma)
    perceptual_keep = initial_distances <= threshold
    signal_keep = np.asarray(
        [not capture.low_signal for capture in usable],
        dtype=bool,
    )
    keep = perceptual_keep & signal_keep
    if int(np.sum(keep)) < 2:
        # Low-signal flags may remove every photo in a difficult set. Keep the
        # two strongest captures, but make the weak evidence explicit.
        strongest = np.argsort(weights)[-2:]
        keep = np.zeros(len(usable), dtype=bool)
        keep[strongest] = True

    kept_labs = labs[keep]
    kept_weights = weights[keep]
    kept = [capture for capture, is_kept in zip(usable, keep) if is_kept]
    final_index = _medoid_index(kept_labs, kept_weights)
    final_lab = kept_labs[final_index]
    anchor = kept[final_index]
    all_distances = deltaE_ciede2000(
        labs,
        np.repeat(final_lab[None, :], len(labs), axis=0),
    )
    kept_distances = all_distances[keep]
    radius = float(np.percentile(kept_distances, 90))
    score = float(np.clip(100.0 * (1.0 - radius / 10.0), 0.0, 100.0))
    excluded = [
        capture.capture_id
        for capture, is_kept in zip(usable, keep)
        if not is_kept
    ]
    excluded_low_signal = [
        capture.capture_id
        for capture, is_kept, has_signal in zip(usable, keep, signal_keep)
        if not is_kept and not has_signal
    ]
    excluded_perceptual = [
        capture.capture_id
        for capture, is_kept, is_perceptual in zip(
            usable,
            keep,
            perceptual_keep,
        )
        if not is_kept and not is_perceptual
    ]
    warnings = []
    if excluded:
        warnings.append(
            "Whole-photo outliers were excluded from consensus: "
            + ", ".join(excluded)
            + "."
        )
    if radius > 6.0:
        warnings.append(
            "The retained captures still disagree materially; use softer, "
            "more even daylight before trusting the shade family."
        )

    return MultiCaptureConsensus(
        success=True,
        lab=tuple(float(value) for value in final_lab),
        anchor_capture_id=anchor.capture_id,
        included_capture_ids=[capture.capture_id for capture in kept],
        excluded_capture_ids=excluded,
        excluded_low_signal_capture_ids=excluded_low_signal,
        excluded_perceptual_outlier_capture_ids=excluded_perceptual,
        delta_e_by_capture={
            capture.capture_id: float(distance)
            for capture, distance in zip(usable, all_distances)
        },
        uncertainty_radius_p90=radius,
        repeatability_score=score,
        warnings=warnings,
    )
