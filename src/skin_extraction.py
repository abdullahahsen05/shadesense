"""Representative skin color extraction and filtering from masked face regions."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.color import rgb2lab

REGION_NAMES = ["forehead", "left_cheek", "right_cheek", "jawline"]

LUMINANCE_LOWER_PERCENTILE = 20
LUMINANCE_UPPER_PERCENTILE = 80
SATURATION_UPPER_PERCENTILE = 95
MIN_VALID_PIXELS_PER_REGION = 100
REGION_DISAGREEMENT_THRESHOLD = 12.0  # Lab distance considered "high disagreement"
OUTLIER_LAB_DISTANCE = 20.0  # Lab distance from consensus beyond which a region is excluded


@dataclass
class RegionSkinResult:
    name: str
    total_pixel_count: int
    valid_pixel_count: int
    valid_ratio: float
    median_rgb: tuple | None
    median_lab: tuple | None
    reliable: bool
    warnings: list = field(default_factory=list)


@dataclass
class SkinToneResult:
    rgb: tuple
    lab: tuple
    region_results: dict
    quality_score: float
    region_consistency: float = 0.0
    avg_valid_pixel_ratio: float = 0.0
    warnings: list = field(default_factory=list)
    success: bool = True


def _to_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) uint8 RGB pixel array to (N, 3) Lab floats."""
    return rgb2lab(pixels_rgb.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)


def _to_saturation(pixels_rgb: np.ndarray) -> np.ndarray:
    """Return the HSV saturation channel (0-255) for an (N, 3) uint8 RGB pixel array."""
    hsv = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv.reshape(-1, 3)[:, 1].astype(np.float64)


def _extract_region(image_rgb: np.ndarray, mask: np.ndarray, name: str) -> RegionSkinResult:
    warnings: list = []
    pixels = image_rgb[mask > 0]
    total_count = int(len(pixels))

    if total_count == 0:
        return RegionSkinResult(
            name=name,
            total_pixel_count=0,
            valid_pixel_count=0,
            valid_ratio=0.0,
            median_rgb=None,
            median_lab=None,
            reliable=False,
            warnings=[f"No pixels found for {name}; region mask was empty."],
        )

    lab = _to_lab(pixels)
    saturation = _to_saturation(pixels)
    luminance = lab[:, 0]

    lum_low = np.percentile(luminance, LUMINANCE_LOWER_PERCENTILE)
    lum_high = np.percentile(luminance, LUMINANCE_UPPER_PERCENTILE)
    sat_high = np.percentile(saturation, SATURATION_UPPER_PERCENTILE)

    keep = (luminance >= lum_low) & (luminance <= lum_high) & (saturation <= sat_high)
    valid_pixels_rgb = pixels[keep]
    valid_lab = lab[keep]
    valid_count = int(len(valid_pixels_rgb))
    valid_ratio = valid_count / total_count if total_count else 0.0

    if valid_count < MIN_VALID_PIXELS_PER_REGION:
        warnings.append(
            f"{name.replace('_', ' ').title()} has only {valid_count} valid pixels "
            f"after filtering (minimum recommended: {MIN_VALID_PIXELS_PER_REGION})."
        )

    if valid_count == 0:
        return RegionSkinResult(
            name=name,
            total_pixel_count=total_count,
            valid_pixel_count=0,
            valid_ratio=0.0,
            median_rgb=None,
            median_lab=None,
            reliable=False,
            warnings=warnings,
        )

    median_rgb = tuple(np.median(valid_pixels_rgb, axis=0).astype(int).tolist())
    median_lab = tuple(np.median(valid_lab, axis=0).tolist())

    return RegionSkinResult(
        name=name,
        total_pixel_count=total_count,
        valid_pixel_count=valid_count,
        valid_ratio=valid_ratio,
        median_rgb=median_rgb,
        median_lab=median_lab,
        reliable=valid_count >= MIN_VALID_PIXELS_PER_REGION,
        warnings=warnings,
    )


def _region_consistency(region_results: dict) -> float:
    """Return a 0-1 consistency score based on pairwise Lab distance between
    reliable regions' median colors. 1.0 = perfect agreement."""
    reliable_labs = [
        np.array(r.median_lab) for r in region_results.values() if r.reliable and r.median_lab is not None
    ]
    if len(reliable_labs) < 2:
        return 1.0 if reliable_labs else 0.0

    spread = float(np.mean(np.std(np.stack(reliable_labs), axis=0)))
    return float(np.clip(1.0 - spread / REGION_DISAGREEMENT_THRESHOLD, 0.0, 1.0))


def _exclude_outlier_regions(reliable_regions: list) -> tuple:
    """Drop regions whose median color is a strong outlier vs. the others.

    A region that passes the valid-pixel-count check can still be
    contaminated (e.g. a forehead band that lands on a hair fringe/bangs
    instead of skin). When at least 3 regions agree closely, a region far
    from that consensus is more likely contaminated than correct, so it is
    excluded from the final color combination (never used to represent
    skin tone), per the rule against using hair pixels as skin tone.
    """
    if len(reliable_regions) < 3:
        return reliable_regions, []

    labs = np.array([r.median_lab for r in reliable_regions])
    consensus = np.median(labs, axis=0)
    distances = np.linalg.norm(labs - consensus, axis=1)

    kept = []
    warnings = []
    for region, dist in zip(reliable_regions, distances):
        if dist > OUTLIER_LAB_DISTANCE:
            warnings.append(
                f"{region.name.replace('_', ' ').title()} color differs substantially from "
                "other regions (possible hair, shadow, or occlusion contamination) and was "
                "excluded from the final skin tone estimate."
            )
        else:
            kept.append(region)

    if not kept:
        # All regions disagreed heavily; safer to keep them all than return nothing.
        return reliable_regions, []

    return kept, warnings


def extract_skin_tone(image_rgb: np.ndarray, masks: dict) -> SkinToneResult:
    """Extract a representative skin color from masked face regions.

    For each of forehead/left_cheek/right_cheek/jawline: filters out
    shadow/highlight luminance extremes and extreme-saturation pixels, then
    takes the median RGB/Lab. Reliable regions (enough valid pixels) are
    combined (weighted by valid pixel count) into one final skin color.
    """
    warnings: list = []
    region_results = {}
    for name in REGION_NAMES:
        mask = masks.get(name)
        if mask is None:
            continue
        region_results[name] = _extract_region(image_rgb, mask, name)
        warnings.extend(region_results[name].warnings)

    reliable_regions = [r for r in region_results.values() if r.reliable]
    reliable_regions, outlier_warnings = _exclude_outlier_regions(reliable_regions)
    warnings.extend(outlier_warnings)

    if not reliable_regions:
        # Fall back to any region with at least some valid pixels so the app
        # can still show a (low-confidence) result instead of failing outright.
        fallback_regions = [r for r in region_results.values() if r.median_rgb is not None]
        if not fallback_regions:
            warnings.append(
                "Could not extract a reliable skin color from any region. "
                "Try a clearer, well-lit, front-facing photo."
            )
            return SkinToneResult(
                rgb=(0, 0, 0),
                lab=(0.0, 0.0, 0.0),
                region_results=region_results,
                quality_score=0.0,
                region_consistency=0.0,
                avg_valid_pixel_ratio=0.0,
                warnings=warnings,
                success=False,
            )
        warnings.append(
            "No region met the minimum valid-pixel threshold; using best-available regions "
            "with reduced confidence."
        )
        reliable_regions = fallback_regions

    weights = np.array([r.valid_pixel_count for r in reliable_regions], dtype=np.float64)
    weights = weights / weights.sum()

    rgb_stack = np.array([r.median_rgb for r in reliable_regions], dtype=np.float64)
    lab_stack = np.array([r.median_lab for r in reliable_regions], dtype=np.float64)

    final_rgb = tuple(np.round(np.average(rgb_stack, axis=0, weights=weights)).astype(int).tolist())
    final_lab = tuple(np.average(lab_stack, axis=0, weights=weights).tolist())

    consistency = _region_consistency(region_results)
    avg_valid_ratio = float(np.mean([r.valid_ratio for r in region_results.values()])) if region_results else 0.0

    if consistency < 0.5:
        warnings.append(
            "Skin regions disagree noticeably in color (possible uneven lighting or shadows)."
        )

    quality_score = float(np.clip(0.5 * consistency + 0.5 * avg_valid_ratio, 0.0, 1.0))

    return SkinToneResult(
        rgb=final_rgb,
        lab=final_lab,
        region_results=region_results,
        quality_score=quality_score,
        region_consistency=consistency,
        avg_valid_pixel_ratio=avg_valid_ratio,
        warnings=warnings,
        success=True,
    )
