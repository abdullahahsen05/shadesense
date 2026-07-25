"""Optional neutral-card white-balance calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NeutralCardCalibration:
    success: bool
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    confidence: float = 0.0
    sampled_rgb: tuple[int, int, int] | None = None
    crop_fraction: float = 0.35
    warnings: list[str] = field(default_factory=list)


def estimate_neutral_card_calibration(
    reference_rgb: np.ndarray,
    *,
    crop_fraction: float = 0.35,
) -> NeutralCardCalibration:
    """Estimate channel gains from the centre of a neutral gray-card photo."""
    image = np.asarray(reference_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return NeutralCardCalibration(
            success=False,
            warnings=["Neutral reference image is invalid."],
        )
    crop_fraction = float(np.clip(crop_fraction, 0.1, 0.8))
    height, width = image.shape[:2]
    crop_h = max(2, round(height * crop_fraction))
    crop_w = max(2, round(width * crop_fraction))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    pixels = image[top : top + crop_h, left : left + crop_w].reshape(-1, 3)
    luminance = pixels.astype(np.float64).mean(axis=1)
    keep = (luminance >= np.percentile(luminance, 10)) & (
        luminance <= np.percentile(luminance, 90)
    )
    pixels = pixels[keep].astype(np.float64)
    if len(pixels) < 20:
        return NeutralCardCalibration(
            success=False,
            crop_fraction=crop_fraction,
            warnings=["The neutral-card centre crop has too few usable pixels."],
        )
    sampled = np.median(pixels, axis=0)
    mean = float(np.mean(sampled))
    if mean < 35.0 or mean > 235.0:
        return NeutralCardCalibration(
            success=False,
            sampled_rgb=tuple(int(round(value)) for value in sampled),
            crop_fraction=crop_fraction,
            warnings=[
                "The neutral reference is too dark or clipped for reliable calibration."
            ],
        )
    gains = np.clip(mean / np.maximum(sampled, 1.0), 0.65, 1.5)
    local_variation = float(
        np.median(np.std(pixels, axis=0)) / max(mean, 1.0)
    )
    extreme_gain = float(
        max(np.max(np.abs(gains - 1.0)) - 0.30, 0.0)
    )
    confidence = float(
        np.clip(1.0 - 1.8 * local_variation - 0.8 * extreme_gain, 0.0, 1.0)
    )
    warnings = []
    if confidence < 0.55:
        warnings.append(
            "The card crop is uneven; use a larger, evenly lit neutral card."
        )
    return NeutralCardCalibration(
        success=confidence >= 0.35,
        gains=tuple(float(value) for value in gains),
        confidence=confidence,
        sampled_rgb=tuple(int(round(value)) for value in sampled),
        crop_fraction=crop_fraction,
        warnings=warnings,
    )


def apply_neutral_card_calibration(
    image_rgb: np.ndarray,
    calibration: NeutralCardCalibration,
) -> np.ndarray:
    """Apply accepted neutral-card gains without changing the input array."""
    if not calibration.success:
        return np.asarray(image_rgb).copy()
    corrected = np.asarray(image_rgb, dtype=np.float64) * np.asarray(
        calibration.gains,
        dtype=np.float64,
    ).reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)
