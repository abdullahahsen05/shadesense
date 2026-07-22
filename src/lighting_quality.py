"""Lighting quality diagnostics for local image assessment."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LightingQuality:
    score: float
    warnings: list[str] = field(default_factory=list)
    explanation: str = ""
    underexposed: bool = False
    overexposed: bool = False
    uneven_lighting: bool = False
    strong_shadow_contrast: bool = False
    strong_highlights: bool = False
    color_cast: bool = False


def analyze_lighting_quality(image_rgb: np.ndarray) -> LightingQuality:
    """Return a gentle lighting-quality score and caveats for confidence."""
    if image_rgb is None or image_rgb.size == 0:
        return LightingQuality(score=0.0, warnings=["No image data available."], explanation="No image data available.")

    image = image_rgb.astype(np.float64)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mean_luma = float(np.mean(gray))
    p05, p25, p75, p90, p95, p99 = [float(v) for v in np.percentile(gray, [5, 25, 75, 90, 95, 99])]
    shadow_contrast = p95 - p05
    highlight_ratio = float(np.mean(gray > 225))
    broad_highlight_ratio = float(np.mean(gray > 205))
    highlight_gap = p99 - p75

    h, w = gray.shape
    left_mean = float(np.mean(gray[:, : max(1, w // 2)]))
    right_mean = float(np.mean(gray[:, w // 2 :]))
    top_mean = float(np.mean(gray[: max(1, h // 2), :]))
    bottom_mean = float(np.mean(gray[h // 2 :, :]))
    uneven_gap = max(abs(left_mean - right_mean), abs(top_mean - bottom_mean))

    channel_means = np.mean(image.reshape(-1, 3), axis=0)
    channel_avg = float(np.mean(channel_means))
    cast_strength = float(np.max(np.abs(channel_means - channel_avg))) if channel_avg else 0.0

    warnings = []
    score = 1.0
    underexposed = mean_luma < 70 or p75 < 85
    overexposed = mean_luma > 205 or p25 > 185
    uneven_lighting = uneven_gap > 36
    strong_shadow_contrast = shadow_contrast > 185
    strong_highlights = highlight_ratio > 0.025 or (broad_highlight_ratio > 0.08 and highlight_gap > 35)
    color_cast = cast_strength > 22

    if underexposed:
        score -= 0.20
        warnings.append("Image appears underexposed; shadowed skin pixels may be less reliable.")
    if overexposed:
        score -= 0.20
        warnings.append("Image appears overexposed; highlight or glare may shift the extracted tone.")
    if uneven_lighting:
        score -= 0.16
        warnings.append("Lighting appears uneven across the face/image.")
    if strong_shadow_contrast:
        score -= 0.14
        warnings.append("Strong shadow/highlight contrast detected.")
    if strong_highlights:
        score -= 0.16
        warnings.append("Strong facial highlights or glossy shine detected; extracted depth may skew too light.")
    if color_cast:
        score -= 0.12
        warnings.append("Possible color cast detected; white balance may affect shade matching.")

    score = float(np.clip(score, 0.25, 1.0))
    explanation = (
        f"Mean luminance {mean_luma:.0f}/255, shadow range {shadow_contrast:.0f}, "
        f"uneven-lighting gap {uneven_gap:.0f}, highlight ratio {highlight_ratio:.1%}, "
        f"color-cast strength {cast_strength:.0f}."
    )
    if not warnings:
        explanation += " Lighting looks suitable for extraction."

    return LightingQuality(
        score=score,
        warnings=warnings,
        explanation=explanation,
        underexposed=underexposed,
        overexposed=overexposed,
        uneven_lighting=uneven_lighting,
        strong_shadow_contrast=strong_shadow_contrast,
        strong_highlights=strong_highlights,
        color_cast=color_cast,
    )
