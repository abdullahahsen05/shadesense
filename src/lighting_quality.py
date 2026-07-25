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
    using_face_regions: bool = False
    face_highlight_ratio: float = 0.0
    face_shadow_ratio: float = 0.0
    face_black_clip_ratio: float = 0.0
    face_luminance_spread: float = 0.0
    left_right_gap: float = 0.0
    central_lower_gap: float = 0.0
    face_median_luma: float = 0.0
    worst_region_shadow_ratio: float = 0.0
    low_signal: bool = False
    recapture_recommended: bool = False
    exposure_score: float = 1.0
    uniformity_score: float = 1.0
    contrast_score: float = 1.0
    highlight_score: float = 1.0
    color_score: float = 1.0
    region_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def _region_metric(image_rgb: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = image_rgb[mask > 0].astype(np.float64)
    luma = gray[mask > 0]
    if len(luma) == 0:
        return {}
    p05, p25, p50, p75, p95, p99 = np.percentile(luma, [5, 25, 50, 75, 95, 99])
    means = np.mean(pixels, axis=0)
    channel_average = float(np.mean(means))
    cast_strength = float(np.max(np.abs(means - channel_average))) if channel_average else 0.0
    return {
        "mean_luma": float(np.mean(luma)),
        "median_luma": float(p50),
        "shadow_ratio": float(np.mean(luma < 45)),
        "black_clip_ratio": float(np.mean(luma < 18)),
        "highlight_ratio": float(np.mean(luma > 225)),
        "broad_highlight_ratio": float(np.mean(luma > 205)),
        "luminance_spread": float(p95 - p05),
        "interquartile_range": float(p75 - p25),
        "highlight_gap": float(p99 - p75),
        "color_cast_strength": cast_strength,
        "pixel_count": float(len(luma)),
    }


def _valid_region_masks(masks: dict | None, image_shape: tuple) -> dict[str, np.ndarray]:
    if not masks:
        return {}
    h, w = image_shape[:2]
    return {
        name: np.asarray(mask, dtype=np.uint8)
        for name, mask in masks.items()
        if name != "combined"
        and mask is not None
        and np.asarray(mask).shape == (h, w)
        and np.any(np.asarray(mask) > 0)
    }


def analyze_lighting_quality(
    image_rgb: np.ndarray,
    masks: dict | None = None,
) -> LightingQuality:
    """Assess lighting globally or, when available, on facial skin regions.

    Face-region statistics prevent a bright wall or dark background from
    dominating the lighting decision used by skin extraction.
    """
    if image_rgb is None or image_rgb.size == 0:
        return LightingQuality(score=0.0, warnings=["No image data available."], explanation="No image data available.")

    image = image_rgb.astype(np.float64)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    region_masks = _valid_region_masks(masks, image_rgb.shape)
    region_metrics = {
        name: _region_metric(image_rgb, gray, mask)
        for name, mask in region_masks.items()
    }
    region_metrics = {name: metric for name, metric in region_metrics.items() if metric}
    using_face_regions = bool(region_metrics)

    if using_face_regions:
        combined_mask = np.zeros(gray.shape, dtype=np.uint8)
        for mask in region_masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        sample_gray = gray[combined_mask > 0]
        sample_image = image_rgb[combined_mask > 0].astype(np.float64)
    else:
        sample_gray = gray.reshape(-1)
        sample_image = image.reshape(-1, 3)

    mean_luma = float(np.mean(sample_gray))
    p05, p25, p50, p75, p90, p95, p99 = [
        float(v) for v in np.percentile(sample_gray, [5, 25, 50, 75, 90, 95, 99])
    ]
    shadow_contrast = p95 - p05
    highlight_ratio = float(np.mean(sample_gray > 225))
    broad_highlight_ratio = float(np.mean(sample_gray > 205))
    shadow_ratio = float(np.mean(sample_gray < 45))
    black_clip_ratio = float(np.mean(sample_gray < 18))
    highlight_gap = p99 - p75

    if using_face_regions:
        left_mean = region_metrics.get("left_cheek", {}).get("median_luma", mean_luma)
        right_mean = region_metrics.get("right_cheek", {}).get("median_luma", mean_luma)
        left_right_gap = abs(left_mean - right_mean)
        central_values = [
            region_metrics[name]["median_luma"]
            for name in ("forehead", "left_cheek", "right_cheek")
            if name in region_metrics
        ]
        central_mean = float(np.mean(central_values)) if central_values else mean_luma
        lower_mean = region_metrics.get("jawline", {}).get("median_luma", central_mean)
        central_lower_gap = central_mean - lower_mean
        uneven_gap = max(left_right_gap, abs(central_lower_gap))
    else:
        h, w = gray.shape
        left_mean = float(np.mean(gray[:, : max(1, w // 2)]))
        right_mean = float(np.mean(gray[:, w // 2 :]))
        top_mean = float(np.mean(gray[: max(1, h // 2), :]))
        bottom_mean = float(np.mean(gray[h // 2 :, :]))
        left_right_gap = abs(left_mean - right_mean)
        central_lower_gap = top_mean - bottom_mean
        uneven_gap = max(left_right_gap, abs(central_lower_gap))

    channel_means = np.mean(sample_image.reshape(-1, 3), axis=0)
    channel_avg = float(np.mean(channel_means))
    cast_strength = float(np.max(np.abs(channel_means - channel_avg))) if channel_avg else 0.0

    warnings = []
    if using_face_regions:
        # Absolute face luminance is not a safe proxy for exposure across skin
        # tones. Require substantial near-black clipping when face masks exist.
        cheek_black_clip_evidence = max(
            (
                float(
                    region_metrics[name].get("black_clip_ratio", 0.0)
                )
                for name in ("left_cheek", "right_cheek")
                if name in region_metrics
            ),
            default=black_clip_ratio,
        )
        underexposed = cheek_black_clip_evidence > 0.10 and p95 < 100
    else:
        underexposed = mean_luma < 70 or p75 < 85
    overexposed = mean_luma > 205 or p25 > 185
    uneven_lighting = uneven_gap > (24 if using_face_regions else 36)
    strong_shadow_contrast = shadow_contrast > 185
    strong_highlights = highlight_ratio > 0.025 or (
        broad_highlight_ratio > 0.08 and highlight_gap > 35
    )
    color_cast = cast_strength > 22

    region_shadow_ratios = [
        float(metric.get("shadow_ratio", 0.0))
        for metric in region_metrics.values()
    ]
    worst_region_shadow_ratio = max(region_shadow_ratios, default=shadow_ratio)
    cheek_black_clip_ratios = [
        float(region_metrics[name].get("black_clip_ratio", 0.0))
        for name in ("left_cheek", "right_cheek")
        if name in region_metrics
    ]
    worst_cheek_black_clip_ratio = max(
        cheek_black_clip_ratios,
        default=black_clip_ratio,
    )
    cheek_medians = [
        float(region_metrics[name]["median_luma"])
        for name in ("left_cheek", "right_cheek")
        if name in region_metrics
    ]
    shadowed_cheek = (
        len(cheek_medians) == 2
        and min(cheek_medians) < 100
        and abs(cheek_medians[0] - cheek_medians[1]) > 45
    )
    # Do not call evenly illuminated deep skin "underexposed" based on its
    # absolute luminance alone. Low signal requires clipping or a strongly
    # shadowed region relative to the other side of the same face.
    low_signal = bool(
        underexposed
        or worst_cheek_black_clip_ratio > 0.08
        or shadowed_cheek
    )

    exposure_score = 1.0
    if underexposed or overexposed:
        exposure_score -= 0.50
    if low_signal and not underexposed:
        exposure_score -= 0.45
    exposure_score -= min(shadow_ratio * 1.5, 0.25)
    exposure_score -= min(highlight_ratio * 2.0, 0.20)
    exposure_score = float(np.clip(exposure_score, 0.0, 1.0))

    uniformity_floor = 10.0 if using_face_regions else 18.0
    uniformity_span = 90.0 if using_face_regions else 120.0
    uniformity_score = float(
        np.clip(1.0 - max(uneven_gap - uniformity_floor, 0.0) / uniformity_span, 0.0, 1.0)
    )
    contrast_score = float(
        np.clip(1.0 - max(shadow_contrast - 90.0, 0.0) / 120.0, 0.0, 1.0)
    )
    highlight_score = float(
        np.clip(
            1.0
            - min(highlight_ratio / 0.08, 0.55)
            - min(broad_highlight_ratio / 0.25, 0.35),
            0.0,
            1.0,
        )
    )
    color_score = float(
        np.clip(1.0 - max(cast_strength - 12.0, 0.0) / 32.0, 0.0, 1.0)
    )
    score = float(
        np.clip(
            0.25 * exposure_score
            + 0.30 * uniformity_score
            + 0.15 * contrast_score
            + 0.15 * highlight_score
            + 0.15 * color_score,
            0.20,
            1.0,
        )
    )

    if underexposed:
        warnings.append("Image appears underexposed; shadowed skin pixels may be less reliable.")
    elif low_signal:
        warnings.append(
            "One or more facial regions have low color signal from shadow; "
            "a brighter, more even recapture is recommended."
        )
    if overexposed:
        warnings.append("Image appears overexposed; highlight or glare may shift the extracted tone.")
    if uneven_lighting:
        warnings.append("Lighting appears uneven across the face/image.")
    if strong_shadow_contrast:
        warnings.append("Strong shadow/highlight contrast detected.")
    if strong_highlights:
        warnings.append("Strong facial highlights or glossy shine detected; extracted depth may skew too light.")
    if color_cast:
        warnings.append("Possible color cast detected; white balance may affect shade matching.")

    recapture_recommended = bool(
        underexposed or overexposed or low_signal or score < 0.55
    )
    scope = "facial skin regions" if using_face_regions else "full image"
    explanation = (
        f"Measured on {scope}: median luminance {p50:.0f}/255, shadow range {shadow_contrast:.0f}, "
        f"black-clipped pixels {black_clip_ratio:.1%}, uneven-lighting gap {uneven_gap:.0f}, "
        f"highlight ratio {highlight_ratio:.1%}, "
        f"color-cast strength {cast_strength:.0f}. "
        f"Subscores — exposure {exposure_score:.0%}, uniformity {uniformity_score:.0%}, "
        f"contrast {contrast_score:.0%}, highlights {highlight_score:.0%}, "
        f"color {color_score:.0%}."
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
        using_face_regions=using_face_regions,
        face_highlight_ratio=highlight_ratio,
        face_shadow_ratio=shadow_ratio,
        face_black_clip_ratio=black_clip_ratio,
        face_luminance_spread=shadow_contrast,
        left_right_gap=float(left_right_gap),
        central_lower_gap=float(central_lower_gap),
        face_median_luma=p50,
        worst_region_shadow_ratio=float(worst_region_shadow_ratio),
        low_signal=low_signal,
        recapture_recommended=recapture_recommended,
        exposure_score=exposure_score,
        uniformity_score=uniformity_score,
        contrast_score=contrast_score,
        highlight_score=highlight_score,
        color_score=color_score,
        region_metrics=region_metrics,
    )
