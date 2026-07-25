"""Image capture quality diagnostics for uploaded selfies."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ImageQualityResult:
    overall_score: float
    label: str
    blur_score: float
    exposure_score: float
    face_size_score: float
    pose_score: float
    color_cast_score: float
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    blur_metric: float = 0.0
    underexposed_ratio: float = 0.0
    overexposed_ratio: float = 0.0
    face_area_ratio: float | None = None
    pose_asymmetry: float | None = None
    color_cast_strength: float = 0.0
    exposure_source: str = "whole image"


def _score_label(score: float) -> str:
    if score >= 88:
        return "excellent"
    if score >= 72:
        return "good"
    if score >= 50:
        return "fair"
    return "poor"


def _blur_score(image_rgb: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = float(np.clip((variance / 350.0) * 100.0, 0.0, 100.0))
    return score, variance


def _exposure_pixels(
    image_rgb: np.ndarray,
    masks: dict | np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return luminance pixels from facial skin regions when available."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if masks is None:
        return gray.reshape(-1), "whole image"

    if isinstance(masks, dict):
        mask = masks.get("combined")
        if mask is None:
            region_masks = [
                np.asarray(masks[name], dtype=np.uint8)
                for name in ("forehead", "left_cheek", "right_cheek", "jawline")
                if name in masks
            ]
            if region_masks:
                mask = np.bitwise_or.reduce(region_masks)
    else:
        mask = masks

    if mask is None:
        return gray.reshape(-1), "whole image"
    mask_array = np.asarray(mask)
    if mask_array.shape != gray.shape:
        return gray.reshape(-1), "whole image"
    selected = gray[mask_array > 0]
    # Very small masks are not representative enough to replace the fallback.
    if selected.size < 64:
        return gray.reshape(-1), "whole image"
    return selected, "facial skin regions"


def _exposure_score(
    image_rgb: np.ndarray,
    masks: dict | np.ndarray | None = None,
) -> tuple[float, float, float, str]:
    pixels, source = _exposure_pixels(image_rgb, masks)
    under_ratio = float(np.mean(pixels < 35))
    over_ratio = float(np.mean(pixels > 235))
    score = 100.0 - 140.0 * under_ratio - 160.0 * over_ratio
    return float(np.clip(score, 0.0, 100.0)), under_ratio, over_ratio, source


def _face_size_score(image_shape: tuple, landmarks: list | None) -> tuple[float, float | None]:
    if not landmarks:
        return 60.0, None
    h, w = image_shape[:2]
    xs = np.array([p[0] for p in landmarks], dtype=np.float64)
    ys = np.array([p[1] for p in landmarks], dtype=np.float64)
    face_area = max(float(xs.max() - xs.min()), 0.0) * max(float(ys.max() - ys.min()), 0.0)
    image_area = max(float(h * w), 1.0)
    ratio = face_area / image_area
    if ratio < 0.05:
        score = 35.0
    elif ratio < 0.10:
        score = 65.0
    elif ratio > 0.75:
        score = 82.0
    else:
        score = 100.0
    return score, ratio


def _pose_score(landmarks: list | None) -> tuple[float, float | None]:
    if not landmarks:
        return 65.0, None
    points = np.array(landmarks, dtype=np.float64)
    xs = points[:, 0]
    ys = points[:, 1]
    face_width = max(float(xs.max() - xs.min()), 1.0)
    center_x = float(np.median(xs))
    left_count = int(np.sum(xs < center_x))
    right_count = int(np.sum(xs > center_x))
    count_asymmetry = abs(left_count - right_count) / max(left_count + right_count, 1)

    # MediaPipe indices: nose tip 1, eye outer corners roughly 33 and 263.
    landmark_asymmetry = 0.0
    if len(points) > 263:
        nose_x = points[1][0]
        left_eye_x = points[33][0]
        right_eye_x = points[263][0]
        left_span = abs(nose_x - left_eye_x)
        right_span = abs(right_eye_x - nose_x)
        landmark_asymmetry = abs(left_span - right_span) / max(left_span + right_span, 1.0)

    asymmetry = float(max(count_asymmetry, landmark_asymmetry))
    score = float(np.clip(100.0 - asymmetry * 180.0, 30.0, 100.0))
    return score, asymmetry


def _color_cast_score(image_rgb: np.ndarray) -> tuple[float, float]:
    image = image_rgb.astype(np.float64)
    means = np.mean(image.reshape(-1, 3), axis=0)
    avg = float(np.mean(means))
    if avg < 1e-6:
        return 30.0, 0.0
    cast_strength = float(np.max(np.abs(means - avg)))
    score = float(np.clip(100.0 - (cast_strength / 55.0) * 100.0, 0.0, 100.0))
    return score, cast_strength


def analyze_image_quality(
    image_rgb: np.ndarray,
    landmarks: list | None = None,
    masks: dict | np.ndarray | None = None,
) -> ImageQualityResult:
    """Return lightweight image-level quality diagnostics."""
    if image_rgb is None or image_rgb.size == 0:
        return ImageQualityResult(
            overall_score=0.0,
            label="poor",
            blur_score=0.0,
            exposure_score=0.0,
            face_size_score=0.0,
            pose_score=0.0,
            color_cast_score=0.0,
            warnings=["No image data available."],
            reasons=["No image data available."],
        )

    blur_score, blur_metric = _blur_score(image_rgb)
    exposure_score, under_ratio, over_ratio, exposure_source = _exposure_score(
        image_rgb, masks
    )
    face_size_score, face_area_ratio = _face_size_score(image_rgb.shape, landmarks)
    pose_score, pose_asymmetry = _pose_score(landmarks)
    color_cast_score, cast_strength = _color_cast_score(image_rgb)

    warnings: list[str] = []
    reasons: list[str] = []
    if blur_score < 55:
        warnings.append("Image may be slightly blurry.")
    if over_ratio > 0.05:
        warnings.append("Strong highlights or overexposure detected.")
    if under_ratio > 0.12:
        warnings.append("Image appears underexposed.")
    if face_area_ratio is not None and face_area_ratio < 0.10:
        warnings.append("Face appears small in the frame.")
    if pose_asymmetry is not None and pose_asymmetry > 0.22:
        warnings.append("Face appears side-angled; one cheek may be less reliable.")
    if color_cast_score < 65:
        warnings.append("Possible color cast detected.")

    reasons.extend(
        [
            f"Blur metric {blur_metric:.1f}; blur score {blur_score:.0f}/100.",
            f"Underexposed pixels {under_ratio:.1%}; overexposed pixels "
            f"{over_ratio:.1%}, measured on {exposure_source}.",
            "Face size score unavailable without landmarks."
            if face_area_ratio is None
            else f"Face area covers {face_area_ratio:.1%} of the image.",
            "Pose score unavailable without landmarks."
            if pose_asymmetry is None
            else f"Pose asymmetry heuristic {pose_asymmetry:.2f}.",
            f"Color-cast strength {cast_strength:.1f}; color-cast score {color_cast_score:.0f}/100.",
        ]
    )

    overall = float(
        np.clip(
            0.25 * blur_score
            + 0.25 * exposure_score
            + 0.20 * face_size_score
            + 0.15 * pose_score
            + 0.15 * color_cast_score,
            0.0,
            100.0,
        )
    )
    return ImageQualityResult(
        overall_score=overall,
        label=_score_label(overall),
        blur_score=blur_score,
        exposure_score=exposure_score,
        face_size_score=face_size_score,
        pose_score=pose_score,
        color_cast_score=color_cast_score,
        warnings=warnings,
        reasons=reasons,
        blur_metric=blur_metric,
        underexposed_ratio=under_ratio,
        overexposed_ratio=over_ratio,
        face_area_ratio=face_area_ratio,
        pose_asymmetry=pose_asymmetry,
        color_cast_strength=cast_strength,
        exposure_source=exposure_source,
    )
