"""Validation gates for images submitted to the shade-analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.config import MIN_FACE_SIZE_RATIO


MIN_IMAGE_SIDE = 60
MIN_LANDMARK_COUNT = 468
MIN_IN_FRAME_LANDMARK_RATIO = 0.90
MIN_FACE_ASPECT_RATIO = 0.40
MAX_FACE_ASPECT_RATIO = 1.85
BLANK_LUMINANCE_RANGE = 8.0
BLANK_LUMINANCE_STD = 2.0


@dataclass(frozen=True)
class InputValidationResult:
    valid: bool
    code: str
    message: str
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def validate_image_content(image_rgb: np.ndarray) -> InputValidationResult:
    """Reject malformed, undersized, or effectively blank RGB images."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        return InputValidationResult(
            valid=False,
            code="invalid_image_shape",
            message="The upload is not a valid three-channel RGB image.",
            diagnostics={"shape": tuple(image.shape)},
        )

    height, width = image.shape[:2]
    if height < MIN_IMAGE_SIDE or width < MIN_IMAGE_SIDE:
        return InputValidationResult(
            valid=False,
            code="image_too_small",
            message=(
                "The image is too small for reliable face and skin analysis. "
                f"Use an image at least {MIN_IMAGE_SIDE}×{MIN_IMAGE_SIDE} pixels."
            ),
            diagnostics={"width": width, "height": height},
        )

    numeric = image.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        return InputValidationResult(
            valid=False,
            code="invalid_pixel_values",
            message="The image contains invalid pixel values and cannot be analysed.",
            diagnostics={"width": width, "height": height},
        )

    luminance = (
        0.2126 * numeric[..., 0]
        + 0.7152 * numeric[..., 1]
        + 0.0722 * numeric[..., 2]
    )
    low, high = np.percentile(luminance, [1.0, 99.0])
    luminance_range = float(high - low)
    luminance_std = float(np.std(luminance))
    diagnostics = {
        "width": width,
        "height": height,
        "luminance_range_p1_p99": luminance_range,
        "luminance_std": luminance_std,
    }
    if (
        luminance_range < BLANK_LUMINANCE_RANGE
        and luminance_std < BLANK_LUMINANCE_STD
    ):
        return InputValidationResult(
            valid=False,
            code="blank_or_uniform_image",
            message=(
                "The upload appears blank or nearly uniform. "
                "Upload a clear photograph containing one human face."
            ),
            diagnostics=diagnostics,
        )

    return InputValidationResult(
        valid=True,
        code="image_content_valid",
        message="Image content is valid for human-subject detection.",
        diagnostics=diagnostics,
    )


def validate_human_subject(face_result, image_shape: tuple) -> InputValidationResult:
    """Require exactly one plausible, sufficiently large human face."""
    if not getattr(face_result, "success", False):
        detector_error = str(getattr(face_result, "error", "") or "")
        if detector_error.startswith("Face detection failed:"):
            return InputValidationResult(
                valid=False,
                code="face_detector_error",
                message=detector_error,
            )
        return InputValidationResult(
            valid=False,
            code="no_human_face",
            message=(
                "No human face was detected. Cars, objects, animals, scenery, "
                "and blank images cannot be used; upload one clear human portrait."
            ),
        )

    face_count = int(getattr(face_result, "face_count", 0))
    if face_count != 1:
        return InputValidationResult(
            valid=False,
            code="multiple_human_faces",
            message=(
                f"{face_count} human faces were detected. "
                "Upload a photo containing exactly one person so the shade result "
                "cannot be assigned to the wrong face."
            ),
            diagnostics={"face_count": face_count},
        )

    landmarks = getattr(face_result, "landmarks", None) or []
    if len(landmarks) < MIN_LANDMARK_COUNT:
        return InputValidationResult(
            valid=False,
            code="incomplete_face_landmarks",
            message=(
                "A complete human face could not be verified. "
                "Use a clearer, front-facing portrait."
            ),
            diagnostics={"landmark_count": len(landmarks)},
        )

    height, width = image_shape[:2]
    points = np.asarray(landmarks, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    in_frame = (
        finite
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    in_frame_ratio = float(np.mean(in_frame))
    if not np.any(finite):
        in_frame_ratio = 0.0
        face_width = 0.0
        face_height = 0.0
    else:
        finite_points = points[finite]
        face_width = float(np.ptp(finite_points[:, 0]))
        face_height = float(np.ptp(finite_points[:, 1]))

    width_ratio = face_width / max(float(width), 1.0)
    height_ratio = face_height / max(float(height), 1.0)
    aspect_ratio = face_width / max(face_height, 1e-6)
    diagnostics = {
        "face_count": face_count,
        "landmark_count": len(landmarks),
        "in_frame_landmark_ratio": in_frame_ratio,
        "face_width_ratio": width_ratio,
        "face_height_ratio": height_ratio,
        "face_aspect_ratio": aspect_ratio,
    }

    if in_frame_ratio < MIN_IN_FRAME_LANDMARK_RATIO:
        return InputValidationResult(
            valid=False,
            code="face_out_of_frame",
            message=(
                "The detected face is substantially cropped or outside the image. "
                "Keep the full forehead, both cheeks, and side jaw visible."
            ),
            diagnostics=diagnostics,
        )

    if width_ratio < MIN_FACE_SIZE_RATIO or height_ratio < MIN_FACE_SIZE_RATIO:
        return InputValidationResult(
            valid=False,
            code="face_too_small",
            message=(
                "A human face was detected, but it is too small for dependable "
                "skin-color measurement. Move closer and retake the photo."
            ),
            diagnostics=diagnostics,
        )

    if not MIN_FACE_ASPECT_RATIO <= aspect_ratio <= MAX_FACE_ASPECT_RATIO:
        return InputValidationResult(
            valid=False,
            code="implausible_face_geometry",
            message=(
                "The detected landmark geometry is not reliable enough for a "
                "human facial skin analysis. Upload a clearer portrait."
            ),
            diagnostics=diagnostics,
        )

    return InputValidationResult(
        valid=True,
        code="single_human_face_valid",
        message="Exactly one sufficiently visible human face was verified.",
        diagnostics=diagnostics,
    )
