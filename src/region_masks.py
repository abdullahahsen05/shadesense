"""Cheek, forehead, and jawline region mask construction.

Rather than hardcoding bespoke polygon vertex lists for "cheek"/"forehead"/
"jawline" (error-prone to get exactly right), this module combines a small
set of well-established MediaPipe Face Mesh landmark index groups (face
oval, eyes, eyebrows, lips, nose) with simple geometric derivation
(y/x thresholds + convex hulls, computed from the actual detected
coordinates) to build each region. This keeps regions anchored to real
facial features and naturally away from eyes/lips/eyebrows/background.
"""

import cv2
import numpy as np

# Well-established MediaPipe Face Mesh (478-point) index groups.
# Left/right MediaPipe naming refers to the subject's anatomical side, which
# is ambiguous when mirrored — we don't rely on that distinction here and
# instead split left/right by actual x-coordinate at runtime.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
]
NOSE_INDICES = [1, 2, 4, 5, 6, 168, 197, 195, 45, 275, 98, 327]

CHIN_TIP_IDX = 152
FOREHEAD_TOP_IDX = 10

MIN_POLYGON_POINTS = 3


def _pts(landmarks, indices):
    return np.array([landmarks[i] for i in indices if i < len(landmarks)], dtype=np.float64)


def _mask_from_points(points: np.ndarray, image_shape) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < MIN_POLYGON_POINTS:
        return mask
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _x_at_y(points: np.ndarray, y: float, side: str, fallback: float) -> float:
    """Estimate the face boundary x-coordinate near a y position."""
    if len(points) == 0:
        return fallback
    band = points[np.abs(points[:, 1] - y) <= max(8.0, 0.08 * np.ptp(points[:, 1]))]
    if len(band) == 0:
        band = points
    return float(np.min(band[:, 0]) if side == "left" else np.max(band[:, 0]))


def _cheek_polygon_mask(
    oval_pts: np.ndarray,
    face_center_x: float,
    face_width: float,
    upper_y: float,
    lower_y: float,
    side: str,
    image_shape,
) -> np.ndarray:
    """Build a symmetric cheek polygon from face geometry.

    Smile/front-facing expressions can leave too few oval landmarks in the
    cheek band on one side. This geometry fallback keeps both cheeks similarly
    sized when visible while later exclusion masks still remove nose/lips/eyes.
    """
    sign = -1.0 if side == "left" else 1.0
    mid_y = (upper_y + lower_y) / 2.0
    boundary_fallback = face_center_x + sign * 0.43 * face_width
    outer_x = _x_at_y(oval_pts, mid_y, side, boundary_fallback)
    inner_x = face_center_x + sign * 0.08 * face_width
    mid_inner_x = face_center_x + sign * 0.16 * face_width

    points = np.array(
        [
            [inner_x, upper_y],
            [outer_x, upper_y + 0.08 * (lower_y - upper_y)],
            [outer_x, lower_y],
            [mid_inner_x, lower_y - 0.05 * (lower_y - upper_y)],
        ],
        dtype=np.float64,
    )
    return _mask_from_points(points, image_shape)


def _side_jaw_mask(
    oval_pts: np.ndarray,
    face_center_x: float,
    face_width: float,
    upper_y: float,
    lower_y: float,
    side: str,
    image_shape,
) -> np.ndarray:
    """Build a side-jaw sampling band while avoiding central chin/neck."""
    sign = -1.0 if side == "left" else 1.0
    mid_y = (upper_y + lower_y) / 2.0
    outer_x = _x_at_y(oval_pts, mid_y, side, face_center_x + sign * 0.44 * face_width)
    inner_upper_x = face_center_x + sign * 0.26 * face_width
    inner_lower_x = face_center_x + sign * 0.20 * face_width
    points = np.array(
        [
            [inner_upper_x, upper_y],
            [outer_x, upper_y],
            [outer_x, lower_y],
            [inner_lower_x, lower_y],
        ],
        dtype=np.float64,
    )
    return _mask_from_points(points, image_shape)


def _inset_region_mask(mask: np.ndarray, face_width: float, strength: float = 0.006) -> np.ndarray:
    """Inset mask boundaries using a face-relative margin.

    Landmark polygons can touch hair, facial-feature, or background edges.
    A small scale-aware erosion is more consistent than a fixed pixel margin
    across phone images of different resolutions.
    """
    radius = int(np.clip(round(face_width * strength), 1, 5))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded if np.any(eroded) else mask


def _expanded_eye_zone(landmarks, image_shape, face_width: float, face_height: float) -> np.ndarray:
    """Return a conservative glasses/reflection exclusion band."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    eye_points = _pts(landmarks, EYE_INDICES)
    if len(eye_points) < MIN_POLYGON_POINTS:
        return mask
    x0 = int(np.clip(np.min(eye_points[:, 0]) - 0.06 * face_width, 0, w - 1))
    x1 = int(np.clip(np.max(eye_points[:, 0]) + 0.06 * face_width, 0, w - 1))
    y0 = int(np.clip(np.min(eye_points[:, 1]) - 0.04 * face_height, 0, h - 1))
    y1 = int(np.clip(np.max(eye_points[:, 1]) + 0.10 * face_height, 0, h - 1))
    if x1 > x0 and y1 > y0:
        mask[y0 : y1 + 1, x0 : x1 + 1] = 255
    return mask


def refine_masks_for_capture(
    image_rgb: np.ndarray,
    masks: dict,
    landmarks,
    pose_asymmetry: float | None = None,
) -> tuple[dict, dict]:
    """Remove likely eyewear reflections and reduce a foreshortened cheek.

    Refinement only removes pixels and falls back to the original mask if a
    candidate edit would erase too much of a region.
    """
    refined = {
        name: np.asarray(mask, dtype=np.uint8).copy()
        for name, mask in masks.items()
    }
    diagnostics = {
        "eyewear_reflection_detected": False,
        "eye_zone_edge_density": 0.0,
        "eye_zone_reflection_ratio": 0.0,
        "pose_reduced_region": None,
        "warnings": [],
    }
    if image_rgb is None or image_rgb.size == 0 or not landmarks:
        return refined, diagnostics

    points = np.asarray(landmarks, dtype=np.float64)
    face_width = max(float(np.ptp(points[:, 0])), 1.0)
    face_height = max(float(np.ptp(points[:, 1])), 1.0)
    eye_zone = _expanded_eye_zone(landmarks, image_rgb.shape, face_width, face_height)
    zone_pixels = eye_zone > 0
    if np.any(zone_pixels):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        edges = cv2.Canny(gray, 60, 150)
        edge_density = float(np.mean(edges[zone_pixels] > 0))
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        neutral_glare = (value > 205) & (saturation < 100)
        colored_glare = (
            (value > 135)
            & (saturation > 45)
            & (hsv[:, :, 0] >= 30)
            & (hsv[:, :, 0] <= 105)
        )
        reflection_ratio = float(
            np.mean((neutral_glare | colored_glare)[zone_pixels])
        )
        eyewear_detected = bool(
            edge_density > 0.075
            and (reflection_ratio > 0.012 or edge_density > 0.14)
        )
        diagnostics.update(
            {
                "eyewear_reflection_detected": eyewear_detected,
                "eye_zone_edge_density": edge_density,
                "eye_zone_reflection_ratio": reflection_ratio,
            }
        )
        if eyewear_detected:
            keep = cv2.bitwise_not(eye_zone)
            for name in ("left_cheek", "right_cheek"):
                candidate = cv2.bitwise_and(refined[name], keep)
                if np.count_nonzero(candidate) >= 0.45 * max(
                    np.count_nonzero(refined[name]), 1
                ):
                    refined[name] = candidate
            diagnostics["warnings"].append(
                "Eyewear edges or lens reflections were detected; upper-cheek "
                "pixels near the frames were excluded."
            )

    if pose_asymmetry is not None and pose_asymmetry > 0.18 and len(points) > 263:
        nose_x = float(points[1, 0])
        left_span = abs(nose_x - float(points[33, 0]))
        right_span = abs(float(points[263, 0]) - nose_x)
        reduced_name = "left_cheek" if left_span < right_span else "right_cheek"
        radius = int(np.clip(round(face_width * 0.018), 2, 10))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        candidate = cv2.erode(refined[reduced_name], kernel, iterations=1)
        if np.count_nonzero(candidate) >= 0.35 * max(
            np.count_nonzero(refined[reduced_name]), 1
        ):
            refined[reduced_name] = candidate
            diagnostics["pose_reduced_region"] = reduced_name
            diagnostics["warnings"].append(
                f"{reduced_name.replace('_', ' ').title()} evidence was reduced "
                "because the cheek is foreshortened by pose."
            )

    refined["combined"] = (
        refined["forehead"]
        | refined["left_cheek"]
        | refined["right_cheek"]
        | refined["jawline"]
    )
    return refined, diagnostics


def build_region_masks(image_shape, landmarks) -> dict:
    """Build cheek, forehead, and jawline masks from face landmarks.

    Args:
        image_shape: (h, w, ...) shape of the source image.
        landmarks: list of (x, y) pixel-space landmark coordinates
            (as produced by `face_detection.detect_face_landmarks`).

    Returns:
        dict with keys: "forehead", "left_cheek", "right_cheek", "jawline",
        "combined" (union of all four), each a uint8 mask (0/255) matching
        image_shape[:2].
    """
    h, w = image_shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)

    if landmarks is None or len(landmarks) < 50:
        return {
            "forehead": empty,
            "left_cheek": empty.copy(),
            "right_cheek": empty.copy(),
            "jawline": empty.copy(),
            "combined": empty.copy(),
        }

    oval_pts = _pts(landmarks, FACE_OVAL)
    eye_pts = _pts(landmarks, EYE_INDICES)
    eyebrow_pts = _pts(landmarks, EYEBROW_INDICES)
    lips_pts = _pts(landmarks, LIPS_INDICES)
    nose_pts = _pts(landmarks, NOSE_INDICES)

    eye_y = float(np.mean(eye_pts[:, 1])) if len(eye_pts) else h * 0.4
    eyebrow_y = float(np.mean(eyebrow_pts[:, 1])) if len(eyebrow_pts) else eye_y - h * 0.05
    mouth_top_y = float(np.min(lips_pts[:, 1])) if len(lips_pts) else h * 0.65
    mouth_bottom_y = float(np.max(lips_pts[:, 1])) if len(lips_pts) else h * 0.7
    face_center_x = float(np.mean(oval_pts[:, 0])) if len(oval_pts) else w / 2.0
    forehead_top_y = float(landmarks[FOREHEAD_TOP_IDX][1]) if len(landmarks) > FOREHEAD_TOP_IDX else eyebrow_y - h * 0.25
    chin_y = float(landmarks[CHIN_TIP_IDX][1]) if len(landmarks) > CHIN_TIP_IDX else mouth_bottom_y + h * 0.1
    face_height = max(chin_y - forehead_top_y, 1.0)
    face_width = float(np.max(oval_pts[:, 0]) - np.min(oval_pts[:, 0])) if len(oval_pts) else w * 0.5

    margin_h = h * 0.02

    # --- Forehead: a band anchored just above the eyebrow line, not the
    # raw hairline landmark. The hairline/oval-top landmark sits at or
    # behind the hair itself for people with bangs/fringes, so anchoring
    # there risks sampling hair pixels instead of skin. A band anchored to
    # the (much more reliable) eyebrow line, inset horizontally away from
    # the temples/hairline sides, stays on bare forehead skin for the vast
    # majority of hairstyles.
    forehead_bottom_y = eyebrow_y - margin_h
    forehead_top_y_bound = eyebrow_y - 0.28 * face_height
    forehead_left_x = face_center_x - 0.30 * face_width
    forehead_right_x = face_center_x + 0.30 * face_width
    forehead_rect_pts = np.array(
        [
            [forehead_left_x, forehead_top_y_bound],
            [forehead_right_x, forehead_top_y_bound],
            [forehead_right_x, forehead_bottom_y],
            [forehead_left_x, forehead_bottom_y],
        ]
    )
    forehead_mask = _mask_from_points(forehead_rect_pts, image_shape)

    # --- Cheeks: oval points in the vertical band between eye line and
    # mouth-top line, split left/right by face center x, plus the nearby
    # nose point on that side as the inner boundary. ---
    band_mask = (oval_pts[:, 1] >= eye_y + margin_h) & (oval_pts[:, 1] <= mouth_top_y - margin_h) if len(oval_pts) else np.array([], dtype=bool)
    cheek_band_pts = oval_pts[band_mask] if len(oval_pts) else np.empty((0, 2))

    left_oval = cheek_band_pts[cheek_band_pts[:, 0] < face_center_x]
    right_oval = cheek_band_pts[cheek_band_pts[:, 0] >= face_center_x]

    nose_left = nose_pts[nose_pts[:, 0] < face_center_x] if len(nose_pts) else np.empty((0, 2))
    nose_right = nose_pts[nose_pts[:, 0] >= face_center_x] if len(nose_pts) else np.empty((0, 2))

    left_cheek_points = np.vstack([left_oval, nose_left]) if len(left_oval) else left_oval
    right_cheek_points = np.vstack([right_oval, nose_right]) if len(right_oval) else right_oval

    left_cheek_mask = _mask_from_points(left_cheek_points, image_shape)
    right_cheek_mask = _mask_from_points(right_cheek_points, image_shape)

    cheek_upper_y = eye_y + 0.035 * face_height
    cheek_lower_y = mouth_top_y - 0.035 * face_height
    if cheek_lower_y > cheek_upper_y:
        left_cheek_mask = cv2.bitwise_or(
            left_cheek_mask,
            _cheek_polygon_mask(
                oval_pts, face_center_x, face_width, cheek_upper_y, cheek_lower_y, "left", image_shape
            ),
        )
        right_cheek_mask = cv2.bitwise_or(
            right_cheek_mask,
            _cheek_polygon_mask(
                oval_pts, face_center_x, face_width, cheek_upper_y, cheek_lower_y, "right", image_shape
            ),
        )

    # --- Jawline: side-jaw skin, not central chin/under-chin. The older
    # under-mouth convex hull could pick up neck shadow or the central chin
    # crease. These side bands prefer lower-cheek/side-jaw skin and leave the
    # central chin area out unless future explicit landmarks justify it.
    jaw_upper_y = mouth_top_y + 0.04 * face_height
    jaw_lower_y = min(mouth_bottom_y + 0.10 * face_height, chin_y - 0.08 * face_height)
    if jaw_lower_y > jaw_upper_y:
        jawline_mask = cv2.bitwise_or(
            _side_jaw_mask(oval_pts, face_center_x, face_width, jaw_upper_y, jaw_lower_y, "left", image_shape),
            _side_jaw_mask(oval_pts, face_center_x, face_width, jaw_upper_y, jaw_lower_y, "right", image_shape),
        )
    else:
        jawline_mask = empty.copy()

    # Remove eyes/eyebrows/lips from every region as a safety net in case the
    # convex hulls above happen to bulge into them.
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    for pts in (eye_pts, eyebrow_pts, lips_pts, nose_pts):
        if len(pts) >= MIN_POLYGON_POINTS:
            hull = cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(exclude_mask, hull, 255)
    # Dilate the exclusion zone by a face-relative margin for consistent
    # feature avoidance at different input resolutions.
    exclude_radius = int(np.clip(round(face_width * 0.012), 2, 7))
    exclude_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (exclude_radius * 2 + 1, exclude_radius * 2 + 1),
    )
    exclude_mask = cv2.dilate(exclude_mask, exclude_kernel, iterations=1)
    keep_mask = cv2.bitwise_not(exclude_mask)

    forehead_mask = cv2.bitwise_and(
        _inset_region_mask(forehead_mask, face_width, strength=0.008),
        keep_mask,
    )
    left_cheek_mask = cv2.bitwise_and(
        _inset_region_mask(left_cheek_mask, face_width),
        keep_mask,
    )
    right_cheek_mask = cv2.bitwise_and(
        _inset_region_mask(right_cheek_mask, face_width),
        keep_mask,
    )
    jawline_mask = cv2.bitwise_and(
        _inset_region_mask(jawline_mask, face_width, strength=0.005),
        keep_mask,
    )

    combined = forehead_mask | left_cheek_mask | right_cheek_mask | jawline_mask

    return {
        "forehead": forehead_mask,
        "left_cheek": left_cheek_mask,
        "right_cheek": right_cheek_mask,
        "jawline": jawline_mask,
        "combined": combined,
    }
