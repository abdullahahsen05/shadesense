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

    margin_h = h * 0.02

    # --- Forehead: oval points above the eyebrow line, plus the eyebrow
    # line itself as the lower boundary. ---
    forehead_oval = oval_pts[oval_pts[:, 1] <= eyebrow_y] if len(oval_pts) else np.empty((0, 2))
    forehead_points = np.vstack([forehead_oval, eyebrow_pts]) if len(eyebrow_pts) else forehead_oval
    forehead_mask = _mask_from_points(forehead_points, image_shape)

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

    # --- Jawline: oval points below the mouth-bottom line (the chin/jaw arc). ---
    jaw_oval = oval_pts[oval_pts[:, 1] >= mouth_bottom_y + margin_h] if len(oval_pts) else np.empty((0, 2))
    jawline_mask = _mask_from_points(jaw_oval, image_shape)

    # Remove eyes/eyebrows/lips from every region as a safety net in case the
    # convex hulls above happen to bulge into them.
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    for pts in (eye_pts, eyebrow_pts, lips_pts):
        if len(pts) >= MIN_POLYGON_POINTS:
            hull = cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(exclude_mask, hull, 255)
    # Dilate the exclusion zone slightly for a safety margin.
    exclude_mask = cv2.dilate(exclude_mask, np.ones((5, 5), np.uint8), iterations=1)
    keep_mask = cv2.bitwise_not(exclude_mask)

    forehead_mask = cv2.bitwise_and(forehead_mask, keep_mask)
    left_cheek_mask = cv2.bitwise_and(left_cheek_mask, keep_mask)
    right_cheek_mask = cv2.bitwise_and(right_cheek_mask, keep_mask)
    jawline_mask = cv2.bitwise_and(jawline_mask, keep_mask)

    combined = forehead_mask | left_cheek_mask | right_cheek_mask | jawline_mask

    return {
        "forehead": forehead_mask,
        "left_cheek": left_cheek_mask,
        "right_cheek": right_cheek_mask,
        "jawline": jawline_mask,
        "combined": combined,
    }
