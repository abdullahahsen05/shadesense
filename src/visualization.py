"""Debug visualization helpers: landmarks, masks, swatches (built up across phases)."""

import cv2
import numpy as np


def draw_face_landmarks(image_rgb: np.ndarray, landmarks: list, radius: int = 1) -> np.ndarray:
    """Draw small dots at each landmark position on a copy of the image."""
    overlay = image_rgb.copy()
    for x, y in landmarks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, (0, 255, 0), -1)
    return overlay
