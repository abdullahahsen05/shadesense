"""Debug visualization helpers: landmarks, masks, swatches (built up across phases)."""

import cv2
import numpy as np


def draw_face_landmarks(image_rgb: np.ndarray, landmarks: list, radius: int = 1) -> np.ndarray:
    """Draw small dots at each landmark position on a copy of the image."""
    overlay = image_rgb.copy()
    for x, y in landmarks:
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, (0, 255, 0), -1)
    return overlay


REGION_COLORS = {
    "forehead": (255, 0, 0),
    "left_cheek": (0, 200, 255),
    "right_cheek": (255, 0, 255),
    "jawline": (0, 255, 0),
}


def draw_region_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    """Blend a single binary mask over the image as a colored overlay."""
    overlay = image_rgb.copy().astype(np.float64)
    color_layer = np.zeros_like(overlay)
    color_layer[:, :] = color
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color_layer[mask_bool] * alpha
    return overlay.astype(np.uint8)


def draw_all_region_masks(image_rgb: np.ndarray, masks: dict, alpha: float = 0.45) -> np.ndarray:
    """Blend all named region masks over the image, each with its own color."""
    overlay = image_rgb.copy()
    for name, color in REGION_COLORS.items():
        if name in masks:
            overlay = draw_region_mask(overlay, masks[name], color=color, alpha=alpha)
    return overlay


def make_skin_swatch(rgb: tuple, size: int = 150) -> np.ndarray:
    """Create a flat-color swatch image for a given RGB color."""
    swatch = np.zeros((size, size, 3), dtype=np.uint8)
    swatch[:, :] = rgb
    return swatch
