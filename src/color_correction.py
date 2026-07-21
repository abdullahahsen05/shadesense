"""Mild lighting/white-balance correction applied before skin extraction."""

import cv2
import numpy as np


def apply_mild_color_correction(image_rgb: np.ndarray) -> tuple:
    """Apply a mild gray-world white balance and light CLAHE on luminance.

    This is intentionally conservative (not a full auto white-balance) since
    aggressive correction can distort skin tone signal we later rely on.

    Returns:
        (corrected_image_rgb, notes) where notes is a list of strings
        describing what corrections were applied.
    """
    notes = []
    img = image_rgb.astype(np.float64)

    # --- Mild gray-world white balance ---
    mean_r = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_b = img[:, :, 2].mean()
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    if min(mean_r, mean_g, mean_b) > 1e-6:
        gain_r = mean_gray / mean_r
        gain_g = mean_gray / mean_g
        gain_b = mean_gray / mean_b

        # Dampen the correction so it is mild, not a full normalization.
        strength = 0.5
        gain_r = 1.0 + (gain_r - 1.0) * strength
        gain_g = 1.0 + (gain_g - 1.0) * strength
        gain_b = 1.0 + (gain_b - 1.0) * strength

        # Clamp gains to avoid extreme shifts on unusual images.
        gain_r, gain_g, gain_b = (float(np.clip(g, 0.7, 1.4)) for g in (gain_r, gain_g, gain_b))

        img[:, :, 0] *= gain_r
        img[:, :, 1] *= gain_g
        img[:, :, 2] *= gain_b
        notes.append(
            f"Applied mild gray-world white balance (gains r={gain_r:.2f}, "
            f"g={gain_g:.2f}, b={gain_b:.2f})."
        )

    corrected = np.clip(img, 0, 255).astype(np.uint8)

    # --- Mild CLAHE on luminance channel only (preserves color/hue) ---
    lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    notes.append("Applied mild CLAHE (clip=1.5) on luminance to reduce shadow/highlight extremes.")

    return corrected, notes
