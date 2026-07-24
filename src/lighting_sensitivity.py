"""Deterministic sensitivity analysis for plausible capture-color variation."""

from dataclasses import dataclass, field

import numpy as np
from skimage.color import deltaE_ciede2000

from src.skin_extraction import extract_skin_tone


EXPOSURE_EV = 0.12
WHITE_BALANCE_GAIN = 0.035
GAMMA_DELTA = 0.06
STABLE_DELTA_E_P90 = 3.0
CAUTION_DELTA_E_P90 = 6.0


@dataclass(frozen=True)
class LightingSensitivityResult:
    variant_labs: list[tuple] = field(default_factory=list)
    score: float = 0.0
    delta_e_p90: float = 12.0
    max_delta_e: float = 12.0
    attempted_variants: int = 0
    successful_variants: int = 0
    stable: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_diagnostics(self) -> dict:
        return {
            "score": self.score,
            "delta_e_p90": self.delta_e_p90,
            "max_delta_e": self.max_delta_e,
            "attempted_variants": self.attempted_variants,
            "successful_variants": self.successful_variants,
            "stable": self.stable,
            "method": (
                "deterministic exposure, white-balance, and gamma perturbation "
                "re-extraction"
            ),
        }


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def _exposure_variant(image_rgb: np.ndarray, ev: float) -> np.ndarray:
    normalized = image_rgb.astype(np.float64) / 255.0
    linear = _srgb_to_linear(normalized)
    adjusted = _linear_to_srgb(np.clip(linear * (2.0**ev), 0.0, 1.0))
    return np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)


def _white_balance_variant(image_rgb: np.ndarray, warm: bool) -> np.ndarray:
    gain = WHITE_BALANCE_GAIN
    gains = np.array(
        [1.0 + gain, 1.0, 1.0 - gain]
        if warm
        else [1.0 - gain, 1.0, 1.0 + gain],
        dtype=np.float64,
    )
    adjusted = image_rgb.astype(np.float64) * gains.reshape(1, 1, 3)
    return np.clip(np.rint(adjusted), 0, 255).astype(np.uint8)


def _gamma_variant(image_rgb: np.ndarray, gamma: float) -> np.ndarray:
    normalized = image_rgb.astype(np.float64) / 255.0
    adjusted = np.power(np.clip(normalized, 0.0, 1.0), gamma)
    return np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)


def generate_lighting_perturbations(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Create small deterministic variations around the accepted capture."""
    return {
        "exposure_darker": _exposure_variant(image_rgb, -EXPOSURE_EV),
        "exposure_brighter": _exposure_variant(image_rgb, EXPOSURE_EV),
        "white_balance_warm": _white_balance_variant(image_rgb, warm=True),
        "white_balance_cool": _white_balance_variant(image_rgb, warm=False),
        "gamma_shadow_deeper": _gamma_variant(image_rgb, 1.0 + GAMMA_DELTA),
        "gamma_shadow_lighter": _gamma_variant(image_rgb, 1.0 - GAMMA_DELTA),
    }


def _matching_lab(result) -> tuple | None:
    if bool(getattr(result, "foundation_target_active", False)):
        target = getattr(result, "foundation_target_lab", None)
        if target is not None:
            return tuple(float(value) for value in target)
    lab = getattr(result, "lab", None)
    if lab is None:
        return None
    return tuple(float(value) for value in lab)


def analyze_lighting_sensitivity(
    image_rgb: np.ndarray,
    masks: dict,
    baseline_result,
) -> LightingSensitivityResult:
    """Re-extract skin tone after conservative capture-color perturbations."""
    baseline_lab = _matching_lab(baseline_result)
    variants = generate_lighting_perturbations(image_rgb)
    if baseline_lab is None or not bool(getattr(baseline_result, "success", False)):
        return LightingSensitivityResult(
            attempted_variants=len(variants),
            warnings=["Lighting sensitivity could not be measured without a usable baseline extraction."],
        )

    variant_labs = []
    for variant_rgb in variants.values():
        result = extract_skin_tone(variant_rgb, masks)
        variant_lab = _matching_lab(result) if result.success else None
        if variant_lab is not None and np.all(np.isfinite(variant_lab)):
            variant_labs.append(variant_lab)

    if not variant_labs:
        return LightingSensitivityResult(
            attempted_variants=len(variants),
            warnings=["Lighting sensitivity variants did not produce usable skin extractions."],
        )

    baseline_grid = np.repeat(
        np.asarray(baseline_lab, dtype=np.float64)[None, :],
        len(variant_labs),
        axis=0,
    )
    variant_array = np.asarray(variant_labs, dtype=np.float64)
    shifts = deltaE_ciede2000(baseline_grid, variant_array)
    p90 = float(np.percentile(shifts, 90))
    maximum = float(np.max(shifts))
    score = float(np.clip(100.0 * (1.0 - p90 / 10.0), 0.0, 100.0))
    stable = p90 <= STABLE_DELTA_E_P90
    warnings = []
    if p90 > CAUTION_DELTA_E_P90:
        warnings.append(
            "Shade evidence is highly sensitive to small exposure or white-balance changes; recapture is recommended."
        )
    elif not stable:
        warnings.append(
            "Shade evidence changes moderately under small exposure or white-balance variations."
        )

    return LightingSensitivityResult(
        variant_labs=variant_labs,
        score=score,
        delta_e_p90=p90,
        max_delta_e=maximum,
        attempted_variants=len(variants),
        successful_variants=len(variant_labs),
        stable=stable,
        warnings=warnings,
    )
