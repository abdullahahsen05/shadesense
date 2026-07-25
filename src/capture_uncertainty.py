"""Systematic capture uncertainty beyond within-photo patch bootstrap."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CaptureUncertainty:
    local_patch_radius: float
    lighting_sensitivity_radius: float
    systematic_radius: float
    total_radius: float
    score: float
    components: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_diagnostics(self) -> dict:
        return {
            "local_patch_radius": self.local_patch_radius,
            "lighting_sensitivity_radius": self.lighting_sensitivity_radius,
            "systematic_radius": self.systematic_radius,
            "total_delta_e_radius_p90": self.total_radius,
            "score": self.score,
            "components": dict(self.components),
            "warnings": list(self.warnings),
            "method": (
                "quadrature combination of patch bootstrap, lighting "
                "sensitivity, face-lighting, pose, and occlusion risks"
            ),
        }


def analyze_capture_uncertainty(
    skin_result,
    lighting_quality=None,
    image_quality=None,
) -> CaptureUncertainty:
    """Estimate error shared by all patches from one camera capture.

    Patch bootstrap cannot see a global exposure or white-balance error because
    every patch inherits it. This model keeps that local radius but adds
    independent capture-level risks. Absolute skin lightness is intentionally
    not used, avoiding an underexposure rule that would penalize deep skin.
    """
    bootstrap = getattr(skin_result, "uncertainty_diagnostics", {}) or {}
    sensitivity = (
        getattr(skin_result, "lighting_sensitivity_diagnostics", {}) or {}
    )
    capture_regions = (
        getattr(skin_result, "capture_region_diagnostics", {}) or {}
    )
    local_radius = float(bootstrap.get("delta_e_radius_p90", 12.0))
    sensitivity_radius = float(sensitivity.get("delta_e_p90", 0.0))

    exposure_score = float(getattr(lighting_quality, "exposure_score", 1.0))
    uniformity_score = float(getattr(lighting_quality, "uniformity_score", 1.0))
    contrast_score = float(getattr(lighting_quality, "contrast_score", 1.0))
    color_score = float(getattr(lighting_quality, "color_score", 1.0))
    low_signal = bool(getattr(lighting_quality, "low_signal", False))
    pose_asymmetry = float(getattr(image_quality, "pose_asymmetry", 0.0) or 0.0)

    components = {
        "exposure": 4.0 * (1.0 - np.clip(exposure_score, 0.0, 1.0)),
        "uniformity": 7.0 * (1.0 - np.clip(uniformity_score, 0.0, 1.0)),
        "contrast": 2.5 * (1.0 - np.clip(contrast_score, 0.0, 1.0)),
        "color_cast": 3.0 * (1.0 - np.clip(color_score, 0.0, 1.0)),
        "pose": 4.0 * np.clip(max(pose_asymmetry - 0.10, 0.0) / 0.35, 0.0, 1.0),
        "occlusion": 2.5
        if capture_regions.get("eyewear_reflection_detected")
        else 0.0,
        "low_signal": 4.0 if low_signal else 0.0,
    }
    systematic_radius = float(
        np.sqrt(sum(float(value) ** 2 for value in components.values()))
    )
    total_radius = float(
        np.sqrt(
            local_radius**2
            + sensitivity_radius**2
            + systematic_radius**2
        )
    )
    score = float(np.clip(100.0 * (1.0 - total_radius / 14.0), 0.0, 100.0))

    warnings = []
    if systematic_radius > 6.0:
        warnings.append(
            "Capture-level exposure or lighting uncertainty is high even if "
            "the retained patches agree with one another."
        )
    elif systematic_radius > 3.5:
        warnings.append(
            "Capture-level lighting uncertainty is moderate and may move the "
            "recommended shade family."
        )
    if low_signal:
        warnings.append(
            "Low-signal facial regions make skin depth ambiguous; recapture "
            "in brighter, even light."
        )

    return CaptureUncertainty(
        local_patch_radius=local_radius,
        lighting_sensitivity_radius=sensitivity_radius,
        systematic_radius=systematic_radius,
        total_radius=total_radius,
        score=score,
        components={key: float(value) for key, value in components.items()},
        warnings=warnings,
    )
