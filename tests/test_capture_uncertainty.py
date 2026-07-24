from types import SimpleNamespace

from src.capture_uncertainty import analyze_capture_uncertainty


def _skin(local=1.0, sensitivity=1.0, eyewear=False):
    return SimpleNamespace(
        uncertainty_diagnostics={"delta_e_radius_p90": local},
        lighting_sensitivity_diagnostics={"delta_e_p90": sensitivity},
        capture_region_diagnostics={
            "eyewear_reflection_detected": eyewear,
        },
    )


def _lighting(
    exposure=1.0,
    uniformity=1.0,
    contrast=1.0,
    color=1.0,
    low_signal=False,
):
    return SimpleNamespace(
        exposure_score=exposure,
        uniformity_score=uniformity,
        contrast_score=contrast,
        color_score=color,
        low_signal=low_signal,
    )


def test_systematic_uncertainty_stays_separate_from_patch_bootstrap():
    stable = analyze_capture_uncertainty(
        _skin(local=1.0, sensitivity=1.0),
        _lighting(),
        SimpleNamespace(pose_asymmetry=0.02),
    )
    split_light = analyze_capture_uncertainty(
        _skin(local=1.0, sensitivity=1.0),
        _lighting(uniformity=0.15, low_signal=True),
        SimpleNamespace(pose_asymmetry=0.02),
    )

    assert stable.local_patch_radius == split_light.local_patch_radius
    assert split_light.systematic_radius > stable.systematic_radius
    assert split_light.total_radius > stable.total_radius
    assert split_light.score < stable.score


def test_occlusion_and_pose_increase_capture_uncertainty():
    clean = analyze_capture_uncertainty(
        _skin(),
        _lighting(),
        SimpleNamespace(pose_asymmetry=0.05),
    )
    obstructed = analyze_capture_uncertainty(
        _skin(eyewear=True),
        _lighting(),
        SimpleNamespace(pose_asymmetry=0.35),
    )

    assert obstructed.components["occlusion"] > 0
    assert obstructed.components["pose"] > 0
    assert obstructed.total_radius > clean.total_radius
