from types import SimpleNamespace

from src.confidence import QualityReport, compute_confidence
from src.recommendation_readiness import (
    RecommendationReadiness,
    build_recommendation_readiness,
)
from src.shade_matcher import ShadeMatch


def _skin(
    radius: float,
    stability: float = 80.0,
    sensitivity_score: float = 100.0,
    sensitivity_radius: float = 0.0,
    eyewear: bool = False,
):
    return SimpleNamespace(
        success=True,
        uncertainty_diagnostics={
            "delta_e_radius_p90": radius,
            "stability_score": stability,
        },
        lighting_sensitivity_diagnostics={
            "score": sensitivity_score,
            "delta_e_p90": sensitivity_radius,
        },
        capture_region_diagnostics={
            "eyewear_reflection_detected": eyewear,
            "eyewear_exclusion_applied": eyewear,
        },
    )


def _lighting(score: float, low_signal: bool = False):
    return SimpleNamespace(score=score, low_signal=low_signal)


def test_readiness_thresholds_are_deterministic():
    ready = build_recommendation_readiness(
        _skin(4.0, 90.0),
        {"overall_score": 80.0},
        _lighting(0.85),
    )
    caution = build_recommendation_readiness(
        _skin(8.0, 65.0),
        {"overall_score": 60.0},
        _lighting(0.60),
    )
    provisional = build_recommendation_readiness(
        _skin(12.0, 40.0),
        {"overall_score": 45.0},
        _lighting(0.50),
    )

    assert ready.state == "ready"
    assert caution.state == "caution"
    assert provisional.state == "provisional"
    assert 0.78 <= ready.confidence_cap <= 0.93
    assert 0.55 <= caution.confidence_cap <= 0.75
    assert provisional.confidence_cap == 0.55
    assert "Top 3" in provisional.warnings[0]


def test_lighting_sensitivity_can_downgrade_readiness():
    ready = build_recommendation_readiness(
        _skin(4.0, 90.0, sensitivity_score=90.0, sensitivity_radius=2.0),
        {"overall_score": 80.0},
        _lighting(0.85),
    )
    caution = build_recommendation_readiness(
        _skin(4.0, 90.0, sensitivity_score=55.0, sensitivity_radius=5.0),
        {"overall_score": 80.0},
        _lighting(0.85),
    )
    provisional = build_recommendation_readiness(
        _skin(4.0, 90.0, sensitivity_score=25.0, sensitivity_radius=8.0),
        {"overall_score": 80.0},
        _lighting(0.85),
    )

    assert ready.state == "ready"
    assert caution.state == "caution"
    assert provisional.state == "provisional"
    assert "Lighting sensitivity" in provisional.reasons[-1]


def test_low_signal_forces_provisional_even_when_other_evidence_is_strong():
    readiness = build_recommendation_readiness(
        _skin(2.0, 95.0, sensitivity_score=95.0, sensitivity_radius=1.0),
        {"overall_score": 90.0},
        _lighting(0.85, low_signal=True),
    )

    assert readiness.state == "provisional"
    assert readiness.confidence_cap == 0.55
    assert any("Low-signal" in reason for reason in readiness.reasons)


def test_small_sensitivity_change_does_not_create_three_delta_e_state_cliff():
    below = build_recommendation_readiness(
        _skin(4.0, 90.0, sensitivity_score=72.0, sensitivity_radius=2.9),
        {"overall_score": 82.0},
        _lighting(0.75),
    )
    above = build_recommendation_readiness(
        _skin(4.0, 90.0, sensitivity_score=70.0, sensitivity_radius=3.1),
        {"overall_score": 82.0},
        _lighting(0.75),
    )

    assert below.state == above.state
    assert abs(below.confidence_cap - above.confidence_cap) < 0.02


def test_detected_eyewear_reduces_readiness_without_hiding_results():
    clean = build_recommendation_readiness(
        _skin(3.0, 92.0, sensitivity_score=90.0, sensitivity_radius=2.0),
        {"overall_score": 85.0},
        _lighting(0.85),
    )
    eyewear = build_recommendation_readiness(
        _skin(
            3.0,
            92.0,
            sensitivity_score=90.0,
            sensitivity_radius=2.0,
            eyewear=True,
        ),
        {"overall_score": 85.0},
        _lighting(0.85),
    )

    assert eyewear.score == clean.score - 4.0
    assert eyewear.confidence_cap <= clean.confidence_cap
    assert any("Glasses/reflection" in reason for reason in eyewear.reasons)


def test_provisional_state_caps_match_confidence_but_keeps_match():
    match = ShadeMatch(
        shade_id="S1",
        brand="Brand",
        shade_name="Shade",
        hex="#806050",
        rgb=(128, 96, 80),
        lab=(45.0, 8.0, 14.0),
        delta_e=0.1,
        recommendation_stability=1.0,
        top3_stability=1.0,
        catalog_quality_score=1.0,
    )
    quality = QualityReport(
        region_consistency=1.0,
        valid_pixel_ratio=1.0,
        face_quality=1.0,
        top_match_separation=1.0,
        lighting_quality=1.0,
        extraction_uncertainty=1.0,
    )
    readiness = RecommendationReadiness(
        state="provisional",
        score=40.0,
        confidence_cap=0.55,
        summary="provisional",
    )

    matches = compute_confidence([match], quality, readiness=readiness)

    assert len(matches) == 1
    assert 0.0 < matches[0].confidence < 0.55
    assert matches[0].confidence_breakdown["readiness_cap"] == 0.55


def test_unstable_shade_family_does_not_change_capture_readiness_state():
    unstable = ShadeMatch(
        shade_id="S1",
        brand="Brand",
        shade_name="Shade",
        hex="#806050",
        rgb=(128, 96, 80),
        lab=(45.0, 8.0, 14.0),
        delta_e=0.1,
        recommendation_stability=0.2,
        top3_stability=0.3,
        lighting_recommendation_stability=0.1,
        lighting_top3_stability=0.2,
        recommendation_family_stability=0.3,
        top3_family_stability=0.4,
        lighting_family_stability=0.2,
        lighting_top3_family_stability=0.25,
    )

    readiness = build_recommendation_readiness(
        _skin(3.0, 92.0, sensitivity_score=90.0, sensitivity_radius=2.0),
        {"overall_score": 85.0},
        _lighting(0.90),
        matches=[unstable],
    )

    assert readiness.state == "ready"
    assert any(
        "reported separately from capture readiness" in reason
        for reason in readiness.reasons
    )


def test_exact_product_instability_is_reported_separately_from_ready_family():
    family_stable = ShadeMatch(
        shade_id="S1",
        brand="Brand",
        shade_name="Shade",
        hex="#806050",
        rgb=(128, 96, 80),
        lab=(45.0, 8.0, 14.0),
        delta_e=0.1,
        recommendation_stability=0.15,
        top3_stability=0.30,
        lighting_recommendation_stability=0.10,
        lighting_top3_stability=0.25,
        recommendation_family_stability=0.95,
        top3_family_stability=1.0,
        lighting_family_stability=0.90,
        lighting_top3_family_stability=0.95,
    )

    readiness = build_recommendation_readiness(
        _skin(3.0, 92.0, sensitivity_score=90.0, sensitivity_radius=2.0),
        {"overall_score": 85.0},
        _lighting(0.90),
        matches=[family_stable],
    )

    assert readiness.state == "ready"
    assert readiness.shade_family_stability_score > 90.0
    assert readiness.exact_product_stability_score < 30.0
