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
    )


def _lighting(score: float):
    return SimpleNamespace(score=score)


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

    assert (ready.state, ready.confidence_cap) == ("ready", 0.93)
    assert (caution.state, caution.confidence_cap) == ("caution", 0.75)
    assert (provisional.state, provisional.confidence_cap) == ("provisional", 0.55)
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
    assert matches[0].confidence == 0.55
    assert matches[0].confidence_breakdown["readiness_cap"] == 0.55


def test_unstable_shade_family_prevents_ready_state_after_matching():
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

    assert readiness.state == "provisional"
    assert any(
        "Shade-family ranking stability" in reason
        for reason in readiness.reasons
    )
