import numpy as np

from src.confidence import (
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    QualityReport,
    build_quality_report,
    compute_confidence,
)
from src.explanation import build_explanation
from src.shade_matcher import ShadeMatch


class FakeFaceResult:
    def __init__(self, warnings):
        self.warnings = warnings


class FakeSkinResult:
    def __init__(
        self,
        region_consistency,
        avg_valid_pixel_ratio,
        cheek_area_balance=1.0,
        usable_region_count=4,
        stability_score=100.0,
    ):
        self.region_consistency = region_consistency
        self.avg_valid_pixel_ratio = avg_valid_pixel_ratio
        self.cheek_area_balance = cheek_area_balance
        self.usable_region_count = usable_region_count
        self.stability_diagnostics = {
            "stability_score": stability_score,
            "warnings": ["Region stability was fair; jawline had stronger influence, so confidence was reduced."]
            if stability_score < 70
            else [],
        }
        self.region_results = {}


class FakeLighting:
    def __init__(self, score, warnings=None):
        self.score = score
        self.warnings = warnings or []


def _match(delta_e, undertone="neutral"):
    return ShadeMatch(
        shade_id="S1",
        brand="Test",
        shade_name="Shade",
        hex="#AABBCC",
        rgb=(170, 187, 204),
        lab=(50.0, 0.0, 0.0),
        delta_e=delta_e,
        undertone=undertone,
    )


def test_confidence_within_bounds():
    matches = [_match(2.0), _match(8.0), _match(15.0)]
    qr = QualityReport(region_consistency=0.8, valid_pixel_ratio=0.8, face_quality=1.0, top_match_separation=0.8)
    result = compute_confidence(matches, qr)
    for m in result:
        assert CONFIDENCE_FLOOR <= m.confidence <= CONFIDENCE_CEILING


def test_never_reaches_near_certain_confidence():
    # Even a perfect Delta E=0 match with perfect quality should not claim ~99%+.
    matches = [_match(0.0)]
    qr = QualityReport(region_consistency=1.0, valid_pixel_ratio=1.0, face_quality=1.0, top_match_separation=1.0)
    result = compute_confidence(matches, qr)
    assert result[0].confidence <= CONFIDENCE_CEILING
    assert result[0].confidence < 0.99


def test_confidence_decreases_with_larger_delta_e():
    matches = [_match(1.0), _match(20.0)]
    qr = QualityReport(region_consistency=0.9, valid_pixel_ratio=0.9, face_quality=1.0, top_match_separation=0.9)
    result = compute_confidence(matches, qr)
    assert result[0].confidence > result[1].confidence


def test_confidence_uses_distribution_aware_distance_when_available():
    stable = _match(3.0)
    stable.distribution_delta_e = 3.0
    uncertain = _match(3.0)
    uncertain.distribution_delta_e = 9.0
    qr = QualityReport(
        region_consistency=0.9,
        valid_pixel_ratio=0.9,
        face_quality=1.0,
        top_match_separation=0.9,
    )

    stable_result = compute_confidence([stable], qr)[0]
    uncertain_result = compute_confidence([uncertain], qr)[0]

    assert uncertain_result.confidence < stable_result.confidence
    assert uncertain_result.confidence_breakdown["distribution_delta_e"] == 9.0


def test_poor_region_consistency_lowers_confidence():
    good_qr = QualityReport(region_consistency=1.0, valid_pixel_ratio=0.8, face_quality=1.0, top_match_separation=0.8)
    bad_qr = QualityReport(region_consistency=0.1, valid_pixel_ratio=0.8, face_quality=1.0, top_match_separation=0.8)
    good = compute_confidence([_match(5.0)], good_qr)[0]
    bad = compute_confidence([_match(5.0)], bad_qr)[0]
    assert bad.confidence < good.confidence


def test_close_top_matches_lower_confidence_via_separation():
    matches_close = [_match(5.0), _match(5.1)]
    matches_far = [_match(5.0), _match(15.0)]
    qr_close = build_quality_report(
        FakeSkinResult(0.9, 0.9), FakeFaceResult([]), matches_close
    )
    qr_far = build_quality_report(
        FakeSkinResult(0.9, 0.9), FakeFaceResult([]), matches_far
    )
    conf_close = compute_confidence([_match(5.0)], qr_close)[0].confidence
    conf_far = compute_confidence([_match(5.0)], qr_far)[0].confidence
    assert conf_close < conf_far


def test_face_quality_penalized_for_multi_face_and_small_face():
    qr_clean = build_quality_report(FakeSkinResult(1.0, 1.0), FakeFaceResult([]), [_match(1.0), _match(10.0)])
    qr_multi = build_quality_report(
        FakeSkinResult(1.0, 1.0), FakeFaceResult(["2 faces detected. Using the largest..."]), [_match(1.0), _match(10.0)]
    )
    assert qr_multi.face_quality < qr_clean.face_quality


def test_asymmetric_cheek_area_warns_and_slightly_penalizes_confidence():
    matches = [_match(5.0), _match(9.0)]
    balanced_qr = build_quality_report(
        FakeSkinResult(0.9, 0.9, cheek_area_balance=1.0), FakeFaceResult([]), matches
    )
    imbalanced_qr = build_quality_report(
        FakeSkinResult(0.9, 0.9, cheek_area_balance=0.3), FakeFaceResult([]), matches
    )
    balanced = compute_confidence([_match(5.0)], balanced_qr)[0]
    imbalanced = compute_confidence([_match(5.0)], imbalanced_qr)[0]
    assert imbalanced.confidence < balanced.confidence
    assert any("cheek" in w.lower() for w in imbalanced_qr.warnings)


def test_region_stability_lowers_confidence_slightly():
    matches = [_match(5.0), _match(9.0)]
    stable_qr = build_quality_report(
        FakeSkinResult(0.9, 0.9, stability_score=95.0), FakeFaceResult([]), matches
    )
    unstable_qr = build_quality_report(
        FakeSkinResult(0.9, 0.9, stability_score=45.0), FakeFaceResult([]), matches
    )
    stable = compute_confidence([_match(5.0)], stable_qr)[0]
    unstable = compute_confidence([_match(5.0)], unstable_qr)[0]

    assert unstable.confidence < stable.confidence
    assert unstable.confidence_breakdown["region_stability_penalty"] > stable.confidence_breakdown["region_stability_penalty"]
    assert any("region stability" in warning.lower() for warning in unstable_qr.warnings)


def test_highlight_influence_lowers_confidence_slightly():
    matches = [_match(5.0), _match(9.0)]
    clean_skin = FakeSkinResult(0.9, 0.9)
    highlighted_skin = FakeSkinResult(0.9, 0.9)
    highlighted_skin.region_results = {
        "forehead": type("Region", (), {"highlight_patches_rejected": 3, "specular_highlight_detected": True})(),
        "left_cheek": type("Region", (), {"highlight_patches_rejected": 1, "specular_highlight_detected": False})(),
    }
    clean_qr = build_quality_report(clean_skin, FakeFaceResult([]), matches)
    highlighted_qr = build_quality_report(highlighted_skin, FakeFaceResult([]), matches)
    clean = compute_confidence([_match(5.0)], clean_qr)[0]
    highlighted = compute_confidence([_match(5.0)], highlighted_qr)[0]

    assert highlighted.confidence < clean.confidence
    assert highlighted.confidence_breakdown["highlight_safety_penalty"] > 0
    assert any("highlight influence" in warning.lower() for warning in highlighted_qr.warnings)


def test_close_match_tie_warning_and_explanation_wording():
    matches = [_match(5.0), _match(5.4), _match(5.9)]
    qr = build_quality_report(FakeSkinResult(0.9, 0.9), FakeFaceResult([]), matches)
    assert qr.close_match_tie
    assert any("Close match tie" in w for w in qr.warnings)
    text = build_explanation(matches[0], FakeSkinResult(0.9, 0.9), qr, rank=1, matches=matches)
    assert "Close match tie: these shades are nearly identical" in text
    assert "equivalent candidates" in text


def test_lighting_quality_lowers_confidence_slightly_and_breakdown_fields_exist():
    matches = [_match(5.0), _match(9.0)]
    good_qr = build_quality_report(
        FakeSkinResult(0.9, 0.9), FakeFaceResult([]), matches, FakeLighting(1.0)
    )
    poor_qr = build_quality_report(
        FakeSkinResult(0.9, 0.9), FakeFaceResult([]), matches, FakeLighting(0.4, ["dim"])
    )
    good = compute_confidence([_match(5.0)], good_qr)[0]
    poor = compute_confidence([_match(5.0)], poor_qr)[0]
    assert poor.confidence < good.confidence
    assert "dim" in poor_qr.warnings
    expected = {
        "color_distance_contribution",
        "region_consistency_contribution",
        "valid_pixel_patch_contribution",
        "lighting_quality_contribution",
        "top_shade_separation_contribution",
    }
    assert expected <= set(poor.confidence_breakdown)


def test_lighting_sensitivity_lowers_confidence_and_adds_warning():
    matches = [_match(5.0), _match(9.0)]
    stable_skin = FakeSkinResult(0.9, 0.9)
    stable_skin.lighting_sensitivity_diagnostics = {
        "score": 95.0,
        "delta_e_p90": 1.0,
    }
    sensitive_skin = FakeSkinResult(0.9, 0.9)
    sensitive_skin.lighting_sensitivity_diagnostics = {
        "score": 30.0,
        "delta_e_p90": 7.0,
    }

    stable_report = build_quality_report(
        stable_skin, FakeFaceResult([]), matches, FakeLighting(0.9)
    )
    sensitive_report = build_quality_report(
        sensitive_skin, FakeFaceResult([]), matches, FakeLighting(0.9)
    )
    stable = compute_confidence([_match(5.0)], stable_report)[0]
    sensitive = compute_confidence([_match(5.0)], sensitive_report)[0]

    assert sensitive.confidence < stable.confidence
    assert (
        sensitive.confidence_breakdown["lighting_sensitivity_penalty"]
        > stable.confidence_breakdown["lighting_sensitivity_penalty"]
    )
    assert any(
        "simulated exposure" in warning.lower()
        for warning in sensitive_report.warnings
    )


def test_explanation_mentions_undertone_and_delta_e():
    match = _match(3.0, undertone="warm")
    qr = QualityReport(region_consistency=0.9, valid_pixel_ratio=0.9, face_quality=1.0, top_match_separation=0.9)
    text = build_explanation(match, FakeSkinResult(0.9, 0.9), qr, rank=1, matches=[match])
    assert "warm" in text
    assert "3.0" in text


def test_explanation_mentions_grouped_product_variants():
    match = _match(3.0)
    match.product_variants = [{"product": "Foundation Stick"}]
    qr = QualityReport(region_consistency=0.9, valid_pixel_ratio=0.9, face_quality=1.0, top_match_separation=0.9)
    text = build_explanation(match, FakeSkinResult(0.9, 0.9), qr, rank=1, matches=[match])
    assert "multiple product formats" in text
    assert "closest matching variant is shown" in text


def test_explanation_flags_region_disagreement_and_low_pixels():
    match = _match(5.0)
    qr = QualityReport(region_consistency=0.2, valid_pixel_ratio=0.2, face_quality=1.0, top_match_separation=0.9)
    text = build_explanation(match, FakeSkinResult(0.2, 0.2), qr, rank=1, matches=[match])
    assert "did not agree" in text
    assert "Fewer valid skin pixels" in text


def test_explanation_is_deterministic():
    match = _match(4.0, undertone="cool")
    qr = QualityReport(region_consistency=0.7, valid_pixel_ratio=0.7, face_quality=0.9, top_match_separation=0.6)
    text1 = build_explanation(match, FakeSkinResult(0.7, 0.7), qr, rank=1, matches=[match])
    text2 = build_explanation(match, FakeSkinResult(0.7, 0.7), qr, rank=1, matches=[match])
    assert text1 == text2
