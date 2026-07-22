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
    def __init__(self, region_consistency, avg_valid_pixel_ratio, cheek_area_balance=1.0):
        self.region_consistency = region_consistency
        self.avg_valid_pixel_ratio = avg_valid_pixel_ratio
        self.cheek_area_balance = cheek_area_balance


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


def test_close_match_tie_warning_and_explanation_wording():
    matches = [_match(5.0), _match(5.4), _match(5.9)]
    qr = build_quality_report(FakeSkinResult(0.9, 0.9), FakeFaceResult([]), matches)
    assert qr.close_match_tie
    assert any("Close match tie" in w for w in qr.warnings)
    text = build_explanation(matches[0], FakeSkinResult(0.9, 0.9), qr, rank=1, matches=matches)
    assert "Close match tie" in text
    assert "equivalent candidates" in text


def test_explanation_mentions_undertone_and_delta_e():
    match = _match(3.0, undertone="warm")
    qr = QualityReport(region_consistency=0.9, valid_pixel_ratio=0.9, face_quality=1.0, top_match_separation=0.9)
    text = build_explanation(match, FakeSkinResult(0.9, 0.9), qr, rank=1, matches=[match])
    assert "warm" in text
    assert "3.0" in text


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
