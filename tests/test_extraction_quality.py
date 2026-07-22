from types import SimpleNamespace

from src.confidence import QualityReport, compute_confidence
from src.extraction_quality import build_extraction_quality_report
from src.shade_matcher import ShadeMatch


def _region(score=90, valid_ratio=0.9, shadow_highlight=0.05, reliable=True, excluded=False):
    return SimpleNamespace(
        quality_score=score,
        valid_ratio=valid_ratio,
        shadow_highlight_ratio=shadow_highlight,
        reliable=reliable,
        excluded=excluded,
    )


def _skin(region_score=90, patch_used=True, stability=92, consistency=0.9):
    return SimpleNamespace(
        region_results={
            "left_cheek": _region(region_score),
            "right_cheek": _region(region_score),
            "jawline": _region(region_score - 5),
        },
        patch_voting_diagnostics={
            "used": patch_used,
            "stable_patches_used": 8 if patch_used else 0,
            "stable_patches_available": 9 if patch_used else 0,
            "outlier_patches_rejected": 0,
            "highlight_patches_rejected": 1,
            "shadow_patches_rejected": 0,
            "midtone_patches_used": 7 if patch_used else 0,
        },
        stability_diagnostics={
            "stability_score": stability,
            "warnings": ["Region stability was fair; jawline had stronger influence, so confidence was reduced."]
            if stability < 70
            else [],
        },
        region_consistency=consistency,
        avg_valid_pixel_ratio=0.9,
    )


def _image(score=90):
    return SimpleNamespace(overall_score=score, warnings=[] if score >= 60 else ["Image may be slightly blurry."])


def _lighting(score=0.9):
    return SimpleNamespace(score=score, warnings=[] if score >= 0.7 else ["Lighting appears uneven across the face/image."])


def _selection(chroma=0.98, lab_difference=3.0, source="original"):
    return SimpleNamespace(
        chroma_preservation_score=chroma,
        lab_difference=lab_difference,
        selected_source=source,
    )


def test_extraction_quality_overall_score_stays_in_range():
    report = build_extraction_quality_report(_skin(), _image(), _lighting(), _selection())

    assert 0 <= report["overall_score"] <= 100
    assert report["label"] in {"excellent", "good", "fair", "poor"}
    assert set(report["subscores"]) == {
        "image_capture",
        "region_reliability",
        "patch_stability",
        "lighting_safety",
        "color_consistency",
        "region_stability",
    }


def test_poor_image_quality_lowers_overall_score():
    good = build_extraction_quality_report(_skin(), _image(95), _lighting(), _selection())
    poor = build_extraction_quality_report(_skin(), _image(30), _lighting(), _selection())

    assert poor["overall_score"] < good["overall_score"]
    assert any("blurry" in warning.lower() for warning in poor["warnings"])


def test_poor_region_quality_lowers_overall_score():
    good = build_extraction_quality_report(_skin(region_score=92), _image(), _lighting(), _selection())
    poor = build_extraction_quality_report(_skin(region_score=35), _image(), _lighting(), _selection())

    assert poor["overall_score"] < good["overall_score"]
    assert poor["subscores"]["region_reliability"] < good["subscores"]["region_reliability"]


def test_strong_patch_stability_improves_score():
    with_patches = build_extraction_quality_report(_skin(patch_used=True), _image(), _lighting(), _selection())
    fallback = build_extraction_quality_report(_skin(patch_used=False), _image(), _lighting(), _selection())

    assert with_patches["subscores"]["patch_stability"] > fallback["subscores"]["patch_stability"]
    assert with_patches["overall_score"] > fallback["overall_score"]


def test_region_instability_lowers_score_and_warns():
    stable = build_extraction_quality_report(_skin(stability=95), _image(), _lighting(), _selection())
    unstable = build_extraction_quality_report(_skin(stability=45), _image(), _lighting(), _selection())

    assert unstable["overall_score"] < stable["overall_score"]
    assert unstable["subscores"]["region_stability"] < stable["subscores"]["region_stability"]
    assert any("region stability" in warning.lower() for warning in unstable["warnings"])


def test_extraction_quality_and_shade_confidence_remain_separate_values():
    extraction_report = build_extraction_quality_report(_skin(), _image(), _lighting(), _selection())
    match = ShadeMatch(
        shade_id="S1",
        brand="Test",
        shade_name="Shade",
        hex="#AABBCC",
        rgb=(170, 187, 204),
        lab=(50.0, 0.0, 0.0),
        delta_e=7.0,
    )
    confidence_report = QualityReport(
        region_consistency=0.8,
        valid_pixel_ratio=0.8,
        face_quality=1.0,
        top_match_separation=0.6,
    )
    confidence = compute_confidence([match], confidence_report)[0].confidence

    assert extraction_report["overall_score"] > 1.0
    assert 0.0 <= confidence <= 1.0
    assert "Skin Extraction Quality" in extraction_report["reasons"][1]
