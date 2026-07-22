from types import SimpleNamespace

from src.extraction_summary import build_skin_extraction_summary
from src.skin_extraction import RegionSkinResult, SkinToneResult


def test_skin_extraction_summary_includes_regions_source_and_reliability():
    skin = SkinToneResult(
        rgb=(120, 80, 60),
        lab=(45.0, 10.0, 15.0),
        region_results={
            "left_cheek": RegionSkinResult(
                "left_cheek", 200, 180, 0.9, (120, 80, 60), (45.0, 10.0, 15.0), True
            ),
            "forehead": RegionSkinResult(
                "forehead", 200, 180, 0.9, (180, 130, 100), (60.0, 12.0, 18.0), True, excluded=True
            ),
            "jawline": RegionSkinResult(
                "jawline",
                200,
                150,
                0.75,
                (115, 78, 58),
                (43.0, 9.0, 14.0),
                True,
                weight_multiplier=0.7,
                downweight_reason="Possible chin/neck shadow, contour, occlusion, or uneven lighting.",
            ),
        },
        quality_score=0.82,
        region_consistency=0.85,
        avg_valid_pixel_ratio=0.9,
        included_region_names=["left_cheek", "jawline"],
        excluded_region_names=["forehead"],
        patch_voting_diagnostics={"used": True},
        stability_diagnostics={
            "stability_label": "good",
            "summary": "Region stability was good; removing any one trusted region did not significantly change the final tone.",
        },
        depth_estimate="tan",
    )
    skin.region_results["left_cheek"].role = "trusted"
    skin.region_results["jawline"].role = "reduced"
    skin.region_results["forehead"].role = "excluded"
    lighting = SimpleNamespace(score=0.91)
    selection = SimpleNamespace(
        selected_source="original",
        reason="Original image color was preserved because correction did not improve extraction reliability.",
    )

    summary = build_skin_extraction_summary(skin, lighting, selection)

    assert "Extraction reliability: Good" in summary
    assert "Trusted regions used: Left Cheek, Jawline" in summary
    assert "Highest-trust regions: Left Cheek" in summary
    assert "Reduced-weight regions: Jawline" in summary
    assert "Final skin tone was aggregated from stable diffuse patches across trusted regions." in summary
    assert "Region stability was good" in summary
    assert "Excluded regions: Forehead" in summary
    assert "Final extraction source: original" in summary
    assert "Final depth estimate: tan" in summary
