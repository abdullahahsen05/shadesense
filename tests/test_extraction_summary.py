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
        },
        quality_score=0.82,
        region_consistency=0.85,
        avg_valid_pixel_ratio=0.9,
        included_region_names=["left_cheek"],
        excluded_region_names=["forehead"],
        depth_estimate="tan",
    )
    lighting = SimpleNamespace(score=0.91)
    selection = SimpleNamespace(
        selected_source="original",
        reason="Original image color was preserved because correction did not improve extraction reliability.",
    )

    summary = build_skin_extraction_summary(skin, lighting, selection)

    assert "Extraction reliability: Good" in summary
    assert "Trusted regions used: Left Cheek" in summary
    assert "Excluded regions: Forehead" in summary
    assert "Final extraction source: original" in summary
    assert "Final depth estimate: tan" in summary
