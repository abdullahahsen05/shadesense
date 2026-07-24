import numpy as np
import pandas as pd
import pytest
from skimage.color import deltaE_ciede2000

from src.config import SHADE_CATALOG_PATH
from src.shade_catalog import CatalogValidationError, load_shade_catalog
from src.shade_matcher import _too_light_penalty, estimate_depth_from_lab_l, match_shades


def test_deltaE_ciede2000_identical_colors_is_zero():
    """Regression test: skimage's deltaE_ciede2000 does not correctly
    broadcast a (3,) color against an (N,3) array, so callers must tile
    explicitly. This guards against reintroducing that bug."""
    lab = np.array([60.0, 5.0, 15.0])
    tiled = np.tile(lab, (5, 1))
    distances = deltaE_ciede2000(tiled, tiled)
    assert np.allclose(distances, 0.0)


def test_mock_catalog_loads_and_has_min_shades():
    df = load_shade_catalog(str(SHADE_CATALOG_PATH))
    assert len(df) >= 15
    for col in ["lab_l", "lab_a", "lab_b", "hex", "r", "g", "b"]:
        assert col in df.columns


def test_missing_required_columns_raises():
    with pytest.raises(CatalogValidationError):
        load_shade_catalog_from_df(pd.DataFrame({"name": ["a"], "color": ["red"]}))


def load_shade_catalog_from_df(df):
    # Helper shim: write df to a temp CSV to exercise load_shade_catalog's
    # CSV-based interface consistently.
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    try:
        return load_shade_catalog(path)
    finally:
        os.remove(path)


def test_invalid_color_rows_are_skipped_not_crashed():
    df = pd.DataFrame(
        {
            "shade_id": ["G1", "B1", "G2"],
            "brand": ["T", "T", "T"],
            "shade_name": ["Good1", "Bad", "Good2"],
            "hex": ["#112233", "notahex", ""],
            "r": [17, None, 50],
            "g": [34, None, 60],
            "b": [51, None, 70],
        }
    )
    result = load_shade_catalog_from_df(df)
    assert len(result) == 2
    assert len(result.attrs["warnings"]) == 1


def test_fewer_than_three_shades_returns_available_without_crashing():
    df = pd.DataFrame(
        {
            "shade_id": ["S1"],
            "brand": ["T"],
            "shade_name": ["One"],
            "hex": ["#AABBCC"],
            "r": [170],
            "g": [187],
            "b": [204],
        }
    )
    catalog = load_shade_catalog_from_df(df)
    matches = match_shades(np.array([50.0, 0.0, 0.0]), catalog, top_k=3)
    assert len(matches) == 1


def test_empty_catalog_returns_empty_matches():
    matches = match_shades(np.array([50.0, 0.0, 0.0]), pd.DataFrame(), top_k=3)
    assert matches == []


def test_match_shades_sorted_ascending_by_delta_e():
    df = load_shade_catalog(str(SHADE_CATALOG_PATH))
    skin_lab = df.iloc[5][["lab_l", "lab_a", "lab_b"]].to_numpy(dtype=float)
    matches = match_shades(skin_lab, df, top_k=3)
    assert len(matches) == 3
    deltas = [m.delta_e for m in matches]
    assert deltas == sorted(deltas)
    assert matches[0].delta_e < 1e-6  # matches its own shade exactly
    assert [m.rank for m in matches] == [1, 2, 3]


def test_depth_estimation_from_lab_l():
    assert estimate_depth_from_lab_l(90) == "fair"
    assert estimate_depth_from_lab_l(70) == "light-medium"
    assert estimate_depth_from_lab_l(50) == "tan"
    assert estimate_depth_from_lab_l(25) == "rich-deep"


def test_depth_tiebreak_affects_only_close_matches():
    df = pd.DataFrame(
        {
            "shade_id": ["P1", "P2", "P3"],
            "brand": ["T", "T", "T"],
            "shade_name": ["Close Wrong Depth", "Close Right Depth", "Clear Winner"],
            "hex": ["#777777", "#777777", "#777777"],
            "r": [119, 119, 119],
            "g": [119, 119, 119],
            "b": [119, 119, 119],
            "lab_l": [50.0, 50.4, 55.0],
            "lab_a": [0.0, 0.0, 0.0],
            "lab_b": [0.0, 0.0, 0.0],
            "depth": ["fair", "tan", "tan"],
        }
    )
    close_matches = match_shades(np.array([50.2, 0.0, 0.0]), df, top_k=2)
    assert close_matches[0].shade_name == "Close Right Depth"
    assert close_matches[0].delta_e <= close_matches[1].delta_e + 0.5

    clear_matches = match_shades(np.array([55.0, 0.0, 0.0]), df, top_k=1)
    assert clear_matches[0].shade_name == "Clear Winner"
    assert clear_matches[0].delta_e < 1e-6


def test_close_too_light_shade_loses_to_similar_deeper_candidate():
    df = pd.DataFrame(
        {
            "shade_id": ["too_light", "deeper"],
            "brand": ["Test", "Test"],
            "shade_name": ["Too Light", "Similar Deeper"],
            "hex": ["#111111", "#222222"],
            "r": [1, 2],
            "g": [1, 2],
            "b": [1, 2],
            "lab_l": [35.0, 31.0],
            "lab_a": [8.0, 13.0],
            "lab_b": [10.0, 13.0],
            "depth": ["deep", "rich-deep"],
        }
    )

    matches = match_shades(np.array([30.0, 8.0, 10.0]), df, top_k=2)

    assert matches[0].shade_name == "Similar Deeper"
    assert matches[1].shade_name == "Too Light"
    assert matches[1].delta_e < matches[0].delta_e
    assert matches[1].depth_penalty > 0


def test_same_brand_same_shade_different_product_appears_once_with_variants():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "A2", "B1", "C1"],
            "brand": ["Hourglass", "Hourglass", "Brand B", "Brand C"],
            "product": ["Liquid Foundation", "Foundation Stick", "Base", "Base"],
            "shade_name": ["Vanilla", "Vanilla", "Sand", "Tan"],
            "hex": ["#C8A080", "#C9A181", "#B89070", "#A07050"],
            "r": [200, 201, 184, 160],
            "g": [160, 161, 144, 112],
            "b": [128, 129, 112, 80],
            "lab_l": [60.0, 60.2, 56.0, 50.0],
            "lab_a": [10.0, 10.1, 9.0, 8.0],
            "lab_b": [18.0, 18.1, 17.0, 16.0],
        }
    )
    matches = match_shades(np.array([60.0, 10.0, 18.0]), df, top_k=3)
    keys = [(m.brand, m.shade_name) for m in matches]
    assert keys.count(("Hourglass", "Vanilla")) == 1
    hourglass = matches[0]
    assert hourglass.product == "Liquid Foundation"
    assert [v["product"] for v in hourglass.product_variants] == ["Foundation Stick"]
    assert len(matches) == 3


def test_same_shade_name_across_different_brands_not_merged():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "B1"],
            "brand": ["Hourglass", "Fenty"],
            "product": ["Liquid", "Liquid"],
            "shade_name": ["Vanilla", "Vanilla"],
            "hex": ["#C8A080", "#C8A080"],
            "r": [200, 200],
            "g": [160, 160],
            "b": [128, 128],
            "lab_l": [60.0, 60.0],
            "lab_a": [10.0, 10.0],
            "lab_b": [18.0, 18.0],
        }
    )
    matches = match_shades(np.array([60.0, 10.0, 18.0]), df, top_k=2)
    assert len(matches) == 2
    assert {m.brand for m in matches} == {"Hourglass", "Fenty"}


def test_exact_duplicate_rows_removed_from_variant_list():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "A1_DUP", "A2", "B1"],
            "brand": ["Hourglass", "Hourglass", "Hourglass", "Brand B"],
            "product": ["Liquid", "Liquid", "Stick", "Base"],
            "shade_name": ["Vanilla", "Vanilla", "Vanilla", "Sand"],
            "hex": ["#C8A080", "#C8A080", "#C9A181", "#B89070"],
            "r": [200, 200, 201, 184],
            "g": [160, 160, 161, 144],
            "b": [128, 128, 129, 112],
            "lab_l": [60.0, 60.0, 60.2, 56.0],
            "lab_a": [10.0, 10.0, 10.1, 9.0],
            "lab_b": [18.0, 18.0, 18.1, 17.0],
        }
    )
    matches = match_shades(np.array([60.0, 10.0, 18.0]), df, top_k=2)
    hourglass = matches[0]
    assert len(hourglass.product_variants) == 1
    assert hourglass.product_variants[0]["product"] == "Stick"


def test_same_brand_product_nearly_identical_hex_similar_shade_names_grouped():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "A2", "B1"],
            "brand": ["Brand A", "Brand A", "Brand B"],
            "product": ["Liquid Base", "Liquid Base", "Liquid Base"],
            "shade_name": ["450N", "450 Neutral", "460N"],
            "hex": ["#4F3B32", "#503C33", "#5B453A"],
            "r": [79, 80, 91],
            "g": [59, 60, 69],
            "b": [50, 51, 58],
            "lab_l": [27.0, 27.2, 31.0],
            "lab_a": [6.0, 6.1, 7.0],
            "lab_b": [9.0, 9.1, 10.0],
        }
    )

    matches = match_shades(np.array([27.0, 6.0, 9.0]), df, top_k=3)

    assert len(matches) == 2
    assert matches[0].shade_name == "450N"
    assert [variant["shade_name"] for variant in matches[0].product_variants] == ["450 Neutral"]


def test_same_brand_product_same_hex_slightly_different_shade_name_grouped():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "A2", "B1"],
            "brand": ["Brand A", "Brand A", "Brand B"],
            "product": ["Liquid Base", "Liquid Base", "Liquid Base"],
            "shade_name": ["450 C", "450 Cool", "451 Cool"],
            "hex": ["#4F3B32", "#4F3B32", "#554037"],
            "r": [79, 79, 85],
            "g": [59, 59, 64],
            "b": [50, 50, 55],
            "lab_l": [27.0, 27.0, 29.5],
            "lab_a": [6.0, 6.0, 6.5],
            "lab_b": [9.0, 9.0, 9.5],
        }
    )

    matches = match_shades(np.array([27.0, 6.0, 9.0]), df, top_k=3)

    assert len(matches) == 2
    assert matches[0].shade_name == "450 C"
    assert [variant["shade_name"] for variant in matches[0].product_variants] == ["450 Cool"]


def test_visual_distinctness_skips_nearly_identical_display_candidate_when_possible():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "B1", "C1", "D1"],
            "brand": ["Brand A", "Brand B", "Brand C", "Brand D"],
            "product": ["Base", "Base", "Base", "Base"],
            "shade_name": ["One", "Twin", "Two", "Three"],
            "hex": ["#4F3B32", "#4F3B32", "#60483C", "#735648"],
            "r": [79, 79, 96, 115],
            "g": [59, 59, 72, 86],
            "b": [50, 50, 60, 72],
            "lab_l": [27.0, 27.0, 32.0, 38.0],
            "lab_a": [6.0, 6.0, 7.0, 8.0],
            "lab_b": [9.0, 9.0, 10.0, 12.0],
        }
    )

    matches = match_shades(np.array([27.0, 6.0, 9.0]), df, top_k=3)

    assert len(matches) == 3
    assert [m.brand for m in matches] == ["Brand A", "Brand C", "Brand D"]


def test_different_brands_same_hex_preserved_when_needed_for_top_k():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "B1"],
            "brand": ["Brand A", "Brand B"],
            "product": ["Base", "Base"],
            "shade_name": ["One", "Twin"],
            "hex": ["#4F3B32", "#4F3B32"],
            "r": [79, 79],
            "g": [59, 59],
            "b": [50, 50],
            "lab_l": [27.0, 27.0],
            "lab_a": [6.0, 6.0],
            "lab_b": [9.0, 9.0],
        }
    )

    matches = match_shades(np.array([27.0, 6.0, 9.0]), df, top_k=2)

    assert len(matches) == 2
    assert {m.brand for m in matches} == {"Brand A", "Brand B"}


def test_top_three_returns_distinct_candidates_when_enough_unique_shades_exist():
    df = pd.DataFrame(
        {
            "shade_id": ["A1", "A2", "B1", "C1", "D1"],
            "brand": ["Brand A", "Brand A", "Brand B", "Brand C", "Brand D"],
            "product": ["Liquid", "Stick", "Base", "Base", "Base"],
            "shade_name": ["One", "One", "Two", "Three", "Four"],
            "hex": ["#A07050", "#A17151", "#A87858", "#B08060", "#C09070"],
            "r": [160, 161, 168, 176, 192],
            "g": [112, 113, 120, 128, 144],
            "b": [80, 81, 88, 96, 112],
            "lab_l": [50.0, 50.1, 51.0, 52.0, 54.0],
            "lab_a": [8.0, 8.1, 9.0, 10.0, 11.0],
            "lab_b": [16.0, 16.1, 17.0, 18.0, 19.0],
        }
    )
    matches = match_shades(np.array([50.0, 8.0, 16.0]), df, top_k=3)
    assert len(matches) == 3
    assert len({(m.brand, m.shade_name) for m in matches}) == 3


def test_uncertainty_samples_add_recommendation_stability():
    df = pd.DataFrame(
        {
            "shade_id": ["A", "B", "C", "D"],
            "brand": ["A", "B", "C", "D"],
            "product": ["Foundation"] * 4,
            "shade_name": ["A", "B", "C", "D"],
            "hex": ["#806050", "#876657", "#927060", "#A08070"],
            "r": [128, 135, 146, 160],
            "g": [96, 102, 112, 128],
            "b": [80, 87, 96, 112],
            "lab_l": [45.0, 48.0, 52.0, 58.0],
            "lab_a": [8.0, 8.5, 9.0, 10.0],
            "lab_b": [14.0, 14.5, 15.0, 16.0],
            "depth": ["tan", "tan", "medium", "medium"],
            "product_type": ["foundation"] * 4,
            "catalog_quality_score": [1.0] * 4,
        }
    )
    samples = np.array(
        [[45.0 + offset, 8.0, 14.0] for offset in np.linspace(-0.5, 0.5, 21)]
    )

    matches = match_shades(
        np.array([45.0, 8.0, 14.0]),
        df,
        top_k=3,
        uncertainty_labs=samples,
    )

    assert matches[0].recommendation_stability is not None
    assert matches[0].recommendation_stability > 0.8
    assert matches[0].top3_stability == 1.0
    assert matches[0].delta_e_median is not None
    assert matches[0].delta_e_p90 is not None


def test_uncertainty_distribution_can_promote_consistently_better_shade():
    df = pd.DataFrame(
        {
            "shade_id": ["point", "robust", "far"],
            "brand": ["Point", "Robust", "Far"],
            "product": ["Foundation"] * 3,
            "shade_name": ["Point", "Robust", "Far"],
            "hex": ["#806050", "#8A6858", "#A08070"],
            "r": [128, 138, 160],
            "g": [96, 104, 128],
            "b": [80, 88, 112],
            "lab_l": [50.0, 54.0, 65.0],
            "lab_a": [8.0, 8.0, 8.0],
            "lab_b": [14.0, 14.0, 14.0],
        }
    )
    samples = np.repeat([[54.0, 8.0, 14.0]], repeats=31, axis=0)

    point_only = match_shades(np.array([50.0, 8.0, 14.0]), df, top_k=1)
    distribution_ranked = match_shades(
        np.array([50.0, 8.0, 14.0]),
        df,
        top_k=1,
        uncertainty_labs=samples,
    )

    assert point_only[0].shade_id == "point"
    assert distribution_ranked[0].shade_id == "robust"
    assert distribution_ranked[0].delta_e > point_only[0].delta_e
    assert distribution_ranked[0].delta_e_median < distribution_ranked[0].delta_e
    assert distribution_ranked[0].uncertainty_adjustment < 0.0


def test_invalid_uncertainty_samples_leave_point_ranking_unchanged():
    df = pd.DataFrame(
        {
            "shade_id": ["A", "B"],
            "brand": ["A", "B"],
            "shade_name": ["A", "B"],
            "hex": ["#806050", "#906858"],
            "r": [128, 144],
            "g": [96, 104],
            "b": [80, 88],
            "lab_l": [50.0, 58.0],
            "lab_a": [8.0, 8.0],
            "lab_b": [14.0, 14.0],
        }
    )

    matches = match_shades(
        np.array([50.0, 8.0, 14.0]),
        df,
        top_k=2,
        uncertainty_labs=np.array([[np.nan, 8.0, 14.0]]),
    )

    assert [match.shade_id for match in matches] == ["A", "B"]
    assert all(match.delta_e_median is None for match in matches)


def test_lighting_sensitivity_samples_influence_ranking_and_stability():
    df = pd.DataFrame(
        {
            "shade_id": ["point", "stable", "far"],
            "brand": ["Point", "Stable", "Far"],
            "product": ["Foundation"] * 3,
            "shade_name": ["Point", "Stable", "Far"],
            "hex": ["#806050", "#8A6858", "#A08070"],
            "r": [128, 138, 160],
            "g": [96, 104, 128],
            "b": [80, 88, 112],
            "lab_l": [50.0, 54.0, 65.0],
            "lab_a": [8.0, 8.0, 8.0],
            "lab_b": [14.0, 14.0, 14.0],
        }
    )
    lighting_samples = np.repeat([[54.0, 8.0, 14.0]], repeats=6, axis=0)

    matches = match_shades(
        np.array([50.0, 8.0, 14.0]),
        df,
        top_k=2,
        lighting_sensitivity_labs=lighting_samples,
    )

    assert matches[0].shade_id == "stable"
    assert matches[0].lighting_recommendation_stability == 1.0
    assert matches[0].lighting_top3_stability == 1.0
    assert matches[0].lighting_delta_e_p90 == 0.0


def test_supported_uncertainty_range_prevents_unjustified_too_light_penalty():
    assert _too_light_penalty(50.0, 57.0) > 0.0
    assert _too_light_penalty(50.0, 57.0, supported_upper_l=55.0) == 0.0
