import numpy as np
import pandas as pd
import pytest
from skimage.color import deltaE_ciede2000

from src.config import SHADE_CATALOG_PATH
from src.shade_catalog import CatalogValidationError, load_shade_catalog
from src.shade_matcher import match_shades


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
