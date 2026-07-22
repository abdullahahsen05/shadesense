import numpy as np
import pandas as pd

from scripts.prepare_public_catalog import (
    SOURCE_LABEL,
    infer_depth,
    infer_undertone,
    normalize_hex,
    prepare_public_catalog,
)
from src.shade_catalog import (
    MOCK_CATALOG_KEY,
    PUBLIC_CATALOG_KEY,
    load_default_catalog,
    load_shade_catalog,
)
from src.shade_matcher import match_shades


def test_normalize_hex_accepts_hashless_and_uppercases():
    assert normalize_hex("aabbcc") == "#AABBCC"
    assert normalize_hex("#aaBBcc") == "#AABBCC"
    assert normalize_hex("not-a-hex") is None
    assert normalize_hex("#12345") is None


def test_infer_undertone_from_shade_text():
    assert infer_undertone("Warm Golden Honey") == "warm"
    assert infer_undertone("Cool Rose Pink") == "cool"
    assert infer_undertone("Neutral Beige Natural") == "neutral"
    assert infer_undertone("Olive medium") == "olive"
    assert infer_undertone("Porcelain 101") == "unknown"


def test_infer_depth_from_hex_and_lab_lightness():
    assert infer_depth("#F4E0CB") in {"fair", "light"}
    assert infer_depth("#A06F4A") in {"medium", "tan"}
    assert infer_depth("#2A1A14") == "rich-deep"
    assert infer_depth("#FFFFFF", lightness=0.5) == "tan"


def test_prepare_public_catalog_with_synthetic_raw_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "brand": ["Brand A", "Brand A", "Brand B", "Brand C"],
            "product": ["Base", "Base", "Tint", "Nope"],
            "description": [
                "101 warm golden undertone",
                "101 warm golden undertone",
                "Deep neutral shade",
                "Invalid color",
            ],
            "name": ["Golden 101", "Golden 101", None, "Bad"],
            "specific": ["101W", "101W", "500N", "000"],
            "hex": ["f1caaa", "#F1CAAA", "#4B3028", "XYZ"],
            "url": ["https://example.test/a", "https://example.test/a", "", ""],
            "lightness": [0.82, 0.82, 0.25, 0.9],
        }
    ).to_csv(raw_dir / "colors.csv", index=False)
    pd.DataFrame({"brand": ["No Hex"], "product": ["Skipped"]}).to_csv(
        raw_dir / "no_hex.csv", index=False
    )

    output_path = tmp_path / "public.csv"
    summary = prepare_public_catalog(raw_dir, output_path)
    out = pd.read_csv(output_path)

    assert summary.total_raw_rows == 5
    assert summary.valid_rows_written == 2
    assert summary.skipped_rows == 2
    assert summary.duplicate_rows_removed == 1
    assert len(summary.warnings) == 2
    assert list(out.columns) == [
        "shade_id",
        "brand",
        "product",
        "shade_name",
        "hex",
        "undertone",
        "depth",
        "source",
        "source_url",
    ]
    assert out.iloc[0]["hex"] == "#F1CAAA"
    assert out.iloc[0]["undertone"] == "warm"
    assert out.iloc[1]["source"] == SOURCE_LABEL


def test_public_catalog_loader_metadata_and_matching(tmp_path):
    public_path = tmp_path / "public.csv"
    pd.DataFrame(
        {
            "shade_id": ["P1", "P2", "P3"],
            "brand": ["Brand A", "Brand A", "Brand B"],
            "product": ["Foundation", "Foundation", "Tint"],
            "shade_name": ["Light Neutral", "Medium Warm", "Deep Cool"],
            "hex": ["#E6C8B0", "#A87553", "#4C3028"],
            "undertone": ["neutral", "warm", "cool"],
            "depth": ["light", "tan", "deep"],
            "source": [SOURCE_LABEL, SOURCE_LABEL, SOURCE_LABEL],
            "source_url": ["u1", "u2", "u3"],
        }
    ).to_csv(public_path, index=False)

    catalog = load_shade_catalog(public_path, catalog_name="Public Test")
    assert catalog.attrs["catalog_name"] == "Public Test"
    assert catalog.attrs["source"] == SOURCE_LABEL
    assert catalog.attrs["valid_count"] == 3

    skin_lab = catalog.iloc[1][["lab_l", "lab_a", "lab_b"]].to_numpy(dtype=float)
    matches = match_shades(np.array(skin_lab), catalog, top_k=3)
    assert len(matches) == 3
    assert matches[0].product == "Foundation"
    assert matches[0].source_url == "u2"


def test_default_catalog_prefers_public_and_falls_back_to_mock(tmp_path):
    public_path = tmp_path / "missing_public.csv"
    mock_path = tmp_path / "mock.csv"
    pd.DataFrame(
        {
            "shade_id": ["M1", "M2", "M3"],
            "brand": ["Mock", "Mock", "Mock"],
            "shade_name": ["One", "Two", "Three"],
            "hex": ["#111111", "#777777", "#EEEEEE"],
        }
    ).to_csv(mock_path, index=False)

    key, catalog, warnings = load_default_catalog(public_path=public_path, mock_path=mock_path)
    assert key == MOCK_CATALOG_KEY
    assert len(catalog) == 3
    assert warnings

    pd.DataFrame(
        {
            "shade_id": ["P1"],
            "brand": ["Public"],
            "product": ["Base"],
            "shade_name": ["One"],
            "hex": ["#222222"],
            "source": [SOURCE_LABEL],
        }
    ).to_csv(public_path, index=False)
    key, catalog, warnings = load_default_catalog(public_path=public_path, mock_path=mock_path)
    assert key == PUBLIC_CATALOG_KEY
    assert len(catalog) == 1
    assert warnings == []
