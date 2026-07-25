import numpy as np
import pandas as pd

from scripts.prepare_public_catalog import (
    SOURCE_LABEL,
    infer_depth,
    infer_undertone,
    looks_like_complexion_product,
    normalize_hex,
    prepare_public_catalog,
)
from src.shade_catalog import (
    FOUNDATION_ONLY_SCOPE,
    MOCK_CATALOG_KEY,
    PUBLIC_CATALOG_KEY,
    classify_product_type,
    filter_catalog_by_product_scope,
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


def test_complexion_product_filter_keeps_base_and_rejects_obvious_other_makeup():
    assert looks_like_complexion_product("Longwear Foundation", "warm beige")
    assert looks_like_complexion_product("Skin Tint SPF 40", "medium neutral")
    assert looks_like_complexion_product("Foundation + Concealer", "deep")
    assert looks_like_complexion_product("CC+ Cream with SPF 50+", "tan")
    assert not looks_like_complexion_product("Matte Lipstick", "red")
    assert not looks_like_complexion_product("Volumizing Mascara", "black")
    assert not looks_like_complexion_product("Shimmer Eyeshadow Palette", "bronze")


def test_product_type_parser_recognizes_real_catalog_variants():
    assert classify_product_type("Your Skin But Better CC+ Cream") == "bb_cc"
    assert classify_product_type("BB Cream SPF 30") == "bb_cc"
    assert classify_product_type("Tinted Face Serum") == "tint"
    assert classify_product_type("Skin Tint") == "tint"
    assert (
        classify_product_type("Foundation and Concealer Stick")
        == "concealer_hybrid"
    )
    assert classify_product_type("Longwear Foundation Stick") == "stick"
    assert classify_product_type("Pressed Foundation Powder") == "powder"


def test_foundation_only_scope_preserves_true_foundation_forms(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "shade_id": [f"S{index}" for index in range(7)],
            "brand": ["B"] * 7,
            "product": [
                "Liquid Foundation",
                "Foundation Stick",
                "Foundation Powder",
                "Cushion Foundation",
                "CC+ Cream",
                "Skin Tint",
                "Foundation Concealer",
            ],
            "shade_name": [f"Shade {index}" for index in range(7)],
            "hex": ["#AA8877"] * 7,
        }
    ).to_csv(path, index=False)
    catalog = load_shade_catalog(path)

    filtered = filter_catalog_by_product_scope(
        catalog,
        FOUNDATION_ONLY_SCOPE,
    )

    assert set(filtered["product_type"]) == {
        "foundation",
        "stick",
        "powder",
    }
    assert len(filtered) == 3
    assert filtered.attrs["unfiltered_count"] == 7


def test_prepare_public_catalog_with_synthetic_raw_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "brand": ["Brand A", "Brand A", "Brand B", "Brand C"],
            "product": ["Base Foundation", "Base Foundation", "Matte Lipstick", "Nope"],
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
    assert summary.valid_rows_written == 1
    assert summary.skipped_rows == 3
    assert summary.duplicate_rows_removed == 1
    assert summary.non_complexion_rows_skipped == 1
    assert len(summary.warnings) == 3
    assert list(out.columns) == [
        "shade_id",
        "brand",
        "product",
        "shade_name",
        "hex",
        "undertone",
        "depth",
        "product_type",
        "catalog_quality_score",
        "source",
        "source_url",
    ]
    assert out.iloc[0]["hex"] == "#F1CAAA"
    assert out.iloc[0]["undertone"] == "warm"
    assert out.iloc[0]["product_type"] == "foundation"
    assert 0.0 <= out.iloc[0]["catalog_quality_score"] <= 1.0
    assert out.iloc[0]["source"] == SOURCE_LABEL


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
    assert list(catalog["product_type"]) == ["foundation", "foundation", "tint"]
    assert catalog["catalog_quality_score"].between(0.0, 1.0).all()

    skin_lab = catalog.iloc[1][["lab_l", "lab_a", "lab_b"]].to_numpy(dtype=float)
    matches = match_shades(np.array(skin_lab), catalog, top_k=3)
    assert len(matches) == 3
    assert matches[0].product == "Foundation"
    assert matches[0].source_url == "u2"
    assert matches[0].product_type == "foundation"
    assert matches[0].catalog_quality_score > 0.5


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
