"""Prepare a normalized public foundation shade catalog from raw CSV exports."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from skimage.color import rgb2lab

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "public_catalog_raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "public_shade_catalog.csv"

SOURCE_LABEL = "Public Sephora-style foundation swatch dataset"
OUTPUT_COLUMNS = [
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

HEX_PATTERN = re.compile(r"^#?[0-9A-Fa-f]{6}$")

BRAND_COLUMNS = ["brand", "brand_name", "brandName"]
PRODUCT_COLUMNS = ["product", "product_name", "productName", "title", "description"]
SHADE_COLUMNS = ["name", "shade", "shade_name", "shadeName", "specific", "imgAlt", "description"]
HEX_COLUMNS = ["hex", "HEX", "Hex", "color_hex", "colour_hex", "swatch_hex"]
URL_COLUMNS = ["url", "source_url", "sourceUrl", "product_url", "productUrl", "imgSrc", "image", "image_url"]
COMPLEXION_INCLUDE_TERMS = [
    "foundation",
    "skin tint",
    "tinted moisturizer",
    "tinted moisturiser",
    "complexion",
    "base",
    "concealer",
    "bb",
    "cc",
    "cushion",
    "cover drops",
    "cover cream",
    "cover creme",
    "makeup",
    "teint",
    "tint",
]
NON_COMPLEXION_EXCLUDE_TERMS = [
    "lipstick",
    "lip gloss",
    "mascara",
    "eyeliner",
    "eyeshadow",
    "eye shadow",
    "brow pencil",
    "brow gel",
    "brow pomade",
    "blush",
    "bronzer",
    "highlighter",
    "fragrance",
    "perfume",
    "nail polish",
]


@dataclass
class PrepareSummary:
    raw_files: list[str] = field(default_factory=list)
    column_names: dict[str, list[str]] = field(default_factory=dict)
    mapped_columns: dict[str, dict[str, str | None]] = field(default_factory=dict)
    total_raw_rows: int = 0
    valid_rows_written: int = 0
    skipped_rows: int = 0
    duplicate_rows_removed: int = 0
    non_complexion_rows_skipped: int = 0
    brand_count: int = 0
    product_count: int = 0
    shade_count: int = 0
    output_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def normalize_hex(value) -> str | None:
    """Return #RRGGBB for valid hex input, otherwise None."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not HEX_PATTERN.match(text):
        return None
    return f"#{text.lstrip('#').upper()}"


def hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    text = hex_value.lstrip("#")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def infer_undertone(*values) -> str:
    text = " ".join(str(v).lower() for v in values if not pd.isna(v))
    checks = [
        ("olive", ["olive"]),
        ("warm", ["warm", "golden", "yellow", "peach", "honey"]),
        ("cool", ["cool", "rose", "pink", "red"]),
        ("neutral", ["neutral", "beige", "natural"]),
    ]
    for label, terms in checks:
        if any(term in text for term in terms):
            return label
    return "unknown"


def infer_depth(hex_value: str, lightness=None) -> str:
    """Infer a conservative depth bucket from Lab L* or dataset lightness."""
    l_star = None
    if lightness is not None and not pd.isna(lightness):
        try:
            numeric = float(lightness)
            l_star = numeric * 100 if numeric <= 1.0 else numeric
        except (TypeError, ValueError):
            l_star = None
    if l_star is None:
        rgb = [c / 255.0 for c in hex_to_rgb(hex_value)]
        l_star = float(rgb2lab([[rgb]])[0][0][0])

    if l_star >= 85:
        return "fair"
    if l_star >= 75:
        return "light"
    if l_star >= 65:
        return "light-medium"
    if l_star >= 55:
        return "medium"
    if l_star >= 45:
        return "tan"
    if l_star >= 32:
        return "deep"
    return "rich-deep"


def _first_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _clean_text(value, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    text = " ".join(str(value).strip().split())
    return text if text else fallback


def looks_like_complexion_product(*values) -> bool:
    """Conservatively identify rows that look relevant to base complexion."""
    text = " ".join(str(v).lower() for v in values if not pd.isna(v))
    has_complexion_signal = any(term in text for term in COMPLEXION_INCLUDE_TERMS)
    has_non_complexion_signal = any(term in text for term in NON_COMPLEXION_EXCLUDE_TERMS)
    if has_complexion_signal:
        return True
    if has_non_complexion_signal:
        return False
    return False


def _shade_name(row: pd.Series, columns: list[str], row_number: int) -> str:
    for col in SHADE_COLUMNS:
        actual = _first_column(columns, [col])
        if actual is None:
            continue
        text = _clean_text(row.get(actual), "")
        if text:
            return text.removesuffix(" selected").strip()
    return f"Shade {row_number:03d}"


def _normalize_file(path: Path, summary: PrepareSummary) -> list[dict[str, str]]:
    df = pd.read_csv(path)
    summary.raw_files.append(path.name)
    summary.column_names[path.name] = list(df.columns)
    summary.total_raw_rows += len(df)

    columns = list(df.columns)
    brand_col = _first_column(columns, BRAND_COLUMNS)
    product_col = _first_column(columns, PRODUCT_COLUMNS)
    hex_col = _first_column(columns, HEX_COLUMNS)
    url_col = _first_column(columns, URL_COLUMNS)
    lightness_col = _first_column(columns, ["lightness", "light_to_dark", "l", "lab_l"])

    summary.mapped_columns[path.name] = {
        "brand": brand_col,
        "product": product_col,
        "shade_name": _first_column(columns, SHADE_COLUMNS),
        "hex": hex_col,
        "source_url": url_col,
        "lightness": lightness_col,
    }

    if hex_col is None:
        summary.skipped_rows += len(df)
        summary.warnings.append(f"{path.name}: no hex-like column found; skipped {len(df)} rows.")
        return []

    rows = []
    for idx, row in df.iterrows():
        hex_value = normalize_hex(row.get(hex_col))
        if hex_value is None:
            summary.skipped_rows += 1
            summary.warnings.append(f"{path.name} row {idx + 2}: invalid hex value; skipped.")
            continue

        row_number = len(rows) + 1
        brand = _clean_text(row.get(brand_col) if brand_col else None, "unknown")
        product = _clean_text(row.get(product_col) if product_col else None, "unknown")
        shade_name = _shade_name(row, columns, row_number)
        text_values = [
            shade_name,
            product,
            row.get("description") if "description" in columns else None,
            row.get("imgAlt") if "imgAlt" in columns else None,
            row.get("specific") if "specific" in columns else None,
        ]
        if not looks_like_complexion_product(*text_values):
            summary.skipped_rows += 1
            summary.non_complexion_rows_skipped += 1
            summary.warnings.append(
                f"{path.name} row {idx + 2}: no base-complexion product signal; skipped."
            )
            continue

        rows.append(
            {
                "brand": brand,
                "product": product,
                "shade_name": shade_name,
                "hex": hex_value,
                "undertone": infer_undertone(*text_values),
                "depth": infer_depth(hex_value, row.get(lightness_col) if lightness_col else None),
                "source": SOURCE_LABEL,
                "source_url": _clean_text(row.get(url_col) if url_col else None, ""),
            }
        )
    return rows


def prepare_public_catalog(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> PrepareSummary:
    summary = PrepareSummary(output_path=output_path)
    csv_paths = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    normalized_rows = []
    for path in csv_paths:
        normalized_rows.extend(_normalize_file(path, summary))

    unique_rows = []
    seen = set()
    for row in normalized_rows:
        key = (
            row["brand"].casefold(),
            row["product"].casefold(),
            row["shade_name"].casefold(),
            row["hex"],
        )
        if key in seen:
            summary.duplicate_rows_removed += 1
            continue
        seen.add(key)
        unique_rows.append(row)

    for idx, row in enumerate(unique_rows, start=1):
        row["shade_id"] = f"PUBLIC-{idx:05d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(unique_rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(output_path, index=False)

    summary.valid_rows_written = len(out_df)
    summary.brand_count = out_df["brand"].nunique() if len(out_df) else 0
    summary.product_count = out_df["product"].nunique() if len(out_df) else 0
    summary.shade_count = out_df["shade_id"].nunique() if len(out_df) else 0
    return summary


def print_summary(summary: PrepareSummary) -> None:
    print("Raw files:")
    for file_name in summary.raw_files:
        print(f"  - {file_name}")
        print(f"    columns: {summary.column_names[file_name]}")
        print(f"    mapped: {summary.mapped_columns[file_name]}")
    print(f"Total raw rows: {summary.total_raw_rows}")
    print(f"Valid rows written: {summary.valid_rows_written}")
    print(f"Skipped rows: {summary.skipped_rows}")
    print(f"Duplicate rows removed: {summary.duplicate_rows_removed}")
    print(f"Non-complexion rows skipped: {summary.non_complexion_rows_skipped}")
    print(f"Number of brands: {summary.brand_count}")
    print(f"Number of products: {summary.product_count}")
    print(f"Number of shades: {summary.shade_count}")
    print(f"Output path: {summary.output_path}")
    if summary.warnings:
        print("Warnings:")
        for warning in summary.warnings[:25]:
            print(f"  - {warning}")
        if len(summary.warnings) > 25:
            print(f"  - ... {len(summary.warnings) - 25} more warnings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = prepare_public_catalog(args.raw_dir, args.output)
    print_summary(summary)


if __name__ == "__main__":
    main()
