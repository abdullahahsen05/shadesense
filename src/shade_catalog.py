"""Shade catalog loading, normalization, and Lab conversion."""

import re

import numpy as np
import pandas as pd
from skimage.color import rgb2lab

REQUIRED_BASE_COLUMNS = ["shade_id", "brand", "shade_name"]
OPTIONAL_COLUMNS = ["undertone", "depth", "notes"]

HEX_PATTERN = re.compile(r"^#?[0-9A-Fa-f]{6}$")


class CatalogValidationError(ValueError):
    """Raised when the shade catalog is structurally invalid or unusable."""


def _normalize_hex(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not HEX_PATTERN.match(text):
        return None
    text = text.lstrip("#").upper()
    return f"#{text}"


def _hex_to_rgb(hex_value: str) -> tuple:
    text = hex_value.lstrip("#")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r, g, b) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _valid_rgb_component(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return 0 <= v <= 255 and not np.isnan(v)


def load_shade_catalog(path: str) -> pd.DataFrame:
    """Load and normalize a shade catalog CSV.

    Validates required columns, normalizes HEX/RGB (deriving whichever is
    missing from the other), converts colors to Lab, and fills in optional
    columns (undertone/depth/notes) if absent. Rows with unusable color data
    are dropped; details are recorded in `df.attrs["warnings"]`.

    Raises:
        FileNotFoundError: the catalog file does not exist.
        CatalogValidationError: required columns are missing, or zero rows
            have usable color data after validation.
    """
    try:
        raw_df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Shade catalog not found at: {path}")

    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in raw_df.columns]
    if missing_base:
        raise CatalogValidationError(
            f"Shade catalog is missing required columns: {missing_base}"
        )

    has_hex_col = "hex" in raw_df.columns
    has_rgb_cols = all(c in raw_df.columns for c in ("r", "g", "b"))
    if not has_hex_col and not has_rgb_cols:
        raise CatalogValidationError(
            "Shade catalog must provide either a 'hex' column or 'r','g','b' columns."
        )

    for col in OPTIONAL_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = None

    warnings = []
    valid_rows = []

    for idx, row in raw_df.iterrows():
        shade_ref = row.get("shade_id", f"row {idx}")

        rgb = None
        if has_rgb_cols and all(_valid_rgb_component(row[c]) for c in ("r", "g", "b")):
            rgb = tuple(int(round(float(row[c]))) for c in ("r", "g", "b"))
        elif has_hex_col:
            norm_hex = _normalize_hex(row.get("hex"))
            if norm_hex is not None:
                rgb = _hex_to_rgb(norm_hex)

        if rgb is None:
            warnings.append(f"Shade '{shade_ref}' has invalid or missing color data; skipped.")
            continue

        hex_value = _normalize_hex(row.get("hex")) if has_hex_col else None
        if hex_value is None:
            hex_value = _rgb_to_hex(*rgb)

        new_row = row.to_dict()
        new_row["r"], new_row["g"], new_row["b"] = rgb
        new_row["hex"] = hex_value
        valid_rows.append(new_row)

    if not valid_rows:
        raise CatalogValidationError(
            "No valid shades found in the catalog after color validation. "
            f"Issues: {warnings}"
        )

    catalog_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    rgb_array = catalog_df[["r", "g", "b"]].to_numpy(dtype=np.float64)
    lab_array = rgb2lab(rgb_array.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    catalog_df["lab_l"] = lab_array[:, 0]
    catalog_df["lab_a"] = lab_array[:, 1]
    catalog_df["lab_b"] = lab_array[:, 2]

    catalog_df.attrs["warnings"] = warnings
    catalog_df.attrs["dropped_count"] = len(raw_df) - len(valid_rows)
    catalog_df.attrs["valid_count"] = len(valid_rows)

    return catalog_df
