"""Shade catalog loading, normalization, metadata, and Lab conversion."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.color import rgb2lab

from src.config import MOCK_SHADE_CATALOG_PATH, PUBLIC_SHADE_CATALOG_PATH

REQUIRED_BASE_COLUMNS = ["shade_id", "brand", "shade_name"]
OPTIONAL_COLUMNS = ["product", "undertone", "depth", "notes", "source", "source_url"]

HEX_PATTERN = re.compile(r"^#?[0-9A-Fa-f]{6}$")
PUBLIC_CATALOG_KEY = "public"
MOCK_CATALOG_KEY = "mock"
PUBLIC_CATALOG_NAME = "Public Sephora-style catalog"
MOCK_CATALOG_NAME = "Mock development catalog"
PUBLIC_CATALOG_SOURCE = "Public Sephora-style foundation swatch dataset"
MOCK_CATALOG_SOURCE = "Local mock development catalog"
PUBLIC_CATALOG_LIMITATION = (
    "Public catalog colors are website-derived swatch approximations and may differ "
    "from real applied foundation due to lighting, display calibration, brand image "
    "processing, oxidation, and skin texture."
)


@dataclass(frozen=True)
class CatalogDefinition:
    key: str
    name: str
    path: Path
    source: str


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


def catalog_definitions(
    public_path: str | Path = PUBLIC_SHADE_CATALOG_PATH,
    mock_path: str | Path = MOCK_SHADE_CATALOG_PATH,
) -> dict[str, CatalogDefinition]:
    """Return supported local catalog definitions."""
    return {
        PUBLIC_CATALOG_KEY: CatalogDefinition(
            key=PUBLIC_CATALOG_KEY,
            name=PUBLIC_CATALOG_NAME,
            path=Path(public_path),
            source=PUBLIC_CATALOG_SOURCE,
        ),
        MOCK_CATALOG_KEY: CatalogDefinition(
            key=MOCK_CATALOG_KEY,
            name=MOCK_CATALOG_NAME,
            path=Path(mock_path),
            source=MOCK_CATALOG_SOURCE,
        ),
    }


def _catalog_attrs(
    catalog_df: pd.DataFrame,
    path: str | Path,
    catalog_name: str | None,
    catalog_source: str | None,
    warnings: list[str],
    dropped_count: int,
) -> None:
    source_values = []
    if "source" in catalog_df.columns:
        source_values = [
            str(v).strip()
            for v in catalog_df["source"].dropna().unique().tolist()
            if str(v).strip()
        ]
    source = catalog_source or (source_values[0] if source_values else "unknown")
    catalog_df.attrs["catalog_name"] = catalog_name or Path(path).stem
    catalog_df.attrs["source"] = source
    catalog_df.attrs["warnings"] = warnings
    catalog_df.attrs["dropped_count"] = dropped_count
    catalog_df.attrs["valid_count"] = len(catalog_df)


def load_shade_catalog(
    path: str | Path,
    catalog_name: str | None = None,
    catalog_source: str | None = None,
) -> pd.DataFrame:
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

    _catalog_attrs(
        catalog_df=catalog_df,
        path=path,
        catalog_name=catalog_name,
        catalog_source=catalog_source,
        warnings=warnings,
        dropped_count=len(raw_df) - len(valid_rows),
    )

    return catalog_df


def load_named_catalog(
    key: str,
    public_path: str | Path = PUBLIC_SHADE_CATALOG_PATH,
    mock_path: str | Path = MOCK_SHADE_CATALOG_PATH,
) -> pd.DataFrame:
    """Load one supported catalog by key."""
    definitions = catalog_definitions(public_path=public_path, mock_path=mock_path)
    if key not in definitions:
        raise CatalogValidationError(f"Unknown catalog key: {key}")
    definition = definitions[key]
    return load_shade_catalog(
        definition.path,
        catalog_name=definition.name,
        catalog_source=definition.source,
    )


def load_default_catalog(
    public_path: str | Path = PUBLIC_SHADE_CATALOG_PATH,
    mock_path: str | Path = MOCK_SHADE_CATALOG_PATH,
) -> tuple[str, pd.DataFrame, list[str]]:
    """Load public catalog when valid, otherwise fall back to mock catalog."""
    fallback_warnings = []
    try:
        public_df = load_named_catalog(
            PUBLIC_CATALOG_KEY, public_path=public_path, mock_path=mock_path
        )
        return PUBLIC_CATALOG_KEY, public_df, []
    except (FileNotFoundError, CatalogValidationError) as exc:
        fallback_warnings.append(
            f"Public catalog unavailable or invalid; using mock catalog instead. {exc}"
        )

    mock_df = load_named_catalog(MOCK_CATALOG_KEY, public_path=public_path, mock_path=mock_path)
    return MOCK_CATALOG_KEY, mock_df, fallback_warnings
