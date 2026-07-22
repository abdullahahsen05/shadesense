"""Shade matching via perceptual (CIEDE2000) color distance in Lab space."""

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

try:
    from skimage.color import deltaE_ciede2000

    _HAS_CIEDE2000 = True
except ImportError:  # pragma: no cover - skimage always ships with deltaE_ciede2000
    _HAS_CIEDE2000 = False


@dataclass
class ShadeMatch:
    shade_id: str
    brand: str
    shade_name: str
    hex: str
    rgb: tuple
    lab: tuple
    delta_e: float
    product: str | None = None
    undertone: str | None = None
    depth: str | None = None
    source: str | None = None
    source_url: str | None = None
    rank: int = 0
    confidence: float | None = None
    confidence_breakdown: dict | None = None
    depth_penalty: float = 0.0
    ranking_score: float | None = None
    product_variants: list | None = None
    explanation: str | None = None


DEPTH_ORDER = {
    "fair": 0,
    "light": 1,
    "light-medium": 2,
    "medium": 3,
    "tan": 4,
    "deep": 5,
    "rich-deep": 6,
}
DEPTH_CLOSE_DELTA_E_WINDOW = 2.0
DEPTH_TIE_PENALTY = 0.35
VARIANT_DELTA_E_WINDOW = 1.5
VARIANT_HEX_RGB_DISTANCE = 10.0


def _normalize_key_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _rgb_distance(a: tuple, b: tuple) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def estimate_depth_from_lab_l(l_value: float) -> str:
    """Estimate broad skin depth from Lab L*."""
    if l_value >= 85:
        return "fair"
    if l_value >= 75:
        return "light"
    if l_value >= 65:
        return "light-medium"
    if l_value >= 55:
        return "medium"
    if l_value >= 45:
        return "tan"
    if l_value >= 32:
        return "deep"
    return "rich-deep"


def _depth_distance(estimated_depth: str, catalog_depth) -> int:
    if catalog_depth is None or pd.isna(catalog_depth):
        return 0
    return abs(DEPTH_ORDER.get(str(estimated_depth), 0) - DEPTH_ORDER.get(str(catalog_depth), DEPTH_ORDER.get(str(estimated_depth), 0)))


def _compute_delta_e(skin_lab: np.ndarray, catalog_lab: np.ndarray) -> np.ndarray:
    """Compute perceptual distance between one skin Lab color and each
    catalog Lab color. Uses CIEDE2000; falls back to Lab Euclidean (CIE76)
    only if CIEDE2000 is unavailable.

    Note: skimage's deltaE_ciede2000 does not correctly broadcast a single
    (3,) color against an (N, 3) array of colors, so the skin color is
    explicitly tiled to match shape before calling it.
    """
    n = len(catalog_lab)
    if _HAS_CIEDE2000:
        skin_tiled = np.tile(np.asarray(skin_lab, dtype=np.float64), (n, 1))
        return deltaE_ciede2000(skin_tiled, catalog_lab)
    # Fallback: CIE76 Euclidean distance in Lab.
    return np.linalg.norm(catalog_lab - np.asarray(skin_lab), axis=1)


def _row_to_match(row, idx, distances, ranking_scores, rank: int = 0) -> ShadeMatch:
    depth_penalty = float(ranking_scores[idx] - distances[idx])
    return ShadeMatch(
        shade_id=str(row["shade_id"]),
        brand=str(row["brand"]),
        shade_name=str(row["shade_name"]),
        hex=str(row["hex"]),
        rgb=(int(row["r"]), int(row["g"]), int(row["b"])),
        lab=(float(row["lab_l"]), float(row["lab_a"]), float(row["lab_b"])),
        delta_e=float(distances[idx]),
        product=row.get("product") if pd.notna(row.get("product")) else None,
        undertone=row.get("undertone") if pd.notna(row.get("undertone")) else None,
        depth=row.get("depth") if pd.notna(row.get("depth")) else None,
        source=row.get("source") if pd.notna(row.get("source")) else None,
        source_url=row.get("source_url") if pd.notna(row.get("source_url")) else None,
        depth_penalty=depth_penalty,
        ranking_score=float(ranking_scores[idx]),
        product_variants=[],
        rank=rank,
    )


def _is_same_shade_candidate(candidate: ShadeMatch, existing: ShadeMatch) -> bool:
    if _normalize_key_text(candidate.brand) != _normalize_key_text(existing.brand):
        return False
    if _normalize_key_text(candidate.shade_name) != _normalize_key_text(existing.shade_name):
        return False
    if candidate.hex == existing.hex:
        return True
    return (
        abs(candidate.delta_e - existing.delta_e) <= VARIANT_DELTA_E_WINDOW
        or _rgb_distance(candidate.rgb, existing.rgb) <= VARIANT_HEX_RGB_DISTANCE
    )


def _variant_payload(match: ShadeMatch) -> dict:
    return {
        "shade_id": match.shade_id,
        "brand": match.brand,
        "product": match.product,
        "shade_name": match.shade_name,
        "hex": match.hex,
        "delta_e": match.delta_e,
    }


def _add_variant(primary: ShadeMatch, variant: ShadeMatch) -> None:
    if primary.product_variants is None:
        primary.product_variants = []
    exact_key = (
        _normalize_key_text(variant.brand),
        _normalize_key_text(variant.product),
        _normalize_key_text(variant.shade_name),
        variant.hex,
    )
    primary_key = (
        _normalize_key_text(primary.brand),
        _normalize_key_text(primary.product),
        _normalize_key_text(primary.shade_name),
        primary.hex,
    )
    if exact_key == primary_key:
        return
    for existing in primary.product_variants:
        existing_key = (
            _normalize_key_text(existing.get("brand")),
            _normalize_key_text(existing.get("product")),
            _normalize_key_text(existing.get("shade_name")),
            existing.get("hex"),
        )
        if existing_key == exact_key:
            return
    primary.product_variants.append(_variant_payload(variant))


def match_shades(skin_lab, catalog_df: pd.DataFrame, top_k: int = 3) -> list:
    """Rank catalog shades by perceptual distance to the extracted skin color.

    Returns up to `top_k` `ShadeMatch` entries sorted by ascending distance
    (best match first). Returns fewer than `top_k` if the catalog has fewer
    rows; returns an empty list if the catalog is empty.
    """
    if catalog_df is None or len(catalog_df) == 0:
        return []

    catalog_lab = catalog_df[["lab_l", "lab_a", "lab_b"]].to_numpy(dtype=np.float64)
    distances = _compute_delta_e(skin_lab, catalog_lab)
    estimated_depth = estimate_depth_from_lab_l(float(np.asarray(skin_lab, dtype=np.float64)[0]))
    best_delta = float(np.min(distances))
    ranking_scores = distances.copy()
    if "depth" in catalog_df.columns:
        for idx, row in catalog_df.iterrows():
            if distances[idx] <= best_delta + DEPTH_CLOSE_DELTA_E_WINDOW:
                ranking_scores[idx] += DEPTH_TIE_PENALTY * _depth_distance(estimated_depth, row.get("depth"))

    order = np.lexsort((distances, ranking_scores))

    matches = []
    for idx in order:
        row = catalog_df.iloc[idx]
        candidate = _row_to_match(row, idx, distances, ranking_scores)
        grouped = False
        for existing in matches:
            if _is_same_shade_candidate(candidate, existing):
                _add_variant(existing, candidate)
                grouped = True
                break
        if grouped:
            continue
        candidate.rank = len(matches) + 1
        matches.append(candidate)
        if len(matches) >= top_k:
            break
    return matches
