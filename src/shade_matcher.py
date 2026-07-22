"""Shade matching via perceptual (CIEDE2000) color distance in Lab space."""

from dataclasses import dataclass

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

    order = np.lexsort((distances, ranking_scores))[:top_k]

    matches = []
    for rank, idx in enumerate(order, start=1):
        row = catalog_df.iloc[idx]
        depth_penalty = float(ranking_scores[idx] - distances[idx])
        matches.append(
            ShadeMatch(
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
                rank=rank,
            )
        )
    return matches
