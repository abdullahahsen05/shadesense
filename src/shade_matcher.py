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

from src.depth_diagnostics import (
    DEPTH_ORDER,
    depth_match_status,
    depth_sanity_note,
    estimate_depth_from_lab_l,
)


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
    extracted_depth: str | None = None
    depth_match_status: str = "unknown"
    depth_sanity_note: str | None = None
    product_type: str = "other_base"
    catalog_quality_score: float = 0.5
    recommendation_stability: float | None = None
    top3_stability: float | None = None
    delta_e_p90: float | None = None


DEPTH_CLOSE_DELTA_E_WINDOW = 2.0
DEPTH_TIE_PENALTY = 0.35
TOO_LIGHT_L_THRESHOLD = 4.0
TOO_LIGHT_CLOSE_PENALTY = 0.08
TOO_LIGHT_MAX_PENALTY = 0.55
VARIANT_DELTA_E_WINDOW = 1.5
VARIANT_HEX_RGB_DISTANCE = 10.0
NEAR_DUPLICATE_DELTA_E = 1.2
DISPLAY_MIN_DELTA_E_STEPS = (1.8, 1.2, 0.6, 0.0)
# Candidates are already sorted by their dominant CIEDE2000 score before
# product/variant grouping. A bounded shortlist keeps deduplication responsive
# for large public catalogs while leaving ample room to find three distinct
# recommendations.
MIN_DEDUP_SCAN_ROWS = 300
DEDUP_SCAN_MULTIPLIER = 100
CATALOG_QUALITY_CLOSE_PENALTY = 0.20
UNCERTAINTY_LIGHT_MARGIN = 2.0
STABILITY_SHORTLIST_SIZE = 100


def _normalize_key_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _rgb_distance(a: tuple, b: tuple) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def _lab_delta_e(a: tuple, b: tuple) -> float:
    return float(_compute_delta_e(np.array(a, dtype=np.float64), np.array([b], dtype=np.float64))[0])


def _shade_names_similar(a, b) -> bool:
    a_norm = _normalize_key_text(a)
    b_norm = _normalize_key_text(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True

    if len(a_norm) >= 4 and len(b_norm) >= 4 and (a_norm in b_norm or b_norm in a_norm):
        return True

    a_digits = set(re.findall(r"\d+", a_norm))
    b_digits = set(re.findall(r"\d+", b_norm))
    if a_digits and b_digits and a_digits & b_digits:
        return True

    return False


def _depth_distance(estimated_depth: str, catalog_depth) -> int:
    if catalog_depth is None or pd.isna(catalog_depth):
        return 0
    estimated_order = DEPTH_ORDER.get(str(estimated_depth))
    catalog_order = DEPTH_ORDER.get(str(catalog_depth))
    if estimated_order is None or catalog_order is None:
        return 0
    return abs(estimated_order - catalog_order)


def _too_light_penalty(
    skin_l: float,
    shade_l: float,
    supported_upper_l: float | None = None,
) -> float:
    if supported_upper_l is None:
        l_gap = float(shade_l - skin_l)
    else:
        l_gap = float(shade_l - (supported_upper_l + UNCERTAINTY_LIGHT_MARGIN))
    if l_gap <= TOO_LIGHT_L_THRESHOLD:
        return 0.0
    return float(np.clip((l_gap - TOO_LIGHT_L_THRESHOLD) * TOO_LIGHT_CLOSE_PENALTY, 0.0, TOO_LIGHT_MAX_PENALTY))


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


def _row_to_match(row, idx, distances, ranking_scores, extracted_depth: str, rank: int = 0) -> ShadeMatch:
    depth_penalty = float(ranking_scores[idx] - distances[idx])
    shade_depth = row.get("depth") if pd.notna(row.get("depth")) else None
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
        depth=shade_depth,
        source=row.get("source") if pd.notna(row.get("source")) else None,
        source_url=row.get("source_url") if pd.notna(row.get("source_url")) else None,
        depth_penalty=depth_penalty,
        ranking_score=float(ranking_scores[idx]),
        product_variants=[],
        extracted_depth=extracted_depth,
        depth_match_status=depth_match_status(extracted_depth, shade_depth),
        depth_sanity_note=depth_sanity_note(extracted_depth, shade_depth),
        product_type=str(row.get("product_type") or "other_base"),
        catalog_quality_score=float(row.get("catalog_quality_score", 0.5)),
        rank=rank,
    )


def _is_same_shade_candidate(candidate: ShadeMatch, existing: ShadeMatch) -> bool:
    if _normalize_key_text(candidate.brand) != _normalize_key_text(existing.brand):
        return False
    candidate_product = _normalize_key_text(candidate.product)
    existing_product = _normalize_key_text(existing.product)
    same_product = bool(candidate_product and existing_product and candidate_product == existing_product)
    similar_shade_name = _shade_names_similar(candidate.shade_name, existing.shade_name)

    if same_product and similar_shade_name and candidate.hex == existing.hex:
        return True
    if same_product and _lab_delta_e(candidate.lab, existing.lab) <= NEAR_DUPLICATE_DELTA_E:
        return True
    if same_product and similar_shade_name and _rgb_distance(candidate.rgb, existing.rgb) <= VARIANT_HEX_RGB_DISTANCE:
        return True
    if not similar_shade_name:
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


def _group_ranked_candidates(candidates: list[ShadeMatch]) -> list[ShadeMatch]:
    grouped_matches = []
    for candidate in candidates:
        grouped = False
        for existing in grouped_matches:
            if _is_same_shade_candidate(candidate, existing):
                _add_variant(existing, candidate)
                grouped = True
                break
        if not grouped:
            grouped_matches.append(candidate)
    return grouped_matches


def _select_visually_distinct_matches(candidates: list[ShadeMatch], top_k: int) -> list[ShadeMatch]:
    if top_k <= 0:
        return []
    if len(candidates) <= top_k:
        selected = candidates[:top_k]
    else:
        selected = []
        for threshold in DISPLAY_MIN_DELTA_E_STEPS:
            selected = []
            for candidate in candidates:
                if all(_lab_delta_e(candidate.lab, existing.lab) >= threshold for existing in selected):
                    selected.append(candidate)
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= top_k:
                    break

    for rank, match in enumerate(selected, start=1):
        match.rank = rank
    return selected


def _apply_uncertainty_stability(
    candidates: list[ShadeMatch],
    uncertainty_labs,
) -> None:
    if uncertainty_labs is None or not candidates:
        return
    samples = np.asarray(uncertainty_labs, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) == 0:
        return
    shortlist = candidates[:STABILITY_SHORTLIST_SIZE]
    labs = np.asarray([candidate.lab for candidate in shortlist], dtype=np.float64)
    sample_grid = np.repeat(samples[:, None, :], len(shortlist), axis=1)
    catalog_grid = np.repeat(labs[None, :, :], len(samples), axis=0)
    distances = deltaE_ciede2000(sample_grid, catalog_grid)
    order = np.argsort(distances, axis=1)
    for idx, candidate in enumerate(shortlist):
        candidate.recommendation_stability = float(np.mean(order[:, 0] == idx))
        candidate.top3_stability = float(np.mean(np.any(order[:, : min(3, len(shortlist))] == idx, axis=1)))
        candidate.delta_e_p90 = float(np.percentile(distances[:, idx], 90))


def match_shades(
    skin_lab,
    catalog_df: pd.DataFrame,
    top_k: int = 3,
    uncertainty_labs=None,
) -> list:
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
    supported_upper_l = None
    if uncertainty_labs is not None:
        uncertainty_array = np.asarray(uncertainty_labs, dtype=np.float64)
        if uncertainty_array.ndim == 2 and uncertainty_array.shape[1] == 3 and len(uncertainty_array):
            supported_upper_l = float(np.percentile(uncertainty_array[:, 0], 95))
    if "depth" in catalog_df.columns:
        for idx, row in catalog_df.iterrows():
            if distances[idx] <= best_delta + DEPTH_CLOSE_DELTA_E_WINDOW:
                ranking_scores[idx] += DEPTH_TIE_PENALTY * _depth_distance(estimated_depth, row.get("depth"))
                ranking_scores[idx] += _too_light_penalty(
                    float(np.asarray(skin_lab, dtype=np.float64)[0]),
                    float(row.get("lab_l")),
                    supported_upper_l=supported_upper_l,
                )
                quality = float(row.get("catalog_quality_score", 0.5))
                ranking_scores[idx] += CATALOG_QUALITY_CLOSE_PENALTY * (
                    1.0 - float(np.clip(quality, 0.0, 1.0))
                )

    order = np.lexsort((distances, ranking_scores))

    dedup_scan_limit = min(len(order), max(MIN_DEDUP_SCAN_ROWS, top_k * DEDUP_SCAN_MULTIPLIER))
    ranked_candidates = []
    for idx in order[:dedup_scan_limit]:
        row = catalog_df.iloc[idx]
        candidate = _row_to_match(row, idx, distances, ranking_scores, estimated_depth)
        ranked_candidates.append(candidate)

    grouped_candidates = _group_ranked_candidates(ranked_candidates)
    _apply_uncertainty_stability(grouped_candidates, uncertainty_labs)
    return _select_visually_distinct_matches(grouped_candidates, top_k)
