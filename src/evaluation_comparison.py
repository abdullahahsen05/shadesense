"""Paired comparison of two frozen-manifest evaluation runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000


READINESS_ORDER = {"provisional": 0, "caution": 1, "ready": 2}


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin(("true", "1", "yes"))


def build_paired_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_recommendations: pd.DataFrame | None = None,
    candidate_recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"benchmark_id", "pipeline_success"}
    if not required.issubset(baseline) or not required.issubset(candidate):
        raise ValueError("Both runs need benchmark_id and pipeline_success.")
    left = baseline.add_suffix("_baseline").rename(
        columns={"benchmark_id_baseline": "benchmark_id"}
    )
    right = candidate.add_suffix("_candidate").rename(
        columns={"benchmark_id_candidate": "benchmark_id"}
    )
    paired = left.merge(right, on="benchmark_id", how="inner", validate="1:1")
    paired["success_changed"] = (
        _as_bool(paired["pipeline_success_baseline"])
        != _as_bool(paired["pipeline_success_candidate"])
    )
    for column in (
        "extraction_quality_score",
        "capture_readiness_score",
        "lighting_score",
        "capture_uncertainty_delta_e_p90",
    ):
        baseline_column = f"{column}_baseline"
        candidate_column = f"{column}_candidate"
        if baseline_column in paired and candidate_column in paired:
            paired[f"{column}_change"] = (
                pd.to_numeric(paired[candidate_column], errors="coerce")
                - pd.to_numeric(paired[baseline_column], errors="coerce")
            )
    lab_columns = [
        "matching_lab_l",
        "matching_lab_a",
        "matching_lab_b",
    ]
    if all(
        f"{column}_{suffix}" in paired
        for column in lab_columns
        for suffix in ("baseline", "candidate")
    ):
        left_labs = paired[
            [f"{column}_baseline" for column in lab_columns]
        ].to_numpy(dtype=np.float64)
        right_labs = paired[
            [f"{column}_candidate" for column in lab_columns]
        ].to_numpy(dtype=np.float64)
        paired["matching_lab_change_delta_e"] = deltaE_ciede2000(
            left_labs,
            right_labs,
        )
    if "readiness_state_baseline" in paired and "readiness_state_candidate" in paired:
        baseline_level = (
            paired["readiness_state_baseline"].map(READINESS_ORDER).fillna(-1)
        )
        candidate_level = (
            paired["readiness_state_candidate"].map(READINESS_ORDER).fillna(-1)
        )
        paired["readiness_level_change"] = candidate_level - baseline_level

    def top_one(frame):
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["benchmark_id", "shade_id", "product_type"])
        return (
            frame[pd.to_numeric(frame["rank"], errors="coerce") == 1]
            [["benchmark_id", "shade_id", "product_type"]]
            .drop_duplicates("benchmark_id")
        )

    baseline_top = top_one(baseline_recommendations).rename(
        columns={
            "shade_id": "top1_shade_baseline",
            "product_type": "top1_product_type_baseline",
        }
    )
    candidate_top = top_one(candidate_recommendations).rename(
        columns={
            "shade_id": "top1_shade_candidate",
            "product_type": "top1_product_type_candidate",
        }
    )
    paired = paired.merge(baseline_top, on="benchmark_id", how="left")
    paired = paired.merge(candidate_top, on="benchmark_id", how="left")
    if "top1_shade_baseline" in paired and "top1_shade_candidate" in paired:
        paired["top1_changed"] = (
            paired["top1_shade_baseline"].fillna("")
            != paired["top1_shade_candidate"].fillna("")
        )
    return paired


def summarize_paired_comparison(paired: pd.DataFrame) -> dict:
    def numeric_summary(column: str) -> dict | None:
        if column not in paired:
            return None
        values = pd.to_numeric(paired[column], errors="coerce").dropna()
        if values.empty:
            return None
        return {
            "median": float(values.median()),
            "mean": float(values.mean()),
            "p90_absolute": float(values.abs().quantile(0.90)),
        }

    return {
        "paired_image_count": int(len(paired)),
        "pipeline_success_change_count": int(
            paired.get("success_changed", pd.Series(dtype=bool)).sum()
        ),
        "top1_change_rate": (
            float(paired["top1_changed"].mean())
            if "top1_changed" in paired
            else None
        ),
        "readiness_upgrade_rate": (
            float((paired["readiness_level_change"] > 0).mean())
            if "readiness_level_change" in paired
            else None
        ),
        "readiness_downgrade_rate": (
            float((paired["readiness_level_change"] < 0).mean())
            if "readiness_level_change" in paired
            else None
        ),
        "extraction_quality_change": numeric_summary(
            "extraction_quality_score_change"
        ),
        "capture_readiness_change": numeric_summary(
            "capture_readiness_score_change"
        ),
        "matching_lab_change_delta_e": numeric_summary(
            "matching_lab_change_delta_e"
        ),
        "interpretation_note": (
            "A changed recommendation is not automatically an accuracy gain. "
            "Physical wearer-to-product labels are unavailable; paired results "
            "support robustness and catalog-scope claims only."
        ),
    }
