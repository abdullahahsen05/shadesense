"""Stratified manual review support for facial-region masks."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIONS = ("forehead", "left_cheek", "right_cheek", "jawline")
REVIEW_LABELS = (
    "not_reviewed",
    "clean",
    "minor_contamination",
    "major_contamination",
    "insufficient_visible_skin",
)


def _boolean_column(
    frame: pd.DataFrame,
    name: str,
    default: bool = False,
) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[name]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.lower().isin(("true", "1", "yes"))


def _risk_score(frame: pd.DataFrame) -> pd.Series:
    uncertainty = pd.to_numeric(
        frame.get("capture_uncertainty_delta_e_p90", 0),
        errors="coerce",
    ).fillna(12.0)
    extraction = pd.to_numeric(
        frame.get("extraction_quality_score", 0),
        errors="coerce",
    ).fillna(0.0)
    pose = pd.to_numeric(
        frame.get("pose_asymmetry", 0),
        errors="coerce",
    ).fillna(0.0)
    score = (
        3.0 * (~_boolean_column(frame, "pipeline_success")).astype(float)
        + 1.2 * np.clip(uncertainty / 12.0, 0.0, 2.0)
        + 0.8 * np.clip((70.0 - extraction) / 70.0, 0.0, 1.0)
        + 0.7 * np.clip(pose / 0.5, 0.0, 1.5)
        + 0.6 * _boolean_column(frame, "eyewear_detected").astype(float)
        + 0.5 * _boolean_column(frame, "lighting_uneven").astype(float)
    )
    return score


def _round_robin(
    frame: pd.DataFrame,
    count: int,
    stratum: str,
    seed: int,
) -> pd.DataFrame:
    if count >= len(frame):
        return frame.copy()
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in frame.groupby(stratum, dropna=False, sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        groups.append(indices.tolist())
    chosen = []
    while len(chosen) < count and any(groups):
        remaining = []
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop())
            if group:
                remaining.append(group)
        groups = remaining
    return frame.loc[chosen].copy()


def build_mask_audit_manifest(
    records: pd.DataFrame,
    *,
    count: int = 100,
    mste_count: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Select both high-risk and representative rows for human mask review."""
    if count <= 0 or mste_count < 0 or mste_count > count:
        raise ValueError("Invalid mask-audit sample counts.")
    available = records.copy()
    available["mask_risk_score"] = _risk_score(available)
    selected_parts = []
    for dataset, quota, stratum in (
        ("mste", mste_count, "mst"),
        ("fairface", count - mste_count, "demographic_group"),
    ):
        group = available[available["dataset"] == dataset].copy()
        high_risk_count = quota // 2
        high_risk = group.nlargest(high_risk_count, "mask_risk_score")
        remaining = group.drop(high_risk.index)
        representative = _round_robin(
            remaining,
            quota - len(high_risk),
            stratum,
            seed + (0 if dataset == "mste" else 100),
        )
        selected_parts.extend([high_risk, representative])
    selected = pd.concat(selected_parts).drop_duplicates(
        "benchmark_id"
    ).head(count)
    if len(selected) != count:
        raise ValueError(
            f"Could only select {len(selected)} mask-audit rows, expected {count}."
        )

    columns = [
        "benchmark_id",
        "dataset",
        "archive_name",
        "archive_member",
        "subject_id",
        "split",
        "mst",
        "demographic_group",
        "lighting",
        "pose",
        "eyewear_detected",
        "pipeline_success",
        "extraction_quality_score",
        "capture_uncertainty_delta_e_p90",
        "mask_risk_score",
    ]
    audit = selected[columns].copy().reset_index(drop=True)
    audit.insert(0, "audit_index", np.arange(1, len(audit) + 1))
    audit["reviewed"] = False
    for region in REGIONS:
        audit[f"{region}_review"] = "not_reviewed"
    audit["review_notes"] = ""
    return audit


def summarize_mask_audit(audit: pd.DataFrame) -> dict:
    reviewed = audit[audit["reviewed"].astype(str).str.lower().isin(["true", "1"])]
    summary = {
        "selected_count": int(len(audit)),
        "reviewed_count": int(len(reviewed)),
        "review_completion": float(len(reviewed) / max(len(audit), 1)),
        "regions": {},
    }
    for region in REGIONS:
        column = f"{region}_review"
        counts = (
            reviewed[column].value_counts().to_dict()
            if column in reviewed and len(reviewed)
            else {}
        )
        major = int(counts.get("major_contamination", 0))
        minor = int(counts.get("minor_contamination", 0))
        denominator = max(len(reviewed), 1)
        summary["regions"][region] = {
            "counts": {str(key): int(value) for key, value in counts.items()},
            "minor_or_major_rate": float((minor + major) / denominator),
            "major_rate": float(major / denominator),
        }
    return summary
