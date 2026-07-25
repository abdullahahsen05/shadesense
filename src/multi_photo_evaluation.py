"""Evaluation metrics for multi-photo consensus using MST-E identities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000

from src.multicapture_consensus import (
    CaptureEvidence,
    build_multicapture_consensus,
)


LAB_COLUMNS = ["matching_lab_l", "matching_lab_a", "matching_lab_b"]


def _select_diverse_inputs(candidates: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    label_order = ["recapture", "challenging", "review_required", "usable"]
    groups = {
        label: candidates[
            candidates["expected_capture_label"] == label
        ].sort_values("benchmark_id", kind="stable")
        for label in label_order
    }
    chosen = []
    while len(chosen) < count:
        added = False
        for label in label_order:
            group = groups[label]
            if not group.empty and len(chosen) < count:
                chosen.append(group.iloc[0])
                groups[label] = group.iloc[1:]
                added = True
        if not added:
            break
    return pd.DataFrame(chosen).reset_index(drop=True)


def build_multi_photo_repeatability(records: pd.DataFrame) -> pd.DataFrame:
    """Compare non-reference photo consensus with each held-out MST-E reference."""
    successful = records[
        (records["dataset"] == "mste")
        & records["pipeline_success"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    rows = []
    for subject_id, group in successful.groupby("subject_id", sort=True):
        reference_rows = group[
            group["is_evaluation_reference"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        ]
        candidates = group.drop(reference_rows.index)
        if reference_rows.empty or len(candidates) < 2:
            continue
        reference = reference_rows.sort_values("benchmark_id").iloc[0]
        # Use up to three independent inputs and round-robin across capture
        # labels so one condition cannot dominate merely by having more files.
        inputs = _select_diverse_inputs(candidates, count=3)
        labs = inputs[LAB_COLUMNS].to_numpy(dtype=np.float64)
        extraction = pd.to_numeric(
            inputs["extraction_quality_score"],
            errors="coerce",
        ).fillna(0.0)
        readiness = pd.to_numeric(
            inputs["capture_readiness_score"],
            errors="coerce",
        ).fillna(extraction)
        lighting = (
            pd.to_numeric(inputs["lighting_score"], errors="coerce")
            .fillna(0.0)
            * 100.0
        )
        captures = []
        for index in range(len(inputs)):
            uncertainty = pd.to_numeric(
                inputs.iloc[index].get(
                    "capture_uncertainty_delta_e_p90",
                    6.0,
                ),
                errors="coerce",
            )
            if not np.isfinite(uncertainty):
                uncertainty = 6.0
            captures.append(
                CaptureEvidence(
                    capture_id=str(index),
                    lab=tuple(float(value) for value in labs[index]),
                    extraction_score=float(extraction.iloc[index]),
                    lighting_score=float(lighting.iloc[index] / 100.0),
                    uncertainty_radius=float(uncertainty),
                    low_signal=str(
                        inputs.iloc[index].get("lighting_low_signal", False)
                    ).lower()
                    in ("true", "1"),
                )
            )
        consensus = build_multicapture_consensus(captures)
        if not consensus.success:
            continue
        reference_lab = reference[LAB_COLUMNS].to_numpy(dtype=np.float64)
        consensus_distance = float(
            deltaE_ciede2000(
                np.asarray(consensus.lab).reshape(1, 3),
                reference_lab.reshape(1, 3),
            )[0]
        )
        individual_distances = deltaE_ciede2000(
            labs,
            np.repeat(reference_lab.reshape(1, 3), len(labs), axis=0),
        )
        rows.append(
            {
                "subject_id": subject_id,
                "split": reference["split"],
                "mst": reference["mst"],
                "reference_benchmark_id": reference["benchmark_id"],
                "input_benchmark_ids": "|".join(
                    inputs["benchmark_id"].astype(str)
                ),
                "input_capture_labels": "|".join(
                    inputs["expected_capture_label"].astype(str)
                ),
                "input_count": int(len(inputs)),
                "retained_count": int(len(consensus.included_capture_ids)),
                "rejected_count": int(len(consensus.excluded_capture_ids)),
                "agreement_delta_e_p90": consensus.uncertainty_radius_p90,
                "consensus_to_reference_delta_e": consensus_distance,
                "individual_to_reference_median_delta_e": float(
                    np.median(individual_distances)
                ),
                "best_individual_to_reference_delta_e": float(
                    np.min(individual_distances)
                ),
                "improvement_vs_individual_median": float(
                    np.median(individual_distances) - consensus_distance
                ),
                "consensus_better_than_individual_median": bool(
                    consensus_distance < np.median(individual_distances)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_multi_photo_repeatability(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"subject_count": 0}
    return {
        "subject_count": int(len(frame)),
        "median_consensus_to_reference_delta_e": float(
            frame["consensus_to_reference_delta_e"].median()
        ),
        "p90_consensus_to_reference_delta_e": float(
            frame["consensus_to_reference_delta_e"].quantile(0.90)
        ),
        "median_individual_to_reference_delta_e": float(
            frame["individual_to_reference_median_delta_e"].median()
        ),
        "median_improvement_delta_e": float(
            frame["improvement_vs_individual_median"].median()
        ),
        "subjects_improved_rate": float(
            frame["consensus_better_than_individual_median"].mean()
        ),
        "outlier_capture_rejection_rate": float(
            (frame["rejected_count"] > 0).mean()
        ),
        "method_note": (
            "Consensus inputs exclude each subject's designated usable "
            "reference; the reference is used only for repeatability scoring."
        ),
    }
