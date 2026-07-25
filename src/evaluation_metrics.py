"""Serialization and aggregate metrics for ShadeSense benchmark runs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _finite(value, default=np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def analysis_record(
    manifest_row: pd.Series,
    analysis,
    *,
    elapsed_seconds: float,
    error: str | None = None,
) -> dict:
    """Flatten one shared-pipeline result into a stable CSV record."""
    record = {
        key: manifest_row.get(key, "")
        for key in manifest_row.index
    }
    record.update(
        {
            "pipeline_success": bool(analysis is not None and analysis.success),
            "error": error or getattr(analysis, "error", None) or "",
            "elapsed_seconds": float(elapsed_seconds),
            "source_image_height": (
                int(analysis.source_shape[0]) if analysis is not None else 0
            ),
            "source_image_width": (
                int(analysis.source_shape[1]) if analysis is not None else 0
            ),
            "analysis_image_height": (
                int(analysis.image_rgb.shape[0]) if analysis is not None else 0
            ),
            "analysis_image_width": (
                int(analysis.image_rgb.shape[1]) if analysis is not None else 0
            ),
            "analysis_scale": (
                float(analysis.analysis_scale) if analysis is not None else 0.0
            ),
        }
    )
    if analysis is None:
        return record

    face = analysis.face_result
    color_metadata = analysis.image_color_metadata or {}
    record.update(
        {
            "face_detected": bool(face.success),
            "face_count": int(face.face_count),
            "face_warnings": _json(face.warnings),
            "embedded_icc_present": bool(
                color_metadata.get("embedded_icc_present", False)
            ),
            "icc_converted_to_srgb": bool(
                color_metadata.get("icc_converted_to_srgb", False)
            ),
            "assumed_srgb": bool(
                color_metadata.get("assumed_srgb", True)
            ),
            "source_profile_description": color_metadata.get(
                "source_profile_description",
                "",
            ),
            "color_profile_warnings": _json(
                color_metadata.get("warnings", [])
            ),
        }
    )
    if not face.success:
        return record

    image_quality = analysis.image_quality
    lighting = analysis.lighting_quality
    skin = analysis.skin_result
    extraction = analysis.extraction_quality_report or {}
    readiness = analysis.recommendation_readiness
    mask_diag = analysis.mask_capture_diagnostics or {}
    local_uncertainty = skin.uncertainty_diagnostics or {}
    capture_uncertainty = skin.systematic_uncertainty_diagnostics or {}
    sensitivity = skin.lighting_sensitivity_diagnostics or {}
    stability = skin.stability_diagnostics or {}

    record.update(
        {
            "image_quality_score": _finite(image_quality.overall_score),
            "image_quality_label": image_quality.label,
            "blur_score": _finite(image_quality.blur_score),
            "exposure_score": _finite(image_quality.exposure_score),
            "face_size_score": _finite(image_quality.face_size_score),
            "pose_score": _finite(image_quality.pose_score),
            "color_cast_score": _finite(image_quality.color_cast_score),
            "pose_asymmetry": _finite(image_quality.pose_asymmetry),
            "lighting_score": _finite(lighting.score),
            "lighting_low_signal": bool(lighting.low_signal),
            "lighting_recapture_recommended": bool(
                lighting.recapture_recommended
            ),
            "lighting_underexposed": bool(lighting.underexposed),
            "lighting_overexposed": bool(lighting.overexposed),
            "lighting_uneven": bool(lighting.uneven_lighting),
            "lighting_color_cast": bool(lighting.color_cast),
            "face_median_luma": _finite(lighting.face_median_luma),
            "face_black_clip_ratio": _finite(
                getattr(lighting, "face_black_clip_ratio", 0.0)
            ),
            "left_right_gap": _finite(lighting.left_right_gap),
            "central_lower_gap": _finite(lighting.central_lower_gap),
            "eyewear_detected": bool(
                mask_diag.get("eyewear_reflection_detected", False)
            ),
            "extraction_source": analysis.extraction_selection.selected_source,
            "raw_region_extraction_score": _finite(skin.quality_score * 100.0),
            "extraction_quality_score": _finite(
                extraction.get("overall_score")
            ),
            "extraction_quality_label": extraction.get("label", ""),
            "skin_lab_l": _finite(skin.lab[0]),
            "skin_lab_a": _finite(skin.lab[1]),
            "skin_lab_b": _finite(skin.lab[2]),
            "skin_rgb": _json(list(skin.rgb)),
            "matching_lab_l": _finite(
                (
                    skin.foundation_target_lab
                    if skin.foundation_target_active
                    else skin.lab
                )[0]
            ),
            "matching_lab_a": _finite(
                (
                    skin.foundation_target_lab
                    if skin.foundation_target_active
                    else skin.lab
                )[1]
            ),
            "matching_lab_b": _finite(
                (
                    skin.foundation_target_lab
                    if skin.foundation_target_active
                    else skin.lab
                )[2]
            ),
            "foundation_target_active": bool(skin.foundation_target_active),
            "depth_estimate": skin.depth_estimate or "",
            "region_consistency": _finite(skin.region_consistency),
            "region_stability_score": _finite(
                stability.get("stability_score")
            ),
            "included_regions": _json(skin.included_region_names),
            "excluded_regions": _json(skin.excluded_region_names),
            "local_uncertainty_delta_e_p90": _finite(
                local_uncertainty.get("delta_e_radius_p90")
            ),
            "capture_uncertainty_delta_e_p90": _finite(
                capture_uncertainty.get("total_delta_e_radius_p90")
            ),
            "lighting_sensitivity_score": _finite(sensitivity.get("score")),
            "lighting_sensitivity_delta_e_p90": _finite(
                sensitivity.get("delta_e_p90")
            ),
            "analysis_warnings": _json(
                list(
                    dict.fromkeys(
                        list(getattr(lighting, "warnings", []))
                        + list(getattr(skin, "warnings", []))
                        + list(extraction.get("warnings", []))
                    )
                )
            ),
        }
    )
    for region_name in ("forehead", "left_cheek", "right_cheek", "jawline"):
        region = skin.region_results.get(region_name)
        prefix = f"region_{region_name}"
        record[f"{prefix}_mask_pixels"] = int(
            np.sum(analysis.masks.get(region_name, 0) > 0)
        )
        record[f"{prefix}_included"] = bool(
            region is not None and region_name in skin.included_region_names
        )
        record[f"{prefix}_quality_score"] = (
            _finite(region.quality_score) if region is not None else np.nan
        )
        record[f"{prefix}_valid_ratio"] = (
            _finite(region.valid_ratio) if region is not None else np.nan
        )
        record[f"{prefix}_weight"] = (
            _finite(region.weight_multiplier) if region is not None else np.nan
        )

    if readiness is not None:
        record.update(
            {
                "readiness_state": readiness.state,
                "readiness_score": _finite(readiness.score),
                "capture_readiness_score": _finite(
                    readiness.capture_readiness_score
                ),
                "shade_family_stability_score": _finite(
                    readiness.shade_family_stability_score
                ),
                "exact_product_stability_score": _finite(
                    readiness.exact_product_stability_score
                ),
                "confidence_cap": _finite(readiness.confidence_cap),
            }
        )
    return record


def recommendation_records(manifest_row: pd.Series, analysis) -> list[dict]:
    rows = []
    if analysis is None:
        return rows
    for match in analysis.matches:
        rows.append(
            {
                "benchmark_id": manifest_row["benchmark_id"],
                "dataset": manifest_row["dataset"],
                "split": manifest_row["split"],
                "rank": int(match.rank),
                "shade_id": match.shade_id,
                "brand": match.brand,
                "shade_name": match.shade_name,
                "product": match.product or "",
                "product_type": match.product_type,
                "hex": match.hex,
                "delta_e": _finite(match.delta_e),
                "distribution_delta_e": _finite(
                    match.distribution_delta_e
                ),
                "confidence": _finite(match.confidence),
                "catalog_quality_score": _finite(
                    match.catalog_quality_score
                ),
                "top1_stability": _finite(
                    match.recommendation_stability
                ),
                "top3_stability": _finite(match.top3_stability),
                "top1_family_stability": _finite(
                    match.recommendation_family_stability
                ),
                "top3_family_stability": _finite(
                    match.top3_family_stability
                ),
                "lighting_top1_stability": _finite(
                    match.lighting_recommendation_stability
                ),
                "lighting_top3_stability": _finite(
                    match.lighting_top3_stability
                ),
            }
        )
    return rows


def _summary(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
    }


def build_repeatability_metrics(records: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare MST-E captures with each subject's usable reference image."""
    successful = records[
        (records["dataset"] == "mste")
        & records.get("pipeline_success", False).astype(bool)
    ].copy()
    rows = []
    for subject_id, group in successful.groupby("subject_id"):
        references = group[
            group["is_evaluation_reference"].astype(str).str.lower()
            .isin(["true", "1"])
        ]
        if references.empty:
            continue
        reference = references.iloc[0]
        reference_lab = np.asarray(
            [
                reference["matching_lab_l"],
                reference["matching_lab_a"],
                reference["matching_lab_b"],
            ],
            dtype=np.float64,
        )
        labs = group[
            ["matching_lab_l", "matching_lab_a", "matching_lab_b"]
        ].to_numpy(dtype=np.float64)
        distances = deltaE_ciede2000(
            labs,
            np.repeat(reference_lab[None, :], len(labs), axis=0),
        )
        for (_, item), distance in zip(group.iterrows(), distances):
            rows.append(
                {
                    "benchmark_id": item["benchmark_id"],
                    "subject_id": subject_id,
                    "mst": item["mst"],
                    "lighting": item["lighting"],
                    "pose": item["pose"],
                    "mask_present": item["mask_present"],
                    "reference_benchmark_id": reference["benchmark_id"],
                    "delta_e_to_reference": float(distance),
                }
            )
    frame = pd.DataFrame(rows)
    metrics = {
        "subjects_with_evaluation_reference": int(
            frame["subject_id"].nunique() if not frame.empty else 0
        ),
        "delta_e_to_reference": (
            _summary(frame["delta_e_to_reference"])
            if not frame.empty
            else _summary(pd.Series(dtype=float))
        ),
    }
    return metrics, frame


def build_subgroup_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = [
        "dataset",
        "mst",
        "demographic_group",
        "lighting",
        "pose",
        "mask_present",
        "eyewear_detected",
    ]
    for dimension in dimensions:
        if dimension not in records:
            continue
        valid = records[
            records[dimension].notna()
            & (records[dimension].astype(str) != "")
        ]
        for value, group in valid.groupby(dimension, dropna=False):
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "count": int(len(group)),
                    "face_detection_rate": float(
                        group["face_detected"].fillna(False).mean()
                    ),
                    "pipeline_success_rate": float(
                        group["pipeline_success"].fillna(False).mean()
                    ),
                    "median_extraction_quality": _summary(
                        group.get(
                            "extraction_quality_score",
                            pd.Series(dtype=float),
                        )
                    )["median"],
                    "median_capture_uncertainty": _summary(
                        group.get(
                            "capture_uncertainty_delta_e_p90",
                            pd.Series(dtype=float),
                        )
                    )["median"],
                    "provisional_rate": float(
                        (
                            group.get("readiness_state", "")
                            == "provisional"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_aggregate_metrics(
    records: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    total = len(records)
    repeatability, repeatability_frame = build_repeatability_metrics(records)
    subgroup_frame = build_subgroup_metrics(records)
    clear = records[
        records["expected_capture_label"].isin(["usable", "recapture"])
    ].copy()
    usable = clear[clear["expected_capture_label"] == "usable"]
    recapture = clear[clear["expected_capture_label"] == "recapture"]

    metrics = {
        "image_count": int(total),
        "face_detection_rate": float(
            records["face_detected"].fillna(False).mean()
        ),
        "pipeline_success_rate": float(
            records["pipeline_success"].fillna(False).mean()
        ),
        "processing_seconds": _summary(records["elapsed_seconds"]),
        "extraction_quality": _summary(
            records.get("extraction_quality_score", pd.Series(dtype=float))
        ),
        "capture_uncertainty_delta_e": _summary(
            records.get(
                "capture_uncertainty_delta_e_p90",
                pd.Series(dtype=float),
            )
        ),
        "readiness_distribution": {
            str(key): int(value)
            for key, value in records.get(
                "readiness_state",
                pd.Series(dtype=str),
            )
            .fillna("unavailable")
            .value_counts()
            .items()
        },
        "metadata_label_readiness": {
            "clear_label_count": int(len(clear)),
            "usable_accept_rate": (
                float((usable["readiness_state"] != "provisional").mean())
                if len(usable)
                else None
            ),
            "recapture_reject_rate": (
                float((recapture["readiness_state"] == "provisional").mean())
                if len(recapture)
                else None
            ),
            "false_ready_rate": (
                float((recapture["readiness_state"] != "provisional").mean())
                if len(recapture)
                else None
            ),
        },
        "top_recommendation_product_types": (
            {
                str(key): int(value)
                for key, value in recommendations[
                    recommendations["rank"] == 1
                ]["product_type"]
                .value_counts()
                .items()
            }
            if not recommendations.empty
            else {}
        ),
        "repeatability": repeatability,
    }
    return metrics, subgroup_frame, repeatability_frame
