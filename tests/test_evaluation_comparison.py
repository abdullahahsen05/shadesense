import pandas as pd

from src.evaluation_comparison import (
    build_paired_comparison,
    summarize_paired_comparison,
)


def _run(score, state, lab):
    return pd.DataFrame(
        [
            {
                "benchmark_id": "one",
                "pipeline_success": True,
                "extraction_quality_score": score,
                "capture_readiness_score": score,
                "lighting_score": 0.8,
                "capture_uncertainty_delta_e_p90": 4,
                "matching_lab_l": lab[0],
                "matching_lab_a": lab[1],
                "matching_lab_b": lab[2],
                "readiness_state": state,
            }
        ]
    )


def _recommendation(shade, product_type):
    return pd.DataFrame(
        [
            {
                "benchmark_id": "one",
                "rank": 1,
                "shade_id": shade,
                "product_type": product_type,
            }
        ]
    )


def test_paired_comparison_reports_changes_without_accuracy_claim():
    paired = build_paired_comparison(
        _run(60, "caution", (50, 10, 15)),
        _run(70, "ready", (51, 10, 15)),
        _recommendation("old", "concealer_hybrid"),
        _recommendation("new", "foundation"),
    )
    summary = summarize_paired_comparison(paired)

    assert paired.iloc[0]["extraction_quality_score_change"] == 10
    assert paired.iloc[0]["readiness_level_change"] == 1
    assert bool(paired.iloc[0]["top1_changed"])
    assert summary["readiness_upgrade_rate"] == 1.0
    assert "not automatically an accuracy gain" in summary["interpretation_note"]


def test_paired_comparison_keeps_failed_rows_with_blank_lab():
    baseline = _run(60, "provisional", (50, 10, 15))
    candidate = _run(0, "provisional", ("", "", ""))
    candidate.loc[0, "pipeline_success"] = False

    paired = build_paired_comparison(baseline, candidate)

    assert len(paired) == 1
    assert bool(paired.iloc[0]["success_changed"])
    assert pd.isna(paired.iloc[0]["matching_lab_change_delta_e"])
