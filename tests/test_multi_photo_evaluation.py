import pandas as pd

from src.multi_photo_evaluation import (
    build_multi_photo_repeatability,
    summarize_multi_photo_repeatability,
)


def _row(benchmark_id, lab, reference=False):
    return {
        "benchmark_id": benchmark_id,
        "dataset": "mste",
        "pipeline_success": True,
        "subject_id": "subject-1",
        "split": "development",
        "mst": 5,
        "is_evaluation_reference": reference,
        "expected_capture_label": "usable" if reference else "challenging",
        "matching_lab_l": lab[0],
        "matching_lab_a": lab[1],
        "matching_lab_b": lab[2],
        "extraction_quality_score": 80,
        "capture_readiness_score": 80,
        "lighting_score": 0.8,
    }


def test_multi_photo_evaluation_holds_reference_out_of_inputs():
    records = pd.DataFrame(
        [
            _row("reference", (50, 10, 15), reference=True),
            _row("input-1", (49, 10, 15)),
            _row("input-2", (51, 10, 15)),
            _row("input-3", (75, -5, 2)),
        ]
    )

    result = build_multi_photo_repeatability(records)
    summary = summarize_multi_photo_repeatability(result)

    assert len(result) == 1
    assert "reference" not in result.iloc[0]["input_benchmark_ids"]
    assert result.iloc[0]["rejected_count"] == 1
    assert result.iloc[0]["consensus_to_reference_delta_e"] < 1.0
    assert summary["subject_count"] == 1
