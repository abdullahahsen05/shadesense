import pandas as pd

from src.evaluation_metrics import build_aggregate_metrics


def test_false_ready_is_separate_from_caution_false_usable():
    records = pd.DataFrame(
        [
            {
                "benchmark_id": "usable",
                "dataset": "mste",
                "expected_capture_label": "usable",
                "readiness_state": "ready",
                "face_detected": True,
                "pipeline_success": True,
                "elapsed_seconds": 1,
                "subject_id": "one",
                "is_evaluation_reference": False,
                "mst": 5,
                "demographic_group": "",
                "lighting": "well_lit",
                "pose": "frontal",
                "mask_present": 0,
                "eyewear_detected": False,
            },
            {
                "benchmark_id": "recapture-caution",
                "dataset": "mste",
                "expected_capture_label": "recapture",
                "readiness_state": "caution",
                "face_detected": True,
                "pipeline_success": True,
                "elapsed_seconds": 1,
                "subject_id": "two",
                "is_evaluation_reference": False,
                "mst": 5,
                "demographic_group": "",
                "lighting": "poorly_lit",
                "pose": "side",
                "mask_present": 0,
                "eyewear_detected": False,
            },
            {
                "benchmark_id": "recapture-ready",
                "dataset": "mste",
                "expected_capture_label": "recapture",
                "readiness_state": "ready",
                "face_detected": True,
                "pipeline_success": True,
                "elapsed_seconds": 1,
                "subject_id": "three",
                "is_evaluation_reference": False,
                "mst": 5,
                "demographic_group": "",
                "lighting": "poorly_lit",
                "pose": "side",
                "mask_present": 0,
                "eyewear_detected": False,
            },
        ]
    )
    recommendations = pd.DataFrame(
        columns=["benchmark_id", "rank", "product_type"]
    )

    metrics, _, _ = build_aggregate_metrics(records, recommendations)
    readiness = metrics["metadata_label_readiness"]

    assert readiness["usable_accept_rate"] == 1.0
    assert readiness["usable_ready_rate"] == 1.0
    assert readiness["false_usable_rate"] == 1.0
    assert readiness["false_ready_rate"] == 0.5
    assert readiness["recapture_reject_rate"] == 0.0
