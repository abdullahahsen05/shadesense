import json

import pandas as pd

from src.readiness_calibration import (
    calibrate_readiness_thresholds,
    load_readiness_thresholds,
)


def _records():
    rows = []
    for split, count in (("development", 60), ("locked_test", 20)):
        for index in range(count):
            usable = index % 2 == 0
            rows.append(
                {
                    "dataset": "mste",
                    "split": split,
                    "expected_capture_label": (
                        "usable" if usable else "recapture"
                    ),
                    "pipeline_success": True,
                    "lighting_low_signal": False,
                    "lighting_score": 0.8,
                    "readiness_score": 72 if usable else 56,
                    "extraction_quality_score": 75 if usable else 58,
                    "capture_uncertainty_delta_e_p90": 5 if usable else 11,
                    "lighting_sensitivity_delta_e_p90": 3 if usable else 7,
                }
            )
    return pd.DataFrame(rows)


def test_readiness_calibration_uses_development_and_reports_locked_test():
    thresholds, evidence = calibrate_readiness_thresholds(_records())

    assert thresholds.source == "MST-E development metadata calibration"
    assert evidence["development"]["usable_accept_rate"] == 1.0
    assert evidence["development"]["usable_ready_rate"] == 1.0
    assert evidence["development"]["false_usable_rate"] == 0.0
    assert evidence["development"]["false_ready_rate"] == 0.0
    assert evidence["locked_test"]["usable_accept_rate"] == 1.0
    assert evidence["locked_test"]["false_ready_rate"] == 0.0


def test_readiness_thresholds_load_from_saved_payload(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "source": "unit-test calibration",
                "thresholds": {
                    "caution_score": 61.0,
                    "caution_max_uncertainty": 8.5,
                },
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_readiness_thresholds(path)

    assert thresholds.caution_score == 61.0
    assert thresholds.caution_max_uncertainty == 8.5
    assert thresholds.source == "unit-test calibration"
