import pandas as pd

from src.evaluation_report import create_charts


def test_repeatability_boxplot_uses_current_matplotlib_api(tmp_path):
    records = pd.DataFrame(
        [
            {
                "dataset": "mste",
                "mst": 5,
                "face_detected": True,
                "pipeline_success": True,
                "readiness_state": "ready",
            }
        ]
    )
    repeatability = pd.DataFrame(
        [
            {
                "lighting": "well_lit",
                "delta_e_to_reference": 2.0,
            },
            {
                "lighting": "poorly_lit",
                "delta_e_to_reference": 5.0,
            },
        ]
    )

    outputs = create_charts(records, repeatability, tmp_path)

    assert (tmp_path / "charts" / "repeatability_by_lighting.png") in outputs
    assert (tmp_path / "charts" / "repeatability_by_lighting.png").exists()
