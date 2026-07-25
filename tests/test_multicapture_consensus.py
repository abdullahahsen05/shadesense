import numpy as np

from src.multicapture_consensus import (
    CaptureEvidence,
    build_multicapture_consensus,
)


def test_consensus_rejects_dim_real_capture_pattern():
    captures = [
        CaptureEvidence("bright-2", (71.75, 9.32, 17.93), 83, 0.70, 5.0),
        CaptureEvidence("bright-4", (77.37, 8.53, 17.72), 78, 0.74, 4.0),
        CaptureEvidence("reference-6", (75.21, 5.27, 15.89), 79, 0.70, 5.0),
        CaptureEvidence("bright-7", (76.33, 8.05, 16.06), 79, 0.68, 5.0),
        CaptureEvidence("dim-3", (43.00, 7.44, 14.48), 53, 0.52, 9.0, True),
        CaptureEvidence("dim-5", (38.76, 6.47, 13.52), 45, 0.45, 20.0, True),
    ]

    result = build_multicapture_consensus(captures)

    assert result.success
    assert result.anchor_capture_id in {
        "bright-2",
        "bright-4",
        "reference-6",
        "bright-7",
    }
    assert {"dim-3", "dim-5"} <= set(result.excluded_capture_ids)
    assert {"dim-3", "dim-5"} <= set(
        result.excluded_low_signal_capture_ids
    )
    assert 70.0 <= result.lab[0] <= 78.0
    assert result.uncertainty_radius_p90 < 6.0


def test_consensus_is_deterministic_and_uses_observed_medoid():
    captures = [
        CaptureEvidence("a", (50.0, 8.0, 14.0)),
        CaptureEvidence("b", (51.0, 8.0, 14.0)),
        CaptureEvidence("c", (52.0, 8.0, 14.0)),
    ]

    first = build_multicapture_consensus(captures)
    second = build_multicapture_consensus(captures)

    assert first == second
    assert first.lab in {capture.lab for capture in captures}
    assert np.isfinite(first.repeatability_score)


def test_consensus_requires_two_usable_captures():
    result = build_multicapture_consensus(
        [CaptureEvidence("only", (50.0, 8.0, 14.0))]
    )
    assert not result.success
    assert "two usable captures" in result.warnings[0]
