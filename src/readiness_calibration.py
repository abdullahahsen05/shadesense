"""Data-backed readiness threshold configuration and calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


READINESS_CALIBRATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "readiness_calibration.json"
)


@dataclass(frozen=True)
class ReadinessThresholds:
    ready_score: float = 74.0
    ready_extraction_score: float = 68.0
    ready_lighting_score: float = 0.60
    ready_max_uncertainty: float = 7.5
    ready_max_sensitivity: float = 4.5
    ready_min_bootstrap_family_top3: float = 0.55
    ready_min_lighting_family_top3: float = 0.45
    caution_score: float = 52.0
    caution_extraction_score: float = 50.0
    caution_max_uncertainty: float = 10.0
    caution_max_sensitivity: float = 6.5
    caution_min_bootstrap_family_top3: float = 0.35
    caution_min_lighting_family_top3: float = 0.30
    source: str = "built-in conservative defaults"


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin(("true", "1", "yes"))


@lru_cache(maxsize=4)
def load_readiness_thresholds(
    path: str | Path = READINESS_CALIBRATION_PATH,
) -> ReadinessThresholds:
    path = Path(path)
    if not path.exists():
        return ReadinessThresholds()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("thresholds", payload)
    allowed = set(ReadinessThresholds.__dataclass_fields__)
    filtered = {key: value for key, value in values.items() if key in allowed}
    filtered["source"] = payload.get(
        "source",
        f"calibrated from {path.name}",
    )
    return ReadinessThresholds(**filtered)


def _classification_metrics(frame: pd.DataFrame, thresholds: dict) -> dict:
    low_signal = _bool_series(frame["lighting_low_signal"])
    base_success = _bool_series(frame["pipeline_success"]) & ~low_signal
    readiness_score = pd.to_numeric(
        frame["readiness_score"],
        errors="coerce",
    )
    extraction_score = pd.to_numeric(
        frame["extraction_quality_score"],
        errors="coerce",
    )
    uncertainty = pd.to_numeric(
        frame["capture_uncertainty_delta_e_p90"],
        errors="coerce",
    )
    sensitivity = pd.to_numeric(
        frame["lighting_sensitivity_delta_e_p90"],
        errors="coerce",
    )
    bootstrap_family_top3 = pd.to_numeric(
        frame.get("bootstrap_top3_family_stability", 1.0),
        errors="coerce",
    )
    lighting_family_top3 = pd.to_numeric(
        frame.get("lighting_top3_family_stability", 1.0),
        errors="coerce",
    )
    predicted_usable = (
        base_success
        & (readiness_score >= thresholds["caution_score"])
        & (extraction_score >= thresholds["caution_extraction_score"])
        & (uncertainty <= thresholds["caution_max_uncertainty"])
        & (sensitivity <= thresholds["caution_max_sensitivity"])
        & (
            bootstrap_family_top3
            >= thresholds["caution_min_bootstrap_family_top3"]
        )
        & (
            lighting_family_top3
            >= thresholds["caution_min_lighting_family_top3"]
        )
    )
    predicted_ready = (
        base_success
        & (readiness_score >= thresholds["ready_score"])
        & (extraction_score >= thresholds["ready_extraction_score"])
        & (
            pd.to_numeric(frame["lighting_score"], errors="coerce")
            >= thresholds["ready_lighting_score"]
        )
        & (uncertainty <= thresholds["ready_max_uncertainty"])
        & (sensitivity <= thresholds["ready_max_sensitivity"])
        & (
            bootstrap_family_top3
            >= thresholds["ready_min_bootstrap_family_top3"]
        )
        & (
            lighting_family_top3
            >= thresholds["ready_min_lighting_family_top3"]
        )
    )
    actual_usable = frame["expected_capture_label"] == "usable"
    actual_recapture = frame["expected_capture_label"] == "recapture"
    usable_accept = (
        float(predicted_usable[actual_usable].mean())
        if actual_usable.any()
        else None
    )
    false_ready = (
        float(predicted_ready[actual_recapture].mean())
        if actual_recapture.any()
        else None
    )
    false_usable = (
        float(predicted_usable[actual_recapture].mean())
        if actual_recapture.any()
        else None
    )
    usable_ready = (
        float(predicted_ready[actual_usable].mean())
        if actual_usable.any()
        else None
    )
    recapture_reject = (
        float((~predicted_usable[actual_recapture]).mean())
        if actual_recapture.any()
        else None
    )
    return {
        "count": int(len(frame)),
        "usable_count": int(actual_usable.sum()),
        "recapture_count": int(actual_recapture.sum()),
        "usable_accept_rate": usable_accept,
        "usable_ready_rate": usable_ready,
        "false_usable_rate": false_usable,
        "false_ready_rate": false_ready,
        "recapture_reject_rate": recapture_reject,
    }


def calibrate_readiness_thresholds(
    records: pd.DataFrame,
) -> tuple[ReadinessThresholds, dict]:
    """Grid-search conservative caution gating on development labels only."""
    clear = records[
        (records["dataset"] == "mste")
        & records["expected_capture_label"].isin(["usable", "recapture"])
    ].copy()
    development = clear[clear["split"] == "development"]
    locked_test = clear[clear["split"] == "locked_test"]
    if len(development) < 10:
        raise ValueError("At least 10 clear development rows are required.")

    default = ReadinessThresholds()
    best_caution = None
    for caution_score in (48.0, 52.0, 56.0, 60.0, 64.0, 68.0):
        for max_uncertainty in (8.0, 9.0, 10.0, 11.0):
            for max_sensitivity in (4.5, 5.5, 6.5, 7.5):
                candidate = {
                    **asdict(default),
                    "caution_score": caution_score,
                    "caution_max_uncertainty": max_uncertainty,
                    "caution_max_sensitivity": max_sensitivity,
                }
                metrics = _classification_metrics(development, candidate)
                usable_accept = metrics["usable_accept_rate"] or 0.0
                false_usable = metrics["false_usable_rate"] or 0.0
                objective = (
                    4.0 * false_usable
                    + (1.0 - usable_accept)
                    + max(0.55 - usable_accept, 0.0) * 10.0
                )
                tie_break = (
                    objective,
                    false_usable,
                    -usable_accept,
                    abs(caution_score - default.caution_score),
                )
                if best_caution is None or tie_break < best_caution[0]:
                    best_caution = (tie_break, candidate)

    best_ready = None
    ready_candidates = [
        (score, uncertainty, sensitivity)
        for score in (70.0, 72.0, 74.0, 76.0, 78.0, 80.0)
        for uncertainty in (6.0, 7.0, 8.0)
        for sensitivity in (4.0, 5.0, 6.0)
    ]
    for (
        ready_score,
        ready_max_uncertainty,
        ready_max_sensitivity,
    ) in ready_candidates:
        candidate = {
            **asdict(default),
            "ready_score": ready_score,
            "ready_max_uncertainty": ready_max_uncertainty,
            "ready_max_sensitivity": ready_max_sensitivity,
        }
        metrics = _classification_metrics(development, candidate)
        usable_ready = metrics["usable_ready_rate"] or 0.0
        false_ready = metrics["false_ready_rate"] or 0.0
        objective = (
            6.0 * false_ready
            + max(0.05 - usable_ready, 0.0) * 3.0
        )
        tie_break = (
            objective,
            false_ready,
            abs(ready_score - default.ready_score),
            abs(
                ready_max_uncertainty
                - default.ready_max_uncertainty
            ),
            abs(
                ready_max_sensitivity
                - default.ready_max_sensitivity
            ),
            -usable_ready,
        )
        if best_ready is None or tie_break < best_ready[0]:
            best_ready = (tie_break, candidate)
    assert best_caution is not None and best_ready is not None
    values = {
        **asdict(default),
        "caution_score": best_caution[1]["caution_score"],
        "caution_max_uncertainty": best_caution[1][
            "caution_max_uncertainty"
        ],
        "caution_max_sensitivity": best_caution[1][
            "caution_max_sensitivity"
        ],
        "ready_score": best_ready[1]["ready_score"],
        "ready_max_uncertainty": best_ready[1][
            "ready_max_uncertainty"
        ],
        "ready_max_sensitivity": best_ready[1][
            "ready_max_sensitivity"
        ],
    }
    values["source"] = "MST-E development metadata calibration"
    thresholds = ReadinessThresholds(
        **{
            key: value
            for key, value in values.items()
            if key in ReadinessThresholds.__dataclass_fields__
        }
    )
    development_metrics = _classification_metrics(
        development,
        asdict(thresholds),
    )
    evidence = {
        "objective": float(best_caution[0][0] + best_ready[0][0]),
        "development": development_metrics,
        "locked_test": _classification_metrics(
            locked_test,
            asdict(thresholds),
        ),
        "note": (
            "MST-E lighting/pose/mask labels are capture-quality proxies, not "
            "foundation product ground truth."
        ),
    }
    return thresholds, evidence


def calibration_payload(
    thresholds: ReadinessThresholds,
    evidence: dict,
    *,
    baseline_run: str,
    manifest_sha256: str,
) -> dict:
    return {
        "source": thresholds.source,
        "baseline_run": baseline_run,
        "manifest_sha256": manifest_sha256,
        "thresholds": asdict(thresholds),
        "evidence": evidence,
    }
