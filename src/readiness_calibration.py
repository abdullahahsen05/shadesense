"""Data-backed readiness threshold configuration and calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
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
    low_signal = frame["lighting_low_signal"].astype(bool)
    predicted_usable = (
        frame["pipeline_success"].astype(bool)
        & ~low_signal
        & (
            pd.to_numeric(frame["readiness_score"], errors="coerce")
            >= thresholds["caution_score"]
        )
        & (
            pd.to_numeric(
                frame["extraction_quality_score"],
                errors="coerce",
            )
            >= thresholds["caution_extraction_score"]
        )
        & (
            pd.to_numeric(
                frame["capture_uncertainty_delta_e_p90"],
                errors="coerce",
            )
            <= thresholds["caution_max_uncertainty"]
        )
        & (
            pd.to_numeric(
                frame["lighting_sensitivity_delta_e_p90"],
                errors="coerce",
            )
            <= thresholds["caution_max_sensitivity"]
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
        float(predicted_usable[actual_recapture].mean())
        if actual_recapture.any()
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
    best = None
    for caution_score in np.arange(48.0, 68.1, 1.0):
        for max_uncertainty in np.arange(8.0, 11.6, 0.5):
            for max_sensitivity in np.arange(4.5, 8.1, 0.5):
                candidate = {
                    **asdict(default),
                    "caution_score": float(caution_score),
                    "caution_max_uncertainty": float(max_uncertainty),
                    "caution_max_sensitivity": float(max_sensitivity),
                }
                metrics = _classification_metrics(development, candidate)
                usable_accept = metrics["usable_accept_rate"] or 0.0
                false_ready = metrics["false_ready_rate"] or 0.0
                false_reject = 1.0 - usable_accept
                # Dangerous false-ready cases cost four times more, while a
                # minimum useful acceptance rate prevents an all-reject answer.
                constraint_penalty = max(0.55 - usable_accept, 0.0) * 10.0
                objective = (
                    4.0 * false_ready
                    + false_reject
                    + constraint_penalty
                )
                tie_break = (
                    objective,
                    false_ready,
                    -usable_accept,
                    abs(caution_score - default.caution_score),
                )
                if best is None or tie_break < best[0]:
                    best = (tie_break, candidate, metrics)
    assert best is not None
    values = best[1]
    values["source"] = "MST-E development metadata calibration"
    thresholds = ReadinessThresholds(
        **{
            key: value
            for key, value in values.items()
            if key in ReadinessThresholds.__dataclass_fields__
        }
    )
    evidence = {
        "objective": float(best[0][0]),
        "development": best[2],
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
