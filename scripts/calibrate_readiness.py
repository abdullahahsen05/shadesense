"""Calibrate conservative readiness gates from a completed baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.readiness_calibration import (
    READINESS_CALIBRATION_PATH,
    calibrate_readiness_thresholds,
    calibration_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=READINESS_CALIBRATION_PATH,
    )
    args = parser.parse_args()

    records = pd.read_csv(args.results)
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    thresholds, evidence = calibrate_readiness_thresholds(records)
    payload = calibration_payload(
        thresholds,
        evidence,
        baseline_run=run_config.get("run_label", "unknown"),
        manifest_sha256=run_config.get("manifest_sha256", "unknown"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
