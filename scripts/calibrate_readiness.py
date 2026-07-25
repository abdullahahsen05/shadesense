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
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    records = pd.read_csv(args.results)
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    thresholds, evidence = calibrate_readiness_thresholds(records)
    payload = calibration_payload(
        thresholds,
        evidence,
        evaluation_run=run_config.get("run_label", "unknown"),
        manifest_sha256=run_config.get("manifest_sha256", "unknown"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.report_output is not None:
        development = evidence["development"]
        locked = evidence["locked_test"]
        lines = [
            "# ShadeSense readiness calibration",
            "",
            f"- Candidate run: `{payload['evaluation_run']}`",
            f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
            "- Source: MST-E development capture metadata",
            "",
            "## Selected thresholds",
            "",
            f"- Caution readiness score: {thresholds.caution_score:.0f}",
            "- Caution maximum uncertainty: "
            f"{thresholds.caution_max_uncertainty:.1f} Delta E",
            "- Caution maximum lighting sensitivity: "
            f"{thresholds.caution_max_sensitivity:.1f} Delta E",
            f"- Ready readiness score: {thresholds.ready_score:.0f}",
            "- Ready maximum uncertainty: "
            f"{thresholds.ready_max_uncertainty:.1f} Delta E",
            "- Ready maximum lighting sensitivity: "
            f"{thresholds.ready_max_sensitivity:.1f} Delta E",
            "",
            "## Development selection evidence",
            "",
            f"- Usable acceptance: {development['usable_accept_rate']:.1%}",
            f"- Recapture false usable: {development['false_usable_rate']:.1%}",
            f"- Dangerous false Ready: {development['false_ready_rate']:.1%}",
            "",
            "## Locked-test audit",
            "",
            f"- Usable acceptance: {locked['usable_accept_rate']:.1%}",
            f"- Recapture false usable: {locked['false_usable_rate']:.1%}",
            f"- Dangerous false Ready: {locked['false_ready_rate']:.1%}",
            "",
            "These labels are capture-quality proxies, not verified physical "
            "foundation matches.",
        ]
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
