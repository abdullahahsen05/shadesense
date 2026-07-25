"""Write paired CSV/JSON/Markdown evidence for two evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_comparison import (
    build_paired_comparison,
    summarize_paired_comparison,
)


def _optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, keep_default_na=False) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paired = build_paired_comparison(
        pd.read_csv(args.baseline / "per_image_results.csv", keep_default_na=False),
        pd.read_csv(args.candidate / "per_image_results.csv", keep_default_na=False),
        _optional_csv(args.baseline / "recommendations.csv"),
        _optional_csv(args.candidate / "recommendations.csv"),
    )
    summary = summarize_paired_comparison(paired)
    args.output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output / "paired_image_comparison.csv", index=False)
    (args.output / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ShadeSense paired evaluation comparison",
        "",
        f"- Paired images: {summary['paired_image_count']}",
        "- Pipeline success changes: "
        f"{summary['pipeline_success_change_count']}",
        "- Top-1 catalog candidate changed: "
        f"{summary['top1_change_rate']:.1%}"
        if summary["top1_change_rate"] is not None
        else "- Top-1 catalog candidate change: unavailable",
        "- Readiness upgrades: "
        f"{summary['readiness_upgrade_rate']:.1%}"
        if summary["readiness_upgrade_rate"] is not None
        else "- Readiness upgrades: unavailable",
        "- Readiness downgrades: "
        f"{summary['readiness_downgrade_rate']:.1%}"
        if summary["readiness_downgrade_rate"] is not None
        else "- Readiness downgrades: unavailable",
        "",
        summary["interpretation_note"],
    ]
    (args.output / "comparison_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
