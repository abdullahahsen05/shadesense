"""Create presentation-ready multi-photo repeatability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_photo_evaluation import (
    build_multi_photo_repeatability,
    summarize_multi_photo_repeatability,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = pd.read_csv(args.results, keep_default_na=False)
    comparisons = build_multi_photo_repeatability(records)
    summary = summarize_multi_photo_repeatability(comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(
        args.output / "multi_photo_repeatability.csv",
        index=False,
    )
    (args.output / "multi_photo_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Multi-photo consensus evaluation",
        "",
        f"- Subjects evaluated: {summary.get('subject_count', 0)}",
    ]
    if comparisons.empty:
        markdown.append("- No subjects had a reference plus two input captures.")
    else:
        markdown.extend(
            [
                "- Median consensus-to-reference Delta E: "
                f"{summary['median_consensus_to_reference_delta_e']:.2f}",
                "- Median individual-to-reference Delta E: "
                f"{summary['median_individual_to_reference_delta_e']:.2f}",
                "- Median improvement: "
                f"{summary['median_improvement_delta_e']:.2f} Delta E",
                "- Subjects improved: "
                f"{summary['subjects_improved_rate']:.1%}",
                "",
                summary["method_note"],
            ]
        )
    (args.output / "multi_photo_summary.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
