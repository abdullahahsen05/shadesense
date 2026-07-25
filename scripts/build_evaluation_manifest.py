"""Build and validate the frozen ShadeSense AI 400-image benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_dataset import (
    build_benchmark_manifest,
    validate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "benchmark_manifest.csv",
    )
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--mste-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_benchmark_manifest(
        args.dataset_root,
        total_count=args.count,
        mste_count=args.mste_count,
        seed=args.seed,
    )
    errors = validate_manifest(manifest, args.dataset_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Saved {len(manifest)} rows to {args.output}")
    print("Dataset counts:")
    print(manifest["dataset"].value_counts().to_string())
    print("Split counts:")
    print(manifest["split"].value_counts().to_string())
    print("MST-E tone counts:")
    print(
        manifest.loc[manifest["dataset"] == "mste", "mst"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("Expected capture labels:")
    print(manifest["expected_capture_label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
