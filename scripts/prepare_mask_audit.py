"""Prepare a stratified 100-image mask audit from an evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mask_audit import build_mask_audit_manifest, summarize_mask_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--mste-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = pd.read_csv(args.results, keep_default_na=False)
    audit = build_mask_audit_manifest(
        records,
        count=args.count,
        mste_count=args.mste_count,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    summary = summarize_mask_audit(audit)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(audit)} review rows to {args.output}")
    print(
        "Run: streamlit run scripts/mask_audit_app.py -- "
        f"--dataset-root <folder> --audit \"{args.output}\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
