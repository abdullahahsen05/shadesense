"""Run a restartable, presentation-oriented ShadeSense dataset evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_pipeline import analyze_rgb_image
from src.config import PUBLIC_SHADE_CATALOG_PATH
from src.evaluation_dataset import ArchiveImageStore, validate_manifest
from src.evaluation_metrics import (
    analysis_record,
    build_aggregate_metrics,
    recommendation_records,
)
from src.evaluation_report import (
    create_charts,
    write_html_report,
    write_metrics_json,
    write_summary_markdown,
)
from src.shade_catalog import load_shade_catalog
from src.visualization import draw_all_region_masks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _save_overlay(analysis, path: Path) -> None:
    overlay = draw_all_region_masks(
        analysis.visualization_rgb,
        analysis.masks,
    )
    image = Image.fromarray(np.asarray(overlay, dtype=np.uint8))
    image.thumbnail((720, 720))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=82, optimize=True)


def _finalize(
    output_dir: Path,
    run_label: str,
    run_metadata: dict,
) -> None:
    records = pd.DataFrame(_read_jsonl(output_dir / "results.jsonl"))
    recommendations = pd.DataFrame(
        _read_jsonl(output_dir / "recommendations.jsonl")
    )
    if records.empty:
        return
    records.to_csv(output_dir / "per_image_results.csv", index=False)
    recommendations.to_csv(
        output_dir / "recommendations.csv",
        index=False,
    )
    metrics, subgroups, repeatability = build_aggregate_metrics(
        records,
        recommendations,
    )
    subgroups.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    repeatability.to_csv(
        output_dir / "repeatability_metrics.csv",
        index=False,
    )
    write_metrics_json(metrics, output_dir / "aggregate_metrics.json")
    create_charts(records, repeatability, output_dir)
    write_summary_markdown(
        metrics,
        records,
        subgroups,
        output_dir / "summary.md",
        run_label=run_label,
        run_metadata=run_metadata,
    )
    write_html_report(
        metrics,
        records,
        subgroups,
        repeatability,
        output_dir / "report.html",
        run_label=run_label,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PUBLIC_SHADE_CATALOG_PATH,
    )
    parser.add_argument("--run-label", default="baseline-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    errors = validate_manifest(manifest, args.dataset_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.max_images is not None:
        manifest = manifest.head(args.max_images).copy()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    recommendations_jsonl = output_dir / "recommendations.jsonl"
    if not args.resume:
        for path in (results_jsonl, recommendations_jsonl):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; use --resume or a new output folder."
                )
    completed = {
        str(row["benchmark_id"])
        for row in _read_jsonl(results_jsonl)
    }

    run_metadata = {
        "run_label": args.run_label,
        "git_commit": _git_commit(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "catalog_path": str(args.catalog.resolve()),
        "catalog_sha256": _sha256(args.catalog),
        "dataset_root": str(args.dataset_root.resolve()),
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "requested_rows": int(len(manifest)),
        "raw_images_committed": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    catalog = load_shade_catalog(args.catalog)
    pending = manifest[
        ~manifest["benchmark_id"].astype(str).isin(completed)
    ]
    total = len(manifest)
    started = time.perf_counter()
    with ArchiveImageStore(args.dataset_root) as image_store:
        for position, (_, row) in enumerate(pending.iterrows(), start=1):
            benchmark_id = str(row["benchmark_id"])
            item_started = time.perf_counter()
            analysis = None
            error = None
            try:
                image_rgb = image_store.load_rgb(row)
                analysis = analyze_rgb_image(image_rgb, catalog)
            except Exception as exc:  # keep the dataset run restartable
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            elapsed = time.perf_counter() - item_started
            record = analysis_record(
                row,
                analysis,
                elapsed_seconds=elapsed,
                error=error,
            )
            _append_jsonl(results_jsonl, record)
            if analysis is not None:
                for recommendation in recommendation_records(row, analysis):
                    _append_jsonl(
                        recommendations_jsonl,
                        recommendation,
                    )
                if args.save_overlays and analysis.success:
                    _save_overlay(
                        analysis,
                        output_dir
                        / "debug_overlays"
                        / f"{benchmark_id}.jpg",
                    )

            done = len(completed) + position
            rate = (time.perf_counter() - started) / max(position, 1)
            remaining = max(total - done, 0) * rate
            print(
                f"[{done}/{total}] {benchmark_id} "
                f"{'ok' if analysis is not None and analysis.success else 'failed'} "
                f"{elapsed:.1f}s; ETA {remaining / 60.0:.1f}m",
                flush=True,
            )

    _finalize(output_dir, args.run_label, run_metadata)
    print(f"Evaluation outputs saved to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
