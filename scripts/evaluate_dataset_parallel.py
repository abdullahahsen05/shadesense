"""Run the existing evaluation harness in deterministic local process shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_dataset import _finalize
from src.config import PUBLIC_SHADE_CATALOG_PATH
from src.evaluation_dataset import validate_manifest
from src.shade_catalog import ALL_BASE_SCOPE, FOUNDATION_ONLY_SCOPE


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PUBLIC_SHADE_CATALOG_PATH)
    parser.add_argument("--run-label", default="candidate-v2")
    parser.add_argument(
        "--product-scope",
        choices=[ALL_BASE_SCOPE, FOUNDATION_ONLY_SCOPE],
        default=FOUNDATION_ONLY_SCOPE,
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8.")
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    errors = validate_manifest(manifest, args.dataset_root)
    if errors:
        raise ValueError("\n".join(errors))
    if args.max_images is not None:
        manifest = manifest.head(args.max_images).copy()
    args.output.mkdir(parents=True, exist_ok=True)
    root_results = args.output / "results.jsonl"
    if root_results.exists() and not args.resume:
        raise FileExistsError(
            f"{root_results} already exists; use --resume or a new output."
        )

    shard_root = args.output / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    worker_count = min(args.workers, len(manifest))
    processes = []
    log_handles = []
    for shard_index in range(worker_count):
        shard_manifest = manifest.iloc[shard_index::worker_count].copy()
        manifest_path = shard_root / f"manifest-{shard_index + 1}.csv"
        output_path = shard_root / f"worker-{shard_index + 1}"
        shard_manifest.to_csv(manifest_path, index=False)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_dataset.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--catalog",
            str(args.catalog),
            "--run-label",
            f"{args.run_label}-worker-{shard_index + 1}",
            "--product-scope",
            args.product_scope,
        ]
        if args.resume and (output_path / "results.jsonl").exists():
            command.append("--resume")
        if args.save_overlays:
            command.append("--save-overlays")
        log_handle = (shard_root / f"worker-{shard_index + 1}.log").open(
            "a",
            encoding="utf-8",
        )
        log_handles.append(log_handle)
        processes.append(
            subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        )
    started = time.perf_counter()
    return_codes = [process.wait() for process in processes]
    for handle in log_handles:
        handle.close()
    if any(code != 0 for code in return_codes):
        raise RuntimeError(
            f"Evaluation worker failure codes: {return_codes}. "
            f"Inspect {shard_root}/*.log."
        )

    order = {
        str(benchmark_id): index
        for index, benchmark_id in enumerate(manifest["benchmark_id"])
    }
    result_rows = []
    recommendation_rows = []
    overlay_target = args.output / "debug_overlays"
    for shard_index in range(worker_count):
        output_path = shard_root / f"worker-{shard_index + 1}"
        result_rows.extend(_read_jsonl(output_path / "results.jsonl"))
        recommendation_rows.extend(
            _read_jsonl(output_path / "recommendations.jsonl")
        )
        source_overlays = output_path / "debug_overlays"
        if args.save_overlays and source_overlays.exists():
            overlay_target.mkdir(parents=True, exist_ok=True)
            for source in source_overlays.glob("*.jpg"):
                shutil.copy2(source, overlay_target / source.name)
    result_rows.sort(key=lambda row: order[str(row["benchmark_id"])])
    recommendation_rows.sort(
        key=lambda row: (
            order[str(row["benchmark_id"])],
            int(row["rank"]),
        )
    )
    _write_jsonl(args.output / "results.jsonl", result_rows)
    _write_jsonl(
        args.output / "recommendations.jsonl",
        recommendation_rows,
    )
    run_metadata = {
        "run_label": args.run_label,
        "git_commit": _git_commit(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "catalog_path": str(args.catalog.resolve()),
        "catalog_sha256": _sha256(args.catalog),
        "dataset_root": str(args.dataset_root.resolve()),
        "product_scope": args.product_scope,
        "python": sys.version,
        "platform": platform.platform(),
        "requested_rows": int(len(manifest)),
        "workers": worker_count,
        "raw_images_committed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _finalize(args.output, args.run_label, run_metadata)
    print(
        f"Parallel evaluation saved {len(result_rows)} rows to "
        f"{args.output.resolve()} in {run_metadata['elapsed_seconds'] / 60:.1f}m."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
