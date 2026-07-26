"""Validate candidate-confidence behavior on a small authorized image set.

This is a focused semantic check, not a shade-accuracy benchmark. It records
whether confidence varies by candidate, whether shade-family stability changes
the score beyond color fit, and whether readiness caps are respected. Source
photos are read in place and are never copied into the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_pipeline import analyze_rgb_image
from src.image_io import open_rgb_image_with_metadata
from src.shade_catalog import (
    FOUNDATION_ONLY_SCOPE,
    filter_catalog_by_product_scope,
    load_default_catalog,
)


def _finite(value) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _analyze(path: Path, catalog_df: pd.DataFrame) -> tuple[list[dict], dict]:
    image, metadata = open_rgb_image_with_metadata(path)
    result = analyze_rgb_image(
        np.asarray(image),
        catalog_df,
        image_color_metadata=metadata.as_dict(),
    )
    if not result.success:
        return [], {
            "image": path.name,
            "success": False,
            "error": result.error or getattr(result.face_result, "error", ""),
        }

    readiness = result.recommendation_readiness
    rows = []
    for match in result.matches:
        breakdown = match.confidence_breakdown or {}
        rows.append(
            {
                "image": path.name,
                "readiness_state": readiness.state,
                "readiness_score": _finite(readiness.score),
                "readiness_cap": _finite(readiness.confidence_cap),
                "rank": int(match.rank),
                "brand": match.brand,
                "product": match.product or "",
                "shade_name": match.shade_name,
                "distribution_delta_e": _finite(match.distribution_delta_e),
                "color_fit": _finite(match.color_fit_score),
                "shade_family_stability": _finite(
                    match.shade_family_stability_score
                ),
                "stability_source": match.confidence_stability_source,
                "catalog_evidence": _finite(match.catalog_quality_score),
                "candidate_evidence": _finite(
                    breakdown.get("candidate_evidence")
                ),
                "candidate_confidence": _finite(match.candidate_confidence),
            }
        )
    return rows, {
        "image": path.name,
        "success": True,
        "readiness_state": readiness.state,
        "readiness_score": _finite(readiness.score),
        "readiness_cap": _finite(readiness.confidence_cap),
        "candidate_spread": (
            max(row["candidate_confidence"] for row in rows)
            - min(row["candidate_confidence"] for row in rows)
            if rows
            else 0.0
        ),
    }


def _correlation(frame: pd.DataFrame, left: str, right: str) -> float | None:
    subset = frame[[left, right]].dropna()
    if len(subset) < 2 or subset[left].nunique() < 2 or subset[right].nunique() < 2:
        return None
    return float(subset[left].corr(subset[right]))


def _summary(rows: pd.DataFrame, captures: list[dict]) -> dict:
    successful = [item for item in captures if item.get("success")]
    if rows.empty:
        return {
            "images_requested": len(captures),
            "images_succeeded": len(successful),
            "candidates": 0,
        }

    normalized_confidence = rows["candidate_confidence"] / rows["readiness_cap"]
    working = rows.assign(normalized_confidence=normalized_confidence)
    inversions = 0
    differentiated = 0
    for _, group in rows.groupby("image"):
        if float(group["candidate_confidence"].max() - group["candidate_confidence"].min()) >= 0.01:
            differentiated += 1
        ordered = group.sort_values("rank")
        values = ordered["candidate_confidence"].tolist()
        if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
            inversions += 1

    evidence_minus_fit = (
        working["candidate_evidence"] - working["color_fit"]
    ).abs()
    catalog_values = sorted(
        float(value) for value in working["catalog_evidence"].dropna().unique()
    )
    fallback_counts = (
        working["stability_source"].value_counts(dropna=False).to_dict()
    )
    return {
        "images_requested": len(captures),
        "images_succeeded": len(successful),
        "candidates": int(len(working)),
        "images_with_at_least_1pp_candidate_spread": differentiated,
        "images_with_rank_confidence_inversion": inversions,
        "mean_candidate_spread": float(
            np.mean([item["candidate_spread"] for item in successful])
        ),
        "color_fit_candidate_evidence_correlation": _correlation(
            working, "color_fit", "candidate_evidence"
        ),
        "color_fit_normalized_confidence_correlation": _correlation(
            working, "color_fit", "normalized_confidence"
        ),
        "mean_absolute_candidate_evidence_minus_color_fit": float(
            evidence_minus_fit.mean()
        ),
        "catalog_evidence_unique_values": catalog_values,
        "catalog_evidence_is_constant": len(catalog_values) <= 1,
        "stability_source_counts": fallback_counts,
        "all_confidences_respect_readiness_cap": bool(
            (
                working["candidate_confidence"]
                <= working["readiness_cap"] + 1e-12
            ).all()
        ),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            [
                str(value).replace("|", r"\|").replace("\n", " ")
                for value in values
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _markdown(summary: dict, captures: list[dict], rows: pd.DataFrame) -> str:
    lines = [
        "# Candidate Confidence Validation",
        "",
        "This focused check uses authorized self-test photos. It validates score "
        "semantics and differentiation; it does not measure correct-shade accuracy.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = json.dumps(value, sort_keys=True)
        lines.append(f"- **{label}:** {rendered}")

    lines.extend(["", "## Per-capture outcome", ""])
    capture_frame = pd.DataFrame(captures)
    lines.append(_markdown_table(capture_frame))

    lines.extend(["", "## Candidate evidence", ""])
    columns = [
        "image",
        "rank",
        "shade_name",
        "readiness_state",
        "readiness_cap",
        "distribution_delta_e",
        "color_fit",
        "shade_family_stability",
        "stability_source",
        "catalog_evidence",
        "candidate_evidence",
        "candidate_confidence",
    ]
    display = rows[columns].copy() if not rows.empty else rows
    for column in (
        "readiness_cap",
        "color_fit",
        "shade_family_stability",
        "catalog_evidence",
        "candidate_evidence",
        "candidate_confidence",
    ):
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.1%}"
            )
    if "distribution_delta_e" in display:
        display["distribution_delta_e"] = display[
            "distribution_delta_e"
        ].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    lines.append(_markdown_table(display))
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Ranking remains color-first; confidence is an evidence-strength score.",
            "- A lower-ranked candidate may have higher confidence when its shade "
            "family is more stable.",
            "- Constant catalog evidence is an intended baseline, not a rank "
            "discriminator.",
            "- Candidate confidence must never exceed its capture-readiness cap.",
            "- Family stability is preferred; exact-product fallback must be marked.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="Image_*.*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        path for path in args.folder.glob(args.pattern) if path.is_file()
    )
    _, catalog_df, catalog_warnings = load_default_catalog()
    catalog_df = filter_catalog_by_product_scope(
        catalog_df, FOUNDATION_ONLY_SCOPE
    )

    candidate_rows = []
    captures = []
    for path in paths:
        rows, capture = _analyze(path, catalog_df)
        candidate_rows.extend(rows)
        captures.append(capture)
        print(
            f"{path.name}: "
            f"{capture.get('readiness_state', capture.get('error', 'failed'))}"
        )

    frame = pd.DataFrame(candidate_rows)
    summary = _summary(frame, captures)
    payload = {
        "source_folder": args.folder.name,
        "source_images_copied": False,
        "catalog_warnings": catalog_warnings,
        "summary": summary,
        "captures": captures,
    }

    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "candidate_results.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(
        _markdown(summary, captures, frame) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["images_succeeded"] == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
