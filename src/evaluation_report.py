"""Presentation-oriented reports and charts for benchmark results."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _percent(value) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _number(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def create_charts(
    records: pd.DataFrame,
    repeatability: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    chart_dir = Path(output_dir) / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    readiness = (
        records.get("readiness_state", pd.Series(dtype=str))
        .fillna("unavailable")
        .value_counts()
        .reindex(["ready", "caution", "provisional", "unavailable"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    readiness.plot(
        kind="bar",
        ax=ax,
        color=["#3D9970", "#E1A95F", "#C95D63", "#8C8C8C"],
    )
    ax.set_title("Recommendation readiness")
    ax.set_ylabel("Images")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    path = chart_dir / "readiness_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    mste = records[records["dataset"] == "mste"].copy()
    if not mste.empty:
        grouped = mste.groupby("mst").agg(
            face_detection_rate=("face_detected", "mean"),
            pipeline_success_rate=("pipeline_success", "mean"),
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        grouped.plot(kind="bar", ax=ax, color=["#4F86C6", "#61B15A"])
        ax.set_title("MST-E success rates by expert Monk tone")
        ax.set_ylabel("Rate")
        ax.set_xlabel("Monk Skin Tone")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        fig.tight_layout()
        path = chart_dir / "mste_success_by_tone.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    if not repeatability.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        order = ["well_lit", "poorly_lit"]
        values = [
            repeatability[
                repeatability["lighting"] == lighting
            ]["delta_e_to_reference"].dropna()
            for lighting in order
        ]
        available = [
            (label, value)
            for label, value in zip(order, values)
            if len(value)
        ]
        if available:
            ax.boxplot(
                [item[1] for item in available],
                labels=[item[0].replace("_", " ") for item in available],
                showfliers=False,
            )
            ax.axhline(3.0, color="#E1A95F", linestyle="--", linewidth=1)
            ax.axhline(6.0, color="#C95D63", linestyle="--", linewidth=1)
            ax.set_title("Same-subject Delta E relative to usable reference")
            ax.set_ylabel("CIEDE2000")
            fig.tight_layout()
            path = chart_dir / "repeatability_by_lighting.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)

    return outputs


def write_summary_markdown(
    metrics: dict,
    records: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    run_label: str,
    run_metadata: dict,
) -> None:
    extraction = metrics["extraction_quality"]
    uncertainty = metrics["capture_uncertainty_delta_e"]
    repeatability = metrics["repeatability"]["delta_e_to_reference"]
    readiness = metrics["metadata_label_readiness"]
    lines = [
        f"# ShadeSense AI evaluation — {run_label}",
        "",
        "## Reproducibility",
        "",
        f"- Git commit: `{run_metadata.get('git_commit', 'unknown')}`",
        f"- Manifest SHA-256: `{run_metadata.get('manifest_sha256', 'unknown')}`",
        f"- Catalog SHA-256: `{run_metadata.get('catalog_sha256', 'unknown')}`",
        f"- Random seed: `{run_metadata.get('seed', 'unknown')}`",
        f"- Images requested: {metrics['image_count']}",
        "",
        "## Core results",
        "",
        f"- Face detection rate: {_percent(metrics['face_detection_rate'])}",
        f"- Full pipeline success rate: {_percent(metrics['pipeline_success_rate'])}",
        f"- Median extraction quality: {_number(extraction['median'], 1)}/100",
        f"- Median total capture uncertainty: {_number(uncertainty['median'], 2)} Delta E",
        f"- 90th-percentile total capture uncertainty: {_number(uncertainty['p90'], 2)} Delta E",
        f"- Median same-subject shift from usable MST-E reference images: {_number(repeatability['median'], 2)} Delta E",
        f"- 90th-percentile same-subject shift: {_number(repeatability['p90'], 2)} Delta E",
        "",
        "## Capture gating against clear MST-E metadata labels",
        "",
        f"- Clearly labelled captures: {readiness['clear_label_count']}",
        f"- Usable-capture acceptance rate: {_percent(readiness['usable_accept_rate'])}",
        f"- Recapture rejection rate: {_percent(readiness['recapture_reject_rate'])}",
        f"- Dangerous false-ready rate: {_percent(readiness['false_ready_rate'])}",
        "",
        "These readiness labels are metadata-derived proxies, not foundation shade "
        "ground truth. Exact product accuracy is not claimed because the public "
        "datasets do not provide verified physical foundation matches.",
        "",
        "## Readiness distribution",
        "",
    ]
    for state, count in metrics["readiness_distribution"].items():
        lines.append(f"- {state}: {count}")
    lines.extend(
        [
            "",
            "## Top-ranked product types",
            "",
        ]
    )
    for product_type, count in metrics["top_recommendation_product_types"].items():
        lines.append(f"- {product_type}: {count}")

    failures = records[~records["pipeline_success"].fillna(False)]
    lines.extend(
        [
            "",
            "## Failures",
            "",
            f"- Images without a complete result: {len(failures)}",
        ]
    )
    if len(failures):
        for _, row in failures.head(12).iterrows():
            lines.append(
                f"- `{row['benchmark_id']}` ({row['dataset']}): "
                f"{row.get('error', 'unknown error')}"
            )

    if not subgroup_metrics.empty:
        lines.extend(
            [
                "",
                "## Subgroup audit",
                "",
                "Full subgroup metrics are saved in `subgroup_metrics.csv`. "
                "Race/demographic labels are used only to audit system coverage "
                "and are never treated as skin-tone ground truth.",
            ]
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(
    metrics: dict,
    records: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    repeatability: pd.DataFrame,
    output_path: str | Path,
    *,
    run_label: str,
) -> None:
    output_path = Path(output_path)
    charts = sorted((output_path.parent / "charts").glob("*.png"))
    chart_html = "".join(
        f'<figure><img src="charts/{escape(path.name)}" alt="{escape(path.stem)}">'
        f"<figcaption>{escape(path.stem.replace('_', ' ').title())}</figcaption></figure>"
        for path in charts
    )
    worst = records.sort_values(
        "capture_uncertainty_delta_e_p90",
        ascending=False,
        na_position="last",
    ).head(20)
    repeatability_worst = (
        repeatability.sort_values("delta_e_to_reference", ascending=False).head(20)
        if not repeatability.empty
        else pd.DataFrame()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShadeSense AI — {escape(run_label)}</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 2rem auto; max-width: 1180px;
       color: #18201c; line-height: 1.45; padding: 0 1rem; }}
h1, h2 {{ color: #173c2d; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1rem; }}
.card {{ border:1px solid #d8e1dc; border-radius:10px; padding:1rem; background:#f8fbf9; }}
.value {{ font-size:1.8rem; font-weight:700; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1rem; }}
figure {{ margin:0; border:1px solid #e0e5e2; padding:.75rem; border-radius:10px; }}
figure img {{ width:100%; }}
table {{ border-collapse:collapse; width:100%; font-size:.82rem; overflow:auto; display:block; }}
th, td {{ border:1px solid #dde3df; padding:.35rem .5rem; white-space:nowrap; }}
th {{ background:#eef5f1; }}
.note {{ background:#fff8dc; border-left:4px solid #d6a928; padding:1rem; }}
</style>
</head>
<body>
<h1>ShadeSense AI evaluation</h1>
<p>{escape(run_label)}</p>
<div class="metrics">
<div class="card"><div>Images</div><div class="value">{metrics['image_count']}</div></div>
<div class="card"><div>Face detection</div><div class="value">{_percent(metrics['face_detection_rate'])}</div></div>
<div class="card"><div>Pipeline success</div><div class="value">{_percent(metrics['pipeline_success_rate'])}</div></div>
<div class="card"><div>Median repeatability ΔE</div><div class="value">{_number(metrics['repeatability']['delta_e_to_reference']['median'])}</div></div>
<div class="card"><div>False-ready rate</div><div class="value">{_percent(metrics['metadata_label_readiness']['false_ready_rate'])}</div></div>
</div>
<p class="note">MST-E and FairFace measure robustness and consistency. They do not
contain verified foundation product matches, so this report does not claim exact
foundation-shade accuracy.</p>
<h2>Charts</h2>
<div class="charts">{chart_html}</div>
<h2>Highest capture uncertainty</h2>
{worst[['benchmark_id','dataset','mst','demographic_group','readiness_state','capture_uncertainty_delta_e_p90','error']].to_html(index=False, escape=True)}
<h2>Largest same-subject shifts</h2>
{repeatability_worst.to_html(index=False, escape=True) if not repeatability_worst.empty else '<p>No repeatability rows available.</p>'}
<h2>Subgroup metrics</h2>
{subgroup_metrics.to_html(index=False, escape=True)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_metrics_json(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
