"""Local Streamlit review tool for the 100-image mask audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_pipeline import analyze_rgb_image
from src.evaluation_dataset import ArchiveImageStore
from src.mask_audit import REGIONS, REVIEW_LABELS, summarize_mask_audit
from src.visualization import draw_all_region_masks, draw_region_mask


def _arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args


args = _arguments()
st.set_page_config(page_title="ShadeSense mask audit", layout="wide")
st.title("ShadeSense AI - manual mask audit")

audit = pd.read_csv(args.audit, keep_default_na=False)
summary = summarize_mask_audit(audit)
st.progress(summary["review_completion"])
st.caption(
    f"{summary['reviewed_count']} of {summary['selected_count']} reviewed."
)

selected_position = st.number_input(
    "Audit image",
    min_value=1,
    max_value=len(audit),
    value=1,
    step=1,
)
index = int(selected_position) - 1
row = audit.iloc[index]

with ArchiveImageStore(args.dataset_root) as store:
    image_rgb, metadata = store.load_rgb_with_metadata(row)
analysis = analyze_rgb_image(image_rgb, image_color_metadata=metadata)

st.subheader(
    f"{row['benchmark_id']} | {row['dataset']} | "
    f"risk {float(row['mask_risk_score']):.2f}"
)
if not analysis.face_result.success:
    st.error(analysis.face_result.error)
else:
    col1, col2 = st.columns(2)
    with col1:
        st.image(analysis.image_rgb, caption="Analysis image", width=500)
    with col2:
        st.image(
            draw_all_region_masks(analysis.visualization_rgb, analysis.masks),
            caption="Combined region masks",
            width=500,
        )
    region_columns = st.columns(4)
    selections = {}
    for column, region in zip(region_columns, REGIONS):
        current = str(row[f"{region}_review"])
        with column:
            st.image(
                draw_region_mask(
                    analysis.visualization_rgb,
                    analysis.masks[region],
                ),
                caption=region.replace("_", " ").title(),
                width=220,
            )
            selections[region] = st.selectbox(
                f"{region.replace('_', ' ').title()} review",
                REVIEW_LABELS,
                index=(
                    REVIEW_LABELS.index(current)
                    if current in REVIEW_LABELS
                    else 0
                ),
                key=f"{row['benchmark_id']}-{region}",
            )
    notes = st.text_area(
        "Review notes",
        value=str(row["review_notes"]),
        key=f"{row['benchmark_id']}-notes",
    )
    if st.button("Save this review"):
        for region, selection in selections.items():
            audit.loc[index, f"{region}_review"] = selection
        audit.loc[index, "review_notes"] = notes
        audit.loc[index, "reviewed"] = all(
            selection != "not_reviewed"
            for selection in selections.values()
        )
        audit.to_csv(args.audit, index=False)
        summary = summarize_mask_audit(audit)
        args.audit.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        st.success("Review saved.")
