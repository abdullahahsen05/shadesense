"""ShadeSense AI — Streamlit UI entrypoint.

This module contains presentation logic only. CV/matching logic lives in `src/`.
"""

import numpy as np
import streamlit as st
from PIL import Image

from src.color_correction import apply_mild_color_correction, correction_settings_for_lighting
from src.confidence import build_quality_report, compute_confidence
from src.config import APP_NAME, TOP_K_SHADES
from src.explanation import build_explanation
from src.extraction_summary import build_skin_extraction_summary
from src.extraction_selection import run_dual_extraction
from src.face_detection import detect_face_landmarks
from src.image_quality import analyze_image_quality
from src.lighting_quality import analyze_lighting_quality
from src.region_masks import build_region_masks
from src.shade_catalog import (
    MOCK_CATALOG_KEY,
    PUBLIC_CATALOG_KEY,
    PUBLIC_CATALOG_LIMITATION,
    CatalogValidationError,
    catalog_definitions,
    load_default_catalog,
    load_shade_catalog,
    load_named_catalog,
)
from src.shade_matcher import match_shades
from src.visualization import (
    draw_all_region_masks,
    draw_face_landmarks,
    draw_region_mask,
    make_skin_swatch,
)

st.set_page_config(page_title=APP_NAME, layout="wide")

st.title(APP_NAME)
catalog_options = catalog_definitions()
try:
    default_catalog_key, _, default_catalog_warnings = load_default_catalog()
except (FileNotFoundError, CatalogValidationError) as exc:
    st.error(f"Could not load any shade catalog: {exc}")
    st.stop()

for warning in default_catalog_warnings:
    st.warning(warning)

catalog_keys = [PUBLIC_CATALOG_KEY, MOCK_CATALOG_KEY]
selected_catalog_key = st.selectbox(
    "Shade catalog",
    catalog_keys,
    format_func=lambda key: catalog_options[key].name,
    index=catalog_keys.index(default_catalog_key),
)

try:
    catalog_df = load_named_catalog(selected_catalog_key)
except (FileNotFoundError, CatalogValidationError) as exc:
    st.error(f"Selected shade catalog error: {exc}")
    st.stop()

SHADE_CATALOG_PATH = catalog_options[selected_catalog_key].path

st.caption(
    f"Selected catalog: {catalog_df.attrs.get('catalog_name', 'unknown')} | "
    f"{catalog_df.attrs.get('valid_count', len(catalog_df))} shades | "
    f"Source: {catalog_df.attrs.get('source', 'unknown')}"
)
st.warning(PUBLIC_CATALOG_LIMITATION)
st.caption("Local AI foundation shade recommender — upload a facial photo to begin.")

st.info(
    "For best results: use soft daylight, face camera directly, avoid filters, "
    "remove sunglasses/hats where possible, and keep cheeks and jawline visible."
)

st.caption(
    "Color handling is automatic. The system preserves original image color unless "
    "correction improves reliability without excessive color shift."
)
extraction_mode = "auto"

uploaded_file = st.file_uploader(
    "Upload a facial image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    lighting_quality = analyze_lighting_quality(image_rgb)
    correction_settings = correction_settings_for_lighting(lighting_quality)
    corrected_rgb, correction_notes = apply_mild_color_correction(image_rgb, **correction_settings)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_rgb, caption=f"{image.width}x{image.height}px", width=400)

    face_result = detect_face_landmarks(corrected_rgb)
    image_quality = analyze_image_quality(
        image_rgb, face_result.landmarks if face_result.success else None
    )

    for warning in face_result.warnings:
        st.warning(warning)

    if not face_result.success:
        st.error(face_result.error)
    else:
        with st.expander("Lighting correction notes"):
            for note in correction_notes:
                st.caption(note)

        with st.expander("Image Capture Quality"):
            st.caption("Image Capture Quality")
            st.metric("Overall capture quality", f"{image_quality.overall_score:.0f}/100")
            st.caption(f"Label: {image_quality.label}")
            metric_cols = st.columns(5)
            metric_values = [
                ("Blur", image_quality.blur_score),
                ("Exposure", image_quality.exposure_score),
                ("Face size", image_quality.face_size_score),
                ("Pose", image_quality.pose_score),
                ("Color cast", image_quality.color_cast_score),
            ]
            for metric_col, (label, value) in zip(metric_cols, metric_values):
                with metric_col:
                    st.metric(label, f"{value:.0f}/100")
            for warning in image_quality.warnings:
                st.warning(warning)
            for reason in image_quality.reasons:
                st.caption(reason)

        st.subheader("Lighting Quality")
        st.metric("Lighting quality score", f"{lighting_quality.score:.0%}")
        st.caption(lighting_quality.explanation)
        for warning in lighting_quality.warnings:
            st.warning(warning)

        masks = build_region_masks(corrected_rgb.shape, face_result.landmarks)
        extraction_selection = run_dual_extraction(
            image_rgb, corrected_rgb, masks, lighting_quality, extraction_mode
        )
        skin_result = extraction_selection.selected
        skin_result.extraction_quality_reasons.append(extraction_selection.reason)
        visualization_rgb = image_rgb if extraction_selection.selected_source == "original" else corrected_rgb
        visual_source_label = "Original image" if extraction_selection.selected_source == "original" else "Corrected image"

        with col2:
            st.subheader("Detected Face Landmarks")
            overlay = draw_face_landmarks(visualization_rgb, face_result.landmarks)
            caption = (
                f"{len(face_result.landmarks)} landmarks | displayed on {visual_source_label}. "
                "Face detection used the corrected preview only for landmark stability."
            )
            st.image(overlay, caption=caption, width=400)
            st.caption(f"Landmark visualization source: displayed on {visual_source_label}.")

        st.subheader("Skin Regions")
        combined_overlay = draw_all_region_masks(visualization_rgb, masks)
        st.image(
            combined_overlay,
            caption=f"Forehead / cheeks / jawline (combined) | displayed on {visual_source_label}",
            width=450,
        )
        st.caption(f"Region visualization source: displayed on {visual_source_label}.")

        region_cols = st.columns(4)
        region_labels = {
            "forehead": "Forehead",
            "left_cheek": "Left Cheek",
            "right_cheek": "Right Cheek",
            "jawline": "Jawline",
        }
        for col, (region_key, label) in zip(region_cols, region_labels.items()):
            with col:
                region_overlay = draw_region_mask(visualization_rgb, masks[region_key])
                pixel_count = int((masks[region_key] > 0).sum())
                st.image(region_overlay, caption=f"{label} ({pixel_count}px)", width=180)
                if pixel_count == 0:
                    st.caption("No usable pixels in this region.")

        for warning in skin_result.warnings:
            st.warning(warning)

        st.subheader("Extracted Skin Tone")
        swatch_col, detail_col = st.columns([1, 2])
        with swatch_col:
            if skin_result.success:
                swatch = make_skin_swatch(skin_result.rgb)
                st.image(swatch, caption=f"RGB {skin_result.rgb}", width=150)
            else:
                st.error("Could not extract a usable skin swatch.")
        with detail_col:
            source_label = (
                f"Auto-selected {extraction_selection.selected_source}"
                if extraction_selection.selection_mode == "auto"
                else ("Original image" if extraction_selection.selected_source == "original" else "Corrected image")
            )
            st.caption(f"Shade extraction source: {source_label}")
            st.caption(f"Selection reason: {extraction_selection.reason}")
            st.metric("Extraction quality score", f"{skin_result.quality_score:.0%}")
            lab_rounded = tuple(round(v, 1) for v in skin_result.lab)
            st.caption(f"Lab: {lab_rounded}")
            if skin_result.ita_degrees is not None:
                st.caption(f"Estimated ITA: {skin_result.ita_degrees:.1f} deg ({skin_result.ita_category})")
            st.caption(f"Estimated skin-depth category: {skin_result.depth_estimate or 'unknown'}")
            for region_name, region in skin_result.region_results.items():
                patch_note = (
                    f", {region.stable_patch_count} stable patches"
                    if region.stable_patch_count
                    else ", full-region fallback"
                    if region.patch_fallback_used and region.median_rgb is not None
                    else ""
                )
                st.caption(
                    f"{region_name.replace('_', ' ').title()}: "
                    f"{region.valid_pixel_count}/{region.total_pixel_count} valid px "
                    f"({region.status_label}{patch_note})"
                )

        st.subheader("Skin Extraction Summary")
        st.caption(build_skin_extraction_summary(skin_result, lighting_quality, extraction_selection))

        with st.expander("Per-Region Quality"):
            st.caption("Per-Region Quality")
            region_order = ["left_cheek", "right_cheek", "forehead", "jawline"]
            quality_rows = []
            for region_name in region_order:
                region = skin_result.region_results.get(region_name)
                if region is None:
                    continue
                reason_text = " ".join(region.quality_reasons[:2]) if region.quality_reasons else region.status_reason or "No additional caveats."
                warning_text = " ".join(region.quality_warnings[:2]) if region.quality_warnings else "None"
                quality_rows.append(
                    {
                        "Region": region_name.replace("_", " ").title(),
                        "Score": f"{region.quality_score:.0f}/100",
                        "Label": region.quality_label,
                        "Role": region.role,
                        "Reason": reason_text,
                        "Warnings": warning_text,
                    }
                )
            if quality_rows:
                st.table(quality_rows)
            else:
                st.caption("No per-region quality diagnostics available.")

        with st.expander("Color Correction Diagnostics"):
            st.caption("Color Correction Diagnostics")
            st.caption(f"Selected extraction source: {extraction_selection.selected_source}")
            st.caption(f"Selection mode: {extraction_selection.selection_mode}")
            st.caption(f"Selection reason: {extraction_selection.reason}")
            diag_cols = st.columns(3)
            with diag_cols[0]:
                st.image(make_skin_swatch(extraction_selection.original.rgb), caption="Original extracted swatch", width=110)
                st.caption(f"RGB: {extraction_selection.original.rgb}")
                st.caption(f"Lab: {tuple(round(v, 1) for v in extraction_selection.original.lab)}")
            with diag_cols[1]:
                st.image(make_skin_swatch(extraction_selection.corrected.rgb), caption="Corrected extracted swatch", width=110)
                st.caption(f"RGB: {extraction_selection.corrected.rgb}")
                st.caption(f"Lab: {tuple(round(v, 1) for v in extraction_selection.corrected.lab)}")
            with diag_cols[2]:
                st.image(make_skin_swatch(skin_result.rgb), caption="Final selected swatch", width=110)
                st.caption(f"RGB: {skin_result.rgb}")
                st.caption(f"Lab: {tuple(round(v, 1) for v in skin_result.lab)}")
            st.caption(f"RGB difference: {extraction_selection.rgb_difference:.1f}")
            st.caption(f"Lab difference: {extraction_selection.lab_difference:.1f}")
            st.caption(f"Chroma preservation score: {extraction_selection.chroma_preservation_score:.0%}")

        with st.expander("Region color diagnostics"):
            st.caption("Region color diagnostics")
            patch_diag = skin_result.patch_voting_diagnostics or {}
            st.markdown("**Patch Voting Summary**")
            st.caption(f"Patch voting used: {'yes' if patch_diag.get('used') else 'no'}")
            st.caption(f"Stable patches available: {patch_diag.get('stable_patches_available', 0)}")
            st.caption(f"Stable patches used: {patch_diag.get('stable_patches_used', 0)}")
            st.caption(f"Outlier patches rejected: {patch_diag.get('outlier_patches_rejected', 0)}")
            st.caption(f"Highlight patches rejected: {patch_diag.get('highlight_patches_rejected', 0)}")
            st.caption(f"Shadow patches rejected: {patch_diag.get('shadow_patches_rejected', 0)}")
            st.caption(f"Mid-tone patches used: {patch_diag.get('midtone_patches_used', 0)}")
            st.caption(f"Dominant/trusted region contribution: {patch_diag.get('dominant_region_contribution', 'none')}")
            if patch_diag.get("fallback_reason"):
                st.caption(f"Patch voting fallback: {patch_diag['fallback_reason']}")
            for region_name, region in skin_result.region_results.items():
                st.markdown(f"**{region_name.replace('_', ' ').title()}**")
                if region.median_rgb is not None:
                    diag_cols = st.columns([1, 3])
                    with diag_cols[0]:
                        st.image(make_skin_swatch(region.median_rgb), width=90)
                    with diag_cols[1]:
                        region_lab = tuple(round(v, 1) for v in region.median_lab)
                        st.caption(f"RGB: {region.median_rgb}")
                        st.caption(f"Lab: {region_lab}")
                        st.caption(f"Status: {region.status_label}")
                        st.caption(f"Quality score: {region.quality_score:.0f}/100")
                        st.caption(f"Quality label: {region.quality_label}")
                        st.caption(f"Role: {region.role}")
                        st.caption(f"Reliability score: {region.reliability_score:.0%}")
                        st.caption(f"Stable patches: {region.stable_patch_count}")
                        st.caption(f"Mid-tone patches used: {region.midtone_patch_count}")
                        st.caption(f"Highlight patches rejected: {region.highlight_patches_rejected}")
                        st.caption(f"Shadow patches rejected: {region.shadow_patches_rejected}")
                        st.caption(f"Shadow/highlight ratio: {region.shadow_highlight_ratio:.0%}")
                        if region_name == "jawline":
                            if region.weight_multiplier < 1.0:
                                st.caption(f"Jawline reduction reason: {region.downweight_reason}")
                            else:
                                st.caption("Jawline reduction reason: not reduced; reliable depth support.")
                        if region.makeup_influence_detected:
                            st.caption("possible makeup/highlight influence detected.")
                        if region.specular_highlight_detected:
                            st.caption("possible specular highlight influence detected.")
                        st.caption(
                            f"Valid pixels: {region.valid_pixel_count}/{region.total_pixel_count}"
                        )
                else:
                    st.caption(f"Status: {region.status_label}")
                    st.caption(f"Valid pixels: {region.valid_pixel_count}/{region.total_pixel_count}")
            st.markdown("**Final Blended Swatch**")
            if skin_result.success:
                st.image(make_skin_swatch(skin_result.rgb), width=90)
                st.caption(f"RGB: {skin_result.rgb}")
                st.caption(f"Lab: {tuple(round(v, 1) for v in skin_result.lab)}")
                st.caption(f"Final depth estimate: {skin_result.depth_estimate or 'unknown'}")

        st.subheader("Extraction Quality Reasons")
        if skin_result.extraction_quality_reasons:
            for reason in skin_result.extraction_quality_reasons:
                st.caption(reason)
        else:
            st.caption("No additional extraction quality caveats.")

        st.subheader("Regions Used for Final Estimate")
        included_labels = [n.replace("_", " ").title() for n in skin_result.included_region_names]
        excluded_labels = [n.replace("_", " ").title() for n in skin_result.excluded_region_names]
        st.markdown(
            f"**Included:** {', '.join(included_labels) if included_labels else 'None'}"
        )
        st.markdown(
            f"**Excluded:** {', '.join(excluded_labels) if excluded_labels else 'None'}"
        )
        for region_name, region in skin_result.region_results.items():
            if region.status_reason:
                st.caption(f"{region_name.replace('_', ' ').title()}: {region.status_reason}")

        if skin_result.success:
            st.subheader("Top 3 Shade Recommendations")
            try:
                catalog_df = load_shade_catalog(str(SHADE_CATALOG_PATH))
            except (FileNotFoundError, CatalogValidationError) as exc:
                st.error(f"Shade catalog error: {exc}")
            else:
                for w in catalog_df.attrs.get("warnings", []):
                    st.warning(w)

                matches = match_shades(np.array(skin_result.lab), catalog_df, top_k=TOP_K_SHADES)

                if not matches:
                    st.error("No shades available to recommend.")
                else:
                    if len(matches) < TOP_K_SHADES:
                        st.warning(
                            f"Catalog only has {len(matches)} usable shade(s); "
                            f"showing all available instead of {TOP_K_SHADES}."
                        )

                    quality_report = build_quality_report(
                        skin_result, face_result, matches, lighting_quality
                    )
                    matches = compute_confidence(matches, quality_report)

                    for w in quality_report.warnings:
                        st.warning(w)

                    shade_cols = st.columns(len(matches))
                    for col, match in zip(shade_cols, matches):
                        with col:
                            st.image(make_skin_swatch(match.rgb), width=150)
                            st.markdown(f"**#{match.rank}: {match.shade_name}**")
                            st.caption(f"Product: {match.product or 'unknown'}")
                            if match.product_variants:
                                variant_products = [
                                    v.get("product")
                                    for v in match.product_variants
                                    if v.get("product") and v.get("product") != match.product
                                ]
                                if variant_products:
                                    unique_variant_products = list(dict.fromkeys(variant_products))
                                    st.caption(
                                        "Also available in: "
                                        + ", ".join(unique_variant_products[:3])
                                    )
                            st.caption(f"{match.brand} · {match.hex}")
                            st.metric("Match confidence", f"{match.confidence:.0%}")
                            if match.confidence_breakdown:
                                st.caption(
                                    "Confidence breakdown: "
                                    f"color {match.confidence_breakdown['color_distance_contribution']:.2f}, "
                                    f"regions {match.confidence_breakdown['region_consistency_contribution']:.2f}, "
                                    f"pixels/patches {match.confidence_breakdown['valid_pixel_patch_contribution']:.2f}, "
                                    f"lighting {match.confidence_breakdown['lighting_quality_contribution']:.2f}, "
                                    f"separation {match.confidence_breakdown['top_shade_separation_contribution']:.2f}."
                                )
                            st.caption(f"Delta E (CIEDE2000): {match.delta_e:.2f}")
                            if match.undertone or match.depth:
                                st.caption(
                                    f"Undertone: {match.undertone or '—'} · Depth: {match.depth or '—'}"
                                )
                            st.caption(
                                "Depth sanity: "
                                f"extracted {match.extracted_depth or skin_result.depth_estimate or 'unknown'} | "
                                f"recommended {match.depth or 'unknown'} | "
                                f"status {match.depth_match_status}."
                            )
                            if match.depth_sanity_note:
                                st.caption(match.depth_sanity_note)
                            explanation = build_explanation(
                                match, skin_result, quality_report, match.rank, matches
                            )
                            st.caption(explanation)
else:
    st.info("Upload an image to see it displayed here.")
