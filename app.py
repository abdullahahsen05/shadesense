"""ShadeSense AI — Streamlit UI entrypoint.

This module contains presentation logic only. CV/matching logic lives in `src/`.
"""

import numpy as np
import streamlit as st
from src.analysis_pipeline import analyze_rgb_image
from src.config import APP_NAME, TOP_K_SHADES
from src.explanation import build_explanation
from src.extraction_summary import build_skin_extraction_summary
from src.image_io import open_rgb_image_with_metadata
from src.multi_photo_consensus import build_multi_photo_consensus
from src.shade_catalog import (
    ALL_BASE_SCOPE,
    FOUNDATION_ONLY_SCOPE,
    MOCK_CATALOG_KEY,
    PUBLIC_CATALOG_KEY,
    PUBLIC_CATALOG_LIMITATION,
    CatalogValidationError,
    catalog_definitions,
    filter_catalog_by_product_scope,
    load_default_catalog,
    load_named_catalog,
)
from src.ui_presentation import (
    APP_STYLES,
    build_region_decision_guidance,
    build_user_guidance,
    guidance_cards_html,
    readiness_display,
    rgb_to_hex,
    shade_strip_html,
)
from src.visualization import (
    draw_all_region_masks,
    draw_face_landmarks,
    draw_region_mask,
    make_skin_swatch,
)

st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown(APP_STYLES, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_default_catalog_cached():
    """Load and validate the default catalog once per source-code version."""
    return load_default_catalog()


@st.cache_data(show_spinner=False)
def _load_named_catalog_cached(catalog_key: str):
    """Reuse validated catalog data across Streamlit reruns."""
    return load_named_catalog(catalog_key)


def _render_primary_result(
    *,
    matches,
    skin_result,
    readiness,
    warning_groups,
) -> None:
    """Render the user decision before the detailed CV evidence."""
    display = readiness_display(readiness.state)
    st.markdown(
        f"""
        <div class="ss-result-hero {display.state}">
            <div class="ss-result-row">
                <span class="ss-status">{display.state}</span>
                <span class="ss-score">Readiness {readiness.score:.0f}/100</span>
            </div>
            <h2>{display.title}</h2>
            <p>{display.summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not matches:
        st.error("No catalog shades were available for this result.")
        return

    target_rgb = skin_result.foundation_target_rgb or skin_result.rgb
    strip_stops = [
        {
            "label": "Measured skin",
            "name": rgb_to_hex(skin_result.rgb),
            "hex": rgb_to_hex(skin_result.rgb),
        },
        {
            "label": "Match target",
            "name": rgb_to_hex(target_rgb),
            "hex": rgb_to_hex(target_rgb),
        },
    ]
    strip_stops.extend(
        {
            "label": f"Choice {match.rank}",
            "name": match.shade_name,
            "hex": match.hex,
        }
        for match in matches
    )

    st.subheader("Your shade shortlist")
    st.markdown(
        '<p class="ss-section-note">Closest catalog colors, ordered by perceptual '
        "match. Candidate confidence combines this shade’s color fit, shade-family "
        "stability, and catalog evidence. Capture readiness is reported separately "
        "and limits the maximum score. Ranking remains color-first, so a lower-ranked "
        "shade can have stronger stability and a slightly higher confidence score. "
        "Neither score is a probability.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(shade_strip_html(strip_stops), unsafe_allow_html=True)

    shade_cols = st.columns(len(matches))
    for col, match in zip(shade_cols, matches):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="ss-shade-color" style="background:{match.hex}"></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**#{match.rank} · {match.shade_name}**")
                st.caption(f"{match.brand} · {match.product or 'Foundation'}")
                candidate_value = getattr(match, "candidate_confidence", None)
                candidate_confidence = (
                    candidate_value
                    if candidate_value is not None
                    else match.confidence
                    if match.confidence is not None
                    else 0.0
                )
                confidence_label = (
                    "Strong shortlist"
                    if candidate_confidence >= 0.76
                    else "Useful comparison"
                    if candidate_confidence >= 0.60
                    else "Verify with another photo"
                )
                st.metric(
                    "Candidate confidence",
                    f"{candidate_confidence:.0%}",
                )
                color_fit_value = getattr(match, "color_fit_score", None)
                color_fit = (
                    color_fit_value
                    if color_fit_value is not None
                    else 0.0
                )
                family_stability = getattr(
                    match, "shade_family_stability_score", None
                )
                stability_text = (
                    f"{family_stability:.0%}"
                    if family_stability is not None
                    else "unavailable"
                )
                st.caption(
                    f"Color fit {color_fit:.0%} · "
                    f"Shade-family stability {stability_text} · "
                    f"Catalog evidence {match.catalog_quality_score:.0%}"
                )
                if (
                    getattr(match, "confidence_stability_source", "unavailable")
                    == "exact_product_fallback"
                ):
                    st.caption(
                        "Stability fallback: exact-product Top-3 stability was "
                        "used because shade-family stability was unavailable."
                    )
                st.caption(
                    f"{confidence_label} · Delta E {match.delta_e:.1f} · "
                    f"{match.depth or 'unknown depth'} · "
                    f"{match.undertone or 'unknown undertone'}"
                )

    capture_guidance = build_user_guidance(
        readiness.state,
        warning_groups,
        max_cards=4,
    )
    region_guidance = build_region_decision_guidance(
        skin_result.region_results
    )
    guidance = []
    if capture_guidance and capture_guidance[0].category == "Next step":
        guidance.append(capture_guidance.pop(0))
    guidance.extend(region_guidance)
    guidance.extend(
        card
        for card in capture_guidance
        if not (
            region_guidance
            and card.category == "Skin evidence"
        )
    )
    guidance = guidance[:4]
    st.subheader("What to know about this result")
    st.markdown(guidance_cards_html(guidance), unsafe_allow_html=True)


def _render_region_quality(skin_result) -> None:
    """Keep region reliability beside the region visualizations it explains."""
    with st.expander("Per-Region Quality"):
        st.caption("Per-Region Quality")
        quality_rows = []
        for region_name in ["left_cheek", "right_cheek", "forehead", "jawline"]:
            region = skin_result.region_results.get(region_name)
            if region is None:
                continue
            reason_text = (
                " ".join(region.quality_reasons[:2])
                if region.quality_reasons
                else region.status_reason or "No additional caveats."
            )
            warning_text = (
                " ".join(region.quality_warnings[:2])
                if region.quality_warnings
                else "None"
            )
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


st.markdown(
    """
    <div class="ss-kicker">Computer vision · foundation matching</div>
    <h1 class="ss-title">ShadeSense AI</h1>
    <p class="ss-lede">
        Turn one to three facial photos into a practical foundation shortlist,
        with clear capture guidance and inspectable color evidence.
    </p>
    """,
    unsafe_allow_html=True,
)
catalog_options = catalog_definitions()
try:
    default_catalog_key, default_catalog_df, default_catalog_warnings = (
        _load_default_catalog_cached()
    )
except (FileNotFoundError, CatalogValidationError) as exc:
    st.error(f"Could not load any shade catalog: {exc}")
    st.stop()

catalog_keys = [PUBLIC_CATALOG_KEY, MOCK_CATALOG_KEY]
with st.sidebar:
    st.markdown("### Match settings")
    selected_catalog_key = st.selectbox(
        "Shade catalog",
        catalog_keys,
        format_func=lambda key: catalog_options[key].name,
        index=catalog_keys.index(default_catalog_key),
    )

try:
    catalog_df = (
        default_catalog_df
        if selected_catalog_key == default_catalog_key
        else _load_named_catalog_cached(selected_catalog_key)
    )
except (FileNotFoundError, CatalogValidationError) as exc:
    st.error(f"Selected shade catalog error: {exc}")
    st.stop()

with st.sidebar:
    product_scope = st.selectbox(
        "Product scope",
        [FOUNDATION_ONLY_SCOPE, ALL_BASE_SCOPE],
        format_func=lambda value: (
            "Foundation only"
            if value == FOUNDATION_ONLY_SCOPE
            else "All base products"
        ),
        help=(
            "Foundation only prioritizes comparable liquid, stick, and powder "
            "foundations. All base products also includes cushions, BB/CC, and tints."
        ),
    )
try:
    recommendation_catalog_df = filter_catalog_by_product_scope(
        catalog_df,
        product_scope,
    )
except CatalogValidationError as exc:
    st.error(f"Product scope error: {exc}")
    st.stop()

with st.sidebar:
    st.caption(
        f"{catalog_df.attrs.get('catalog_name', 'Unknown catalog')} · "
        f"{len(recommendation_catalog_df)} eligible shades · "
        f"{catalog_df.attrs.get('valid_count', len(catalog_df))} validated rows"
    )
with st.sidebar:
    st.caption(PUBLIC_CATALOG_LIMITATION)

st.info(
    "Best capture: soft daylight, face the camera, avoid filters, and keep both "
    "cheeks plus the side jaw visible."
)

extraction_mode = "auto"

uploaded_files = st.file_uploader(
    "Choose one to three facial photos",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    key="face-photos",
)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.warning("Only the first three photos are analysed.")
        uploaded_files = uploaded_files[:3]

    decoded_images = []
    analyses = []
    rejected_uploads = []
    with st.spinner(f"Analysing {len(uploaded_files)} photo(s)..."):
        for uploaded_file in uploaded_files:
            try:
                decoded_image, color_metadata = open_rgb_image_with_metadata(
                    uploaded_file
                )
            except (OSError, ValueError) as exc:
                rejected_uploads.append(
                    (
                        uploaded_file.name,
                        "The file could not be decoded as a supported image "
                        f"({type(exc).__name__}).",
                    )
                )
                continue
            original_rgb = np.asarray(decoded_image)
            metadata = color_metadata.as_dict()
            metadata["neutral_card_calibrated"] = False
            image_analysis = analyze_rgb_image(
                original_rgb,
                recommendation_catalog_df,
                extraction_mode=extraction_mode,
                top_k=TOP_K_SHADES,
                image_color_metadata=metadata,
            )
            if not image_analysis.face_result.success:
                rejected_uploads.append(
                    (
                        uploaded_file.name,
                        image_analysis.error
                        or image_analysis.face_result.error
                        or "The upload did not pass human-portrait validation.",
                    )
                )
                continue
            decoded_images.append(
                (decoded_image, original_rgb, color_metadata, uploaded_file.name)
            )
            analyses.append(image_analysis)

    if rejected_uploads:
        st.subheader("Upload validation")
        for filename, reason in rejected_uploads:
            st.error(f"{filename}: {reason}")

    if not analyses:
        st.info(
            "No valid portrait is available for analysis. Upload one clear photo "
            "containing exactly one human face."
        )
        st.stop()

    consensus_result = (
        build_multi_photo_consensus(
            analyses,
            recommendation_catalog_df,
            top_k=TOP_K_SHADES,
        )
        if len(analyses) > 1
        else None
    )
    reference_index = (
        consensus_result.reference_index
        if consensus_result is not None
        and consensus_result.success
        and consensus_result.reference_index is not None
        else 0
    )
    image, image_rgb, image_color_metadata, image_name = decoded_images[
        reference_index
    ]
    analysis = analyses[reference_index]

    if consensus_result is not None:
        st.subheader("Multi-photo consensus")
        if consensus_result.success:
            st.markdown(
                f"**{consensus_result.readiness.state.title()} | "
                f"{len(consensus_result.retained_indices)} of "
                f"{len(analyses)} captures retained**"
            )
            st.caption(consensus_result.explanation)
            st.caption(
                "Cross-photo agreement: "
                f"{consensus_result.agreement_delta_e_p90:.1f} Delta E "
                "(90th percentile)."
            )
            st.image(
                make_skin_swatch(consensus_result.consensus_rgb),
                caption=(
                    "Consensus foundation target: RGB "
                    f"{consensus_result.consensus_rgb}, Lab "
                    f"{tuple(round(value, 1) for value in consensus_result.consensus_lab)}"
                ),
                width=150,
            )
            evidence_rows = [
                {
                    "Photo": evidence.capture_index + 1,
                    "File": decoded_images[evidence.capture_index][3],
                    "Included": "yes" if evidence.included else "no",
                    "Readiness": evidence.readiness_state,
                    "Distance from medoid": (
                        f"{evidence.distance_from_medoid:.1f} Delta E"
                    ),
                    "Quality weight": f"{evidence.weight:.0%}",
                }
                for evidence in consensus_result.evidence
            ]
            st.table(evidence_rows)
            for warning in consensus_result.warnings:
                st.warning(warning)
        else:
            for warning in consensus_result.warnings:
                st.error(warning)

    face_result = analysis.face_result

    for warning in face_result.warnings:
        st.warning(warning)

    if not face_result.success:
        st.subheader("Reference Image")
        st.image(
            image_rgb,
            caption=f"{image_name} | {image.width}x{image.height}px",
            width=400,
        )
        st.error(face_result.error)
    else:
        masks = analysis.masks
        mask_capture_diagnostics = analysis.mask_capture_diagnostics
        image_quality = analysis.image_quality
        lighting_quality = analysis.lighting_quality
        corrected_rgb = analysis.corrected_rgb
        correction_notes = analysis.correction_notes
        extraction_selection = analysis.extraction_selection
        skin_result = analysis.skin_result
        lighting_sensitivity = analysis.lighting_sensitivity
        capture_uncertainty = analysis.capture_uncertainty
        extraction_quality_report = analysis.extraction_quality_report
        recommendation_readiness = analysis.recommendation_readiness
        if consensus_result is not None and consensus_result.success:
            recommendation_readiness = consensus_result.readiness
        visualization_rgb = analysis.visualization_rgb
        visual_source_label = analysis.visual_source_label
        matches = (
            consensus_result.matches
            if consensus_result is not None and consensus_result.success
            else analysis.matches
        )
        quality_report = analysis.quality_report
        warning_groups = {
            "capture": image_quality.warnings,
            "lighting": lighting_quality.warnings,
            "mask": mask_capture_diagnostics["warnings"],
            "skin": skin_result.warnings,
            "extraction": extraction_quality_report["warnings"],
            "readiness": recommendation_readiness.warnings,
            "catalog": quality_report.warnings,
        }

        _render_primary_result(
            matches=matches,
            skin_result=skin_result,
            readiness=recommendation_readiness,
            warning_groups=warning_groups,
        )
        st.divider()
        st.subheader("Visual evidence")
        st.markdown(
            '<p class="ss-section-note">See which facial regions contributed to '
            "the result. Color overlays are diagnostic and are not applied to "
            "your photo.</p>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Reference photo**")
            st.image(
                image_rgb,
                caption=f"{image_name} · {image.width}×{image.height}px",
                width=400,
            )
            if analysis.analysis_scale < 1.0:
                st.caption(
                    "CV analysis used a proportional "
                    f"{analysis.image_rgb.shape[1]}×{analysis.image_rgb.shape[0]}px "
                    "copy for stable runtime."
                )
            if image_color_metadata.icc_converted_to_srgb:
                st.caption(
                    "Embedded color profile converted to sRGB: "
                    f"{image_color_metadata.source_profile_description}."
                )
            for warning in image_color_metadata.warnings:
                st.caption(warning)
        with col2:
            st.markdown("**Detected landmarks**")
            overlay = draw_face_landmarks(visualization_rgb, face_result.landmarks)
            st.image(
                overlay,
                caption=f"{len(face_result.landmarks)} landmarks · {visual_source_label}",
                width=400,
            )

        with st.expander("Lighting correction notes"):
            st.caption(
                f"Automatic extraction source: "
                f"{extraction_selection.selected_source}."
            )
            st.caption(
                f"Correction candidate shift: "
                f"{extraction_selection.lightness_shift:+.1f} L* and "
                f"{extraction_selection.undertone_shift:.1f} a*/b* units."
            )
            for flag in extraction_selection.safety_flags:
                st.warning(flag)
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
        if lighting_quality.using_face_regions:
            st.caption(
                f"Face highlight ratio {lighting_quality.face_highlight_ratio:.1%} · "
                f"shadow ratio {lighting_quality.face_shadow_ratio:.1%} · "
                f"left/right gap {lighting_quality.left_right_gap:.1f} · "
                f"central/lower gap {lighting_quality.central_lower_gap:.1f}."
            )
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

        _render_region_quality(skin_result)

        st.subheader("Extracted Skin Tone")
        swatch_col, detail_col = st.columns([1, 2])
        with swatch_col:
            if skin_result.success:
                swatch = make_skin_swatch(skin_result.rgb)
                st.image(swatch, caption=f"Measured visible skin tone: RGB {skin_result.rgb}", width=150)
                if skin_result.foundation_target_active and skin_result.foundation_target_rgb:
                    st.image(
                        make_skin_swatch(skin_result.foundation_target_rgb),
                        caption=f"Foundation target tone: RGB {skin_result.foundation_target_rgb}",
                        width=150,
                    )
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
            st.metric(
                "Raw region extraction score",
                f"{skin_result.quality_score:.0%}",
            )
            st.caption(
                "This is an internal region/pixel score. The formal Skin "
                "Extraction Quality below also includes capture, lighting, "
                "uncertainty, and cross-region stability."
            )
            lab_rounded = tuple(round(v, 1) for v in skin_result.lab)
            st.caption(f"Measured visible skin tone: RGB {skin_result.rgb}")
            st.caption(f"Measured visible Lab: {lab_rounded}")
            target_lab = tuple(round(v, 1) for v in (skin_result.foundation_target_lab or skin_result.lab))
            st.caption(f"Foundation target tone: RGB {skin_result.foundation_target_rgb or skin_result.rgb}")
            st.caption(f"Foundation target Lab: {target_lab}")
            if skin_result.foundation_target_active:
                st.caption(skin_result.foundation_target_reason)
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

        st.subheader("Skin Extraction Quality")
        st.markdown(
            f"**Skin Extraction Quality: {extraction_quality_report['overall_score']:.0f}/100 "
            f"- {extraction_quality_report['label'].title()}**"
        )
        if extraction_quality_report["reasons"]:
            st.caption(extraction_quality_report["reasons"][0])
        with st.expander("Skin Extraction Quality Details"):
            st.caption("Skin Extraction Quality Details")
            st.caption(
                "Skin Extraction Quality is separate from candidate confidence. "
                "It measures how reliable the extracted skin color is before catalog matching."
            )
            for name, score in extraction_quality_report["subscores"].items():
                st.caption(f"{name.replace('_', ' ').title()}: {score:.0f}/100")
            for reason in extraction_quality_report["reasons"][1:]:
                st.caption(reason)

        st.subheader("Capture & Extraction Readiness")
        st.markdown(
            f"**{recommendation_readiness.state.title()} · "
            f"{recommendation_readiness.score:.0f}/100**"
        )
        st.caption(recommendation_readiness.summary)
        readiness_cols = st.columns(3)
        with readiness_cols[0]:
            st.metric(
                "Capture readiness",
                f"{recommendation_readiness.capture_readiness_score:.0f}/100",
            )
        with readiness_cols[1]:
            st.metric(
                "Shade-family stability",
                f"{recommendation_readiness.shade_family_stability_score:.0f}/100",
            )
        with readiness_cols[2]:
            st.metric(
                "Exact-product stability",
                f"{recommendation_readiness.exact_product_stability_score:.0f}/100",
            )
        st.caption(
            "Capture readiness describes the photo and extraction. Shade-family "
            "stability describes color-family repeatability. Exact-product "
            "stability describes whether the same catalog SKU remains ranked first."
        )
        for reason in recommendation_readiness.reasons:
            st.caption(reason)

        with st.expander("Full diagnostic messages"):
            st.caption("Full diagnostic messages")
            st.caption(
                "Detailed model and capture messages are grouped here for auditability."
            )
            for source, messages in warning_groups.items():
                unique_messages = list(dict.fromkeys(messages))
                if not unique_messages:
                    continue
                st.markdown(f"**{source.replace('_', ' ').title()}**")
                for warning in unique_messages:
                    st.caption(f"• {warning}")

        st.subheader("Skin Extraction Summary")
        st.caption(build_skin_extraction_summary(skin_result, lighting_quality, extraction_selection))

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
            st.caption(
                f"Lightness shift: {extraction_selection.lightness_shift:+.1f} L*"
            )
            st.caption(
                "Undertone shift: "
                f"{extraction_selection.undertone_shift:.1f} a*/b* units"
            )
            st.caption(f"Chroma preservation score: {extraction_selection.chroma_preservation_score:.0%}")
            if extraction_selection.safety_flags:
                st.markdown("**Correction safety guard**")
                for flag in extraction_selection.safety_flags:
                    st.warning(flag)

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
            st.caption(f"Consensus method: {patch_diag.get('consensus_method', 'region fallback')}")
            st.caption(
                f"Perceptual outlier threshold: "
                f"{patch_diag.get('outlier_threshold_delta_e', 0):.1f} Delta E"
            )
            if patch_diag.get("adaptive_patch_sizes"):
                st.caption(
                    "Adaptive patch sizes: "
                    + ", ".join(str(value) for value in patch_diag["adaptive_patch_sizes"])
                    + " px"
                )
            region_contributions = patch_diag.get("region_contributions", {})
            if region_contributions:
                st.caption(
                    "Region contributions: "
                    + ", ".join(
                        f"{name.replace('_', ' ')} {value:.0%}"
                        for name, value in region_contributions.items()
                    )
                )
            uncertainty_diag = skin_result.uncertainty_diagnostics or {}
            st.markdown("**Extraction Uncertainty**")
            st.caption(
                f"Bootstrap samples: {uncertainty_diag.get('bootstrap_iterations', 0)}"
            )
            st.caption(
                "90th-percentile uncertainty radius: "
                f"{uncertainty_diag.get('delta_e_radius_p90', 12.0):.1f} Delta E"
            )
            st.caption(
                f"Bootstrap stability: {uncertainty_diag.get('stability_score', 45.0):.0f}/100"
            )
            if uncertainty_diag.get("l_interval_90"):
                lower_l, upper_l = uncertainty_diag["l_interval_90"]
                st.caption(f"90% L* interval: {lower_l:.1f}–{upper_l:.1f}")
            sensitivity_diag = skin_result.lighting_sensitivity_diagnostics or {}
            st.markdown("**Lighting Sensitivity**")
            st.caption(
                f"Sensitivity score: {sensitivity_diag.get('score', 0.0):.0f}/100"
            )
            st.caption(
                "90th-percentile perturbation shift: "
                f"{sensitivity_diag.get('delta_e_p90', 12.0):.1f} Delta E"
            )
            st.caption(
                "Usable perturbations: "
                f"{sensitivity_diag.get('successful_variants', 0)}/"
                f"{sensitivity_diag.get('attempted_variants', 0)}"
            )
            systematic_diag = skin_result.systematic_uncertainty_diagnostics or {}
            st.markdown("**Systematic Capture Uncertainty**")
            st.caption(
                "Capture-only radius: "
                f"{systematic_diag.get('systematic_radius', 0.0):.1f} Delta E"
            )
            st.caption(
                "Combined patch + capture radius: "
                f"{systematic_diag.get('total_delta_e_radius_p90', 12.0):.1f} Delta E"
            )
            st.caption(
                f"Capture stability: {systematic_diag.get('score', 0.0):.0f}/100"
            )
            if patch_diag.get("fallback_reason"):
                st.caption(f"Patch voting fallback: {patch_diag['fallback_reason']}")
            stability_diag = skin_result.stability_diagnostics or {}
            st.markdown("**Region Stability Analysis**")
            st.caption(f"Region stability score: {stability_diag.get('stability_score', 0):.0f}/100")
            st.caption(f"Region stability label: {stability_diag.get('stability_label', 'unknown')}")
            st.caption(
                "Region support mode: "
                f"{str(stability_diag.get('support_mode', 'agreement')).replace('_', ' ')}"
            )
            st.caption(
                "Most sensitive leave-one-out region: "
                f"{str(stability_diag.get('most_influential_region', 'none')).replace('_', ' ').title()}"
            )
            st.caption(f"Stability summary: {stability_diag.get('summary', 'not available')}")
            leave_one_out = stability_diag.get("leave_one_out_delta_e", {})
            adjusted_leave_one_out = stability_diag.get(
                "influence_adjusted_leave_one_out_delta_e",
                {},
            )
            region_contributions = stability_diag.get(
                "region_contributions",
                {},
            )
            if leave_one_out:
                st.table(
                    [
                        {
                            "Left out region": name.replace("_", " ").title(),
                            "Retained influence": (
                                f"{region_contributions.get(name, 0.0):.0%}"
                            ),
                            "Raw Delta E shift": f"{delta:.2f}",
                            "Influence-adjusted shift": (
                                f"{adjusted_leave_one_out.get(name, delta):.2f}"
                            ),
                        }
                        for name, delta in leave_one_out.items()
                    ]
                )
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
                            st.caption(
                                "Facial-hair texture score: "
                                f"{region.facial_hair_score:.2f}"
                            )
                            if region.excluded:
                                st.caption(
                                    "Side-jaw exclusion reason: "
                                    f"{region.exclusion_reason}"
                                )
                            elif region.weight_multiplier < 1.0:
                                st.caption(
                                    "Side-jaw reduction reason: "
                                    f"{region.downweight_reason}"
                                )
                            else:
                                st.caption(
                                    "Side-jaw status: not reduced; clean evidence "
                                    "corroborated cheek tone/depth."
                                )
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
                if skin_result.foundation_target_rgb and skin_result.foundation_target_lab:
                    st.markdown("**Foundation Target Swatch**")
                    st.image(make_skin_swatch(skin_result.foundation_target_rgb), width=90)
                    st.caption(f"RGB: {skin_result.foundation_target_rgb}")
                    st.caption(f"Lab: {tuple(round(v, 1) for v in skin_result.foundation_target_lab)}")
                    st.caption(f"Active for matching: {'yes' if skin_result.foundation_target_active else 'no'}")
                    st.caption(f"Reason: {skin_result.foundation_target_reason}")
                    target_diag = skin_result.foundation_target_diagnostics or {}
                    st.caption(f"Lower-face depth L*: {target_diag.get('lower_face_depth_l', 'unavailable')}")
                    st.caption(f"Central minus lower-face L*: {target_diag.get('central_minus_lower_l', 0):.1f}")
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
            st.subheader("Technical evidence for evaluators")
            st.markdown(
                '<p class="ss-section-note">The decision trail is preserved below '
                "without competing with the user-facing shortlist.</p>",
                unsafe_allow_html=True,
            )
            with st.expander("Detailed recommendation evidence"):
                st.caption("Detailed recommendation evidence")
                st.caption(
                    "CIEDE2000 remains the primary color distance. Distribution, "
                    "uncertainty, product type, and stability only moderate close matches."
                )
                st.caption(
                    "Candidate confidence = capture-readiness cap × normalized "
                    "candidate evidence (65% color fit, 25% shade-family stability, "
                    "10% catalog evidence). Missing factors are omitted and the "
                    "remaining weights are normalized. Color fit = exp(-distribution-"
                    "aware Delta E / 15). Ranking remains color-first."
                )
                detail_rows = []
                for match in matches:
                    candidate_value = getattr(
                        match, "candidate_confidence", None
                    )
                    color_fit_value = getattr(match, "color_fit_score", None)
                    family_stability = getattr(
                        match, "shade_family_stability_score", None
                    )
                    detail_rows.append(
                        {
                            "Rank": match.rank,
                            "Brand": match.brand,
                            "Product": match.product or "unknown",
                            "Shade": match.shade_name,
                            "Product type": match.product_type.replace("_", " "),
                            "HEX": match.hex,
                            "Delta E": f"{match.delta_e:.2f}",
                            "Distribution-aware Delta E": (
                                f"{match.distribution_delta_e:.2f}"
                                if match.distribution_delta_e is not None
                                else "n/a"
                            ),
                            "Candidate confidence": (
                                f"{(candidate_value or 0.0):.0%}"
                            ),
                            "Color fit": f"{(color_fit_value or 0.0):.0%}",
                            "Shade-family stability": (
                                f"{family_stability:.0%}"
                                if family_stability is not None
                                else "unavailable"
                            ),
                            "Stability source": (
                                getattr(
                                    match,
                                    "confidence_stability_source",
                                    "unavailable",
                                ).replace("_", " ")
                            ),
                            "Catalog evidence": f"{match.catalog_quality_score:.0%}",
                            "Top-1 stability": (
                                f"{match.recommendation_stability:.0%}"
                                if match.recommendation_stability is not None
                                else "n/a"
                            ),
                            "Top-3 stability": (
                                f"{match.top3_stability:.0%}"
                                if match.top3_stability is not None
                                else "n/a"
                            ),
                        }
                    )
                st.table(detail_rows)
                for match in matches:
                    st.markdown(f"**#{match.rank} · {match.brand} {match.shade_name}**")
                    st.caption(
                        build_explanation(
                            match,
                            skin_result,
                            quality_report,
                            match.rank,
                            matches,
                            readiness=recommendation_readiness,
                        )
                    )

else:
    st.markdown(
        """
        <div class="ss-result-hero caution">
            <div class="ss-result-row">
                <span class="ss-status">Start here</span>
                <span class="ss-score">1–3 photos</span>
            </div>
            <h2>Choose a clear facial photo</h2>
            <p>
                ShadeSense will measure reliable cheek, forehead, and side-jaw
                regions, then compare the result with the selected foundation catalog.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
