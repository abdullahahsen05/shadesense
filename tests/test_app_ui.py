from pathlib import Path

from streamlit.testing.v1 import AppTest


def _page_text(at):
    return "\n".join(
        [
            getattr(item, "value", "")
            for group in [at.subheader, at.markdown, at.caption, at.info, at.metric]
            for item in group
        ]
    )


def _run_uploaded_app():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    image_path = Path("data/sample_images/face_astronaut.png")
    at.file_uploader[0].upload(image_path.name, image_path.read_bytes())
    at.run(timeout=90)
    return at


def test_region_color_diagnostics_are_rendered_for_uploaded_face():
    at = _run_uploaded_app()
    page_text = _page_text(at)

    assert "Region color diagnostics" in page_text
    assert "Measured visible skin tone:" in page_text
    assert "Foundation target tone:" in page_text
    assert "Foundation target Lab:" in page_text
    assert "Forehead" in page_text
    assert "Left Cheek" in page_text
    assert "Right Cheek" in page_text
    assert "Jawline" in page_text
    assert "Final Blended Swatch" in page_text
    assert "Foundation Target Swatch" in page_text
    assert "Active for matching:" in page_text
    assert "Stable patches:" in page_text
    assert "Valid pixels:" in page_text
    assert "Reliability score:" in page_text
    assert "Shadow/highlight ratio:" in page_text
    assert "Highlight patches rejected:" in page_text
    assert "Shadow patches rejected:" in page_text
    assert "Mid-tone patches used:" in page_text
    assert "Patch Voting Summary" in page_text
    assert "Patch voting used:" in page_text
    assert "Outlier patches rejected:" in page_text
    assert "Dominant/trusted region contribution:" in page_text
    assert "Region Stability Analysis" in page_text
    assert "Region stability score:" in page_text
    assert "Most sensitive leave-one-out region:" in page_text
    assert any(
        getattr(metric, "label", "") == "Raw region extraction score"
        for metric in at.metric
    )
    assert "internal region/pixel score" in page_text
    assert "Stability summary:" in page_text
    assert (
        "Side-jaw exclusion reason:" in page_text
        or "Side-jaw reduction reason:" in page_text
        or "Side-jaw status:" in page_text
    )
    assert "Final depth estimate:" in page_text
    assert "Color Correction Diagnostics" in page_text
    assert "Selected extraction source:" in page_text
    assert "Shade extraction source:" in page_text
    assert "Selection reason:" in page_text
    assert "displayed on Original image" in page_text
    assert "Estimated ITA:" in page_text
    assert "Estimated skin-depth category:" in page_text
    assert "Skin Extraction Summary" in page_text
    assert "Skin Extraction Quality" in page_text
    assert "Skin Extraction Quality Details" in page_text
    assert "separate from candidate confidence" in page_text
    assert "Region Reliability:" in page_text
    assert "Extraction reliability:" in page_text
    assert "Trusted regions used:" in page_text
    assert "Per-Region Quality" in page_text
    assert "Quality score:" in page_text
    assert "Quality label:" in page_text
    assert "Role:" in page_text
    assert "Image Capture Quality" in page_text
    assert "Label:" in page_text
    assert "Blur metric" in page_text
    assert "Capture & Extraction Readiness" in page_text
    assert "Exact-product stability" in page_text
    assert "Bootstrap uncertainty radius" in page_text
    assert "Consensus method:" in page_text
    assert "Perceptual outlier threshold:" in page_text
    assert "Extraction Uncertainty" in page_text
    assert "Bootstrap samples:" in page_text
    assert "90th-percentile uncertainty radius:" in page_text
    assert "Lighting Sensitivity" in page_text
    assert "Sensitivity score:" in page_text
    assert "90th-percentile perturbation shift:" in page_text
    assert "Usable perturbations:" in page_text
    assert "Your shade shortlist" in page_text
    assert "Candidate confidence" in page_text
    assert "Color fit" in page_text
    assert "Shade-family stability" in page_text
    assert "Capture readiness is reported separately" in page_text
    assert "What to know about this result" in page_text
    assert (
        "One facial region was excluded" in page_text
        or "facial regions were excluded" in page_text
    )
    assert (
        "One region had reduced influence" in page_text
        or "Dense facial-hair texture was detected" in page_text
    )
    assert (
        "retained 12% influence" in page_text
        or "excluded from skin-color consensus" in page_text
    )
    assert "Visual evidence" in page_text
    assert "Technical evidence for evaluators" in page_text
    assert "Detailed recommendation evidence" in page_text
    assert "CIEDE2000 remains the primary color distance" in page_text
    assert "Full diagnostic messages" in page_text
    assert len(at.warning) < 8


def test_capture_guidance_text_exists():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    page_text = "\n".join(
        [
            getattr(item, "value", "")
            for group in [at.info, at.caption, at.markdown]
            for item in group
        ]
    )
    assert "Best capture" in page_text
    assert "soft daylight" in page_text
    assert "face the camera" in page_text
    assert "side jaw visible" in page_text
    assert "Choose a clear facial photo" in page_text
    assert len(at.file_uploader) == 1
    assert "neutral-card" not in page_text.lower()


def test_extraction_mode_controls_are_not_exposed_in_ui():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    page_text = _page_text(at)

    assert len(at.radio) == 0
    assert "Extraction debug mode" not in page_text
    assert "Force original extraction" not in page_text
    assert "Force corrected extraction" not in page_text


def test_per_region_quality_is_placed_beside_region_visuals():
    source = Path("app.py").read_text(encoding="utf-8")

    region_quality_position = source.index(
        "        _render_region_quality(skin_result)"
    )
    extracted_tone_position = source.index(
        '        st.subheader("Extracted Skin Tone")'
    )
    assert region_quality_position < extracted_tone_position


def test_two_uploaded_photos_render_consensus_diagnostics():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    image_path = Path("data/sample_images/face_astronaut.png")
    content = image_path.read_bytes()
    at.file_uploader[0].upload("capture-one.png", content)
    at.file_uploader[0].upload("capture-two.png", content)
    at.run(timeout=180)
    page_text = _page_text(at)

    assert not at.error
    assert "Multi-photo consensus" in page_text
    assert "2 of 2 captures retained" in page_text
    assert "Cross-photo agreement:" in page_text
    assert len(at.table) >= 1
