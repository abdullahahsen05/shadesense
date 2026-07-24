from pathlib import Path

from streamlit.testing.v1 import AppTest


def _page_text(at):
    return "\n".join(
        [
            getattr(item, "value", "")
            for group in [at.subheader, at.markdown, at.caption, at.info]
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
    assert "Most influential region:" in page_text
    assert "Stability summary:" in page_text
    assert "Jawline reduction reason:" in page_text
    assert "Final depth estimate:" in page_text
    assert "Color Correction Diagnostics" in page_text
    assert "Selected extraction source:" in page_text
    assert "Shade extraction source:" in page_text
    assert "Selection reason:" in page_text
    assert "displayed on Original image" in page_text
    assert "Estimated ITA:" in page_text
    assert "Estimated skin-depth category:" in page_text
    assert "Depth sanity:" in page_text
    assert "Skin Extraction Summary" in page_text
    assert "Skin Extraction Quality" in page_text
    assert "Skin Extraction Quality Details" in page_text
    assert "separate from shade Match confidence" in page_text
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
    assert "Recommendation Readiness" in page_text
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
    assert "Product type:" in page_text
    assert "catalog evidence" in page_text
    assert "Distribution-aware ranking Delta E:" in page_text
    assert "Bootstrap stability:" in page_text
    assert "Lighting sensitivity:" in page_text


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
    assert "For best results" in page_text
    assert "soft daylight" in page_text
    assert "face camera directly" in page_text
    assert "cheeks and jawline visible" in page_text
    assert "Color handling is automatic" in page_text
    assert "preserves original image color" in page_text


def test_extraction_mode_controls_are_not_exposed_in_ui():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    page_text = _page_text(at)

    assert len(at.radio) == 0
    assert "Extraction debug mode" not in page_text
    assert "Force original extraction" not in page_text
    assert "Force corrected extraction" not in page_text
