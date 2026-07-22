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


def _run_uploaded_app(extraction_mode=None):
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    if extraction_mode is not None:
        at.radio[0].set_value(extraction_mode)
        at.run(timeout=30)
    image_path = Path("data/sample_images/face_astronaut.png")
    at.file_uploader[0].upload(image_path.name, image_path.read_bytes())
    at.run(timeout=90)
    return at


def test_region_color_diagnostics_are_rendered_for_uploaded_face():
    at = _run_uploaded_app()
    page_text = _page_text(at)

    assert "Region color diagnostics" in page_text
    assert "Forehead" in page_text
    assert "Left Cheek" in page_text
    assert "Right Cheek" in page_text
    assert "Jawline" in page_text
    assert "Final Blended Swatch" in page_text
    assert "Stable patches:" in page_text
    assert "Valid pixels:" in page_text
    assert "Reliability score:" in page_text
    assert "Shadow/highlight ratio:" in page_text
    assert "Highlight patches rejected:" in page_text
    assert "Shadow patches rejected:" in page_text
    assert "Mid-tone patches used:" in page_text
    assert "Jawline reduction reason:" in page_text
    assert "Final depth estimate:" in page_text
    assert "Color Correction Diagnostics" in page_text
    assert "Selected extraction source:" in page_text
    assert "Shade extraction source:" in page_text
    assert "Selection reason:" in page_text
    assert "displayed on Original image" in page_text


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


def test_force_original_mode_uses_original_extraction_in_ui():
    at = _run_uploaded_app("Force original extraction")
    page_text = _page_text(at)

    assert "Shade extraction source: Original image" in page_text
    assert "debug mode forced original extraction" in page_text
    assert "displayed on Original image" in page_text


def test_force_corrected_mode_uses_corrected_extraction_in_ui():
    at = _run_uploaded_app("Force corrected extraction")
    page_text = _page_text(at)

    assert "Shade extraction source: Corrected image" in page_text
    assert "debug mode forced corrected extraction" in page_text
    assert "displayed on Corrected image" in page_text
