from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_region_color_diagnostics_are_rendered_for_uploaded_face():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    image_path = Path("data/sample_images/face_astronaut.png")
    at.file_uploader[0].upload(image_path.name, image_path.read_bytes())
    at.run(timeout=90)

    page_text = "\n".join(
        [getattr(item, "value", "") for group in [at.subheader, at.markdown, at.caption] for item in group]
    )
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
