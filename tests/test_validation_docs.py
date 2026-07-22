from pathlib import Path


def test_validation_report_template_exists_with_required_cases():
    report = Path("docs/validation_report.md")

    assert report.exists()
    text = report.read_text(encoding="utf-8").casefold()

    required_cases = [
        "fair",
        "light",
        "medium",
        "tan",
        "deep",
        "rich-deep",
        "yellow indoor lighting",
        "strong shadow",
        "side-angle",
        "sunglasses",
        "shadow",
        "no-face",
        "multi-face",
    ]
    required_columns = [
        "image filename",
        "skin-depth group",
        "lighting condition",
        "face detected",
        "regions used",
        "regions excluded/down-weighted",
        "extraction quality",
        "match confidence",
        "top 3 shades",
        "visual mask quality",
        "result looks reasonable? yes/no",
        "notes",
    ]
    for required_text in required_cases + required_columns:
        assert required_text in text

    assert "do not fill results unless" in text


def test_demo_talking_points_contains_variation_handling_section():
    path = Path("docs/demo_talking_points.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8").casefold()

    assert "how the system handles real-world variation" in text
    assert "why true skin color is hard from a single selfie" in text
    for required_text in [
        "different skin tones",
        "deep-skin-safe adaptive filtering",
        "shadows",
        "facial highlights",
        "mild makeup",
        "jawline and forehead contamination",
        "color correction safeguard",
        "confidence reduction",
        "public catalog limitations",
        "camera processing",
        "white balance",
    ]:
        assert required_text in text
